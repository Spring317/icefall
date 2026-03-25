#!/usr/bin/env python3
# Copyright    2021-2022    Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                          Wei Kang,
#                                                          Mingshuang Luo,)
#                                                          Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from decoder import Decoder
from joiner import Joiner
from lhotse.cut import Cut, CutSet
from lhotse.utils import fix_random_seed
from model import Transducer
from optim import Eden, ScaledAdam
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from zipformer import Zipformer

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.err import raise_grad_scale_is_too_small_error
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    create_grad_scaler,
    setup_logger,
    str2bool,
    torch_autocast,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        model = model.module
    for module in model.modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count


def add_model_arguments(parser: argparse.ArgumentParser):
    # --- MODIFIED: Tiny Defaults for Zipformer ---
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,2,2,2,2",
        help="Number of zipformer encoder layers, comma separated.",
    )
    parser.add_argument(
        "--feedforward-dims",
        type=str,
        default="512,512,512,512,512",
        help="Feedforward dimension of the zipformer encoder layers.",
    )
    parser.add_argument(
        "--nhead",
        type=str,
        default="4,4,4,4,4",
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--encoder-dims",
        type=str,
        default="192,192,192,192,192",
        help="Embedding dimension in the zipformer encoder layers.",
    )
    parser.add_argument(
        "--attention-dims",
        type=str,
        default="128,128,128,128,128",
        help="Attention dimension.",
    )
    parser.add_argument(
        "--encoder-unmasked-dims",
        type=str,
        default="128,128,128,128,128",
        help="Unmasked dimensions.",
    )
    # ----------------------------------------------

    parser.add_argument(
        "--zipformer-downsampling-factors",
        type=str,
        default="1,2,4,8,2",
        help="Downsampling factor for each stack.",
    )
    parser.add_argument(
        "--cnn-module-kernels",
        type=str,
        default="31,31,31,31,31",
        help="Sizes of kernels in convolution modules",
    )
    parser.add_argument(
        "--decoder-dim", type=int, default=512, help="Decoder embedding dim."
    )
    parser.add_argument(
        "--joiner-dim", type=int, default=512, help="Joiner dimension."
    )
    parser.add_argument(
        "--short-chunk-size", type=int, default=50, help="Chunk length of dynamic training."
    )
    parser.add_argument(
        "--num-left-chunks", type=int, default=4, help="Left context chunks."
    )
    parser.add_argument(
        "--decode-chunk-len", type=int, default=32, help="Chunk size for decoding."
    )


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=12354)
    parser.add_argument("--tensorboard", type=str2bool, default=True)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--exp-dir", type=str, default="pruned_transducer_stateless7_streaming/exp_tiny")
    parser.add_argument("--bpe-model", type=str, default="data/lang_bpe_500/bpe.model")
    parser.add_argument("--base-lr", type=float, default=0.05)
    parser.add_argument("--lr-batches", type=float, default=5000)
    parser.add_argument("--lr-epochs", type=float, default=3.5)
    parser.add_argument("--context-size", type=int, default=2)
    parser.add_argument("--prune-range", type=int, default=5)
    parser.add_argument("--lm-scale", type=float, default=0.25)
    parser.add_argument("--am-scale", type=float, default=0.0)
    parser.add_argument("--simple-loss-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-diagnostics", type=str2bool, default=False)
    parser.add_argument("--inf-check", type=str2bool, default=False)
    parser.add_argument("--save-every-n", type=int, default=2000)
    parser.add_argument("--keep-last-k", type=int, default=30)
    parser.add_argument("--average-period", type=int, default=200)
    parser.add_argument("--use-fp16", type=str2bool, default=False)
    add_model_arguments(parser)
    return parser


def get_params() -> AttributeDict:
    return AttributeDict({
        "best_train_loss": float("inf"), "best_valid_loss": float("inf"),
        "best_train_epoch": -1, "best_valid_epoch": -1, "batch_idx_train": 0,
        "log_interval": 50, "reset_interval": 200, "valid_interval": 3000,
        "feature_dim": 80, "subsampling_factor": 4, "warm_step": 2000,
        "env_info": get_env_info(),
    })


def get_encoder_model(params: AttributeDict) -> nn.Module:
    def to_int_tuple(s: str): return tuple(map(int, s.split(",")))
    return Zipformer(
        num_features=params.feature_dim, output_downsampling_factor=2,
        zipformer_downsampling_factors=to_int_tuple(params.zipformer_downsampling_factors),
        encoder_dims=to_int_tuple(params.encoder_dims),
        attention_dim=to_int_tuple(params.attention_dims),
        encoder_unmasked_dims=to_int_tuple(params.encoder_unmasked_dims),
        nhead=to_int_tuple(params.nhead), feedforward_dim=to_int_tuple(params.feedforward_dims),
        cnn_module_kernels=to_int_tuple(params.cnn_module_kernels),
        num_encoder_layers=to_int_tuple(params.num_encoder_layers),
        num_left_chunks=params.num_left_chunks, short_chunk_size=params.short_chunk_size,
        decode_chunk_size=params.decode_chunk_len // 2,
    )


def get_transducer_model(params: AttributeDict) -> nn.Module:
    encoder = get_encoder_model(params)
    decoder = Decoder(vocab_size=params.vocab_size, decoder_dim=params.decoder_dim, blank_id=params.blank_id, context_size=params.context_size)
    joiner = Joiner(encoder_dim=int(params.encoder_dims.split(",")[-1]), decoder_dim=params.decoder_dim, joiner_dim=params.joiner_dim, vocab_size=params.vocab_size)
    return Transducer(encoder=encoder, decoder=decoder, joiner=joiner, encoder_dim=int(params.encoder_dims.split(",")[-1]), decoder_dim=params.decoder_dim, joiner_dim=params.joiner_dim, vocab_size=params.vocab_size)


def compute_loss(params, model, sp, batch, is_training):
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"].to(device)
    feature_lens = batch["supervisions"]["num_frames"].to(device)
    y = k2.RaggedTensor(sp.encode(batch["supervisions"]["text"], out_type=int)).to(device)
    with torch.set_grad_enabled(is_training):
        simple_loss, pruned_loss = model(x=feature, x_lens=feature_lens, y=y, prune_range=params.prune_range, am_scale=params.am_scale, lm_scale=params.lm_scale)
        s = params.simple_loss_scale
        simple_loss_scale = s if params.batch_idx_train >= params.warm_step else 1.0 - (params.batch_idx_train / params.warm_step) * (1.0 - s)
        pruned_loss_scale = 1.0 if params.batch_idx_train >= params.warm_step else 0.1 + 0.9 * (params.batch_idx_train / params.warm_step)
        loss = simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss
    info = MetricsTracker()
    info["frames"] = (feature_lens // params.subsampling_factor).sum().item()
    info["loss"], info["simple_loss"], info["pruned_loss"] = loss.detach().cpu().item(), simple_loss.detach().cpu().item(), pruned_loss.detach().cpu().item()
    return loss, info


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Optional[Dict[str, Any]]:
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"
    saved_params = load_checkpoint(
        filename, model=model, model_avg=model_avg, optimizer=optimizer, scheduler=scheduler, scaler=scaler
    )
    keys = ["best_train_epoch", "best_valid_epoch", "batch_idx_train", "best_train_loss", "best_valid_loss"]
    for k in keys:
        if k in saved_params: params[k] = saved_params[k]
    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    rank: int = 0,
) -> None:
    if rank != 0: return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename, model=model, model_avg=model_avg, params=params,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler, rank=rank,
    )
    if params.best_train_epoch == params.cur_epoch:
        copyfile(src=filename, dst=params.exp_dir / "best-train-loss.pt")
    if params.best_valid_epoch == params.cur_epoch:
        copyfile(src=filename, dst=params.exp_dir / "best-valid-loss.pt")


def train_one_epoch(params, model, optimizer, scheduler, sp, train_dl, valid_dl, scaler, model_avg=None, rank=0):
    model.train()
    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        with torch_autocast(enabled=params.use_fp16):
            loss, loss_info = compute_loss(params, model, sp, batch, True)
        scaler.scale(loss).backward()
        set_batch_count(model, params.batch_idx_train)
        scheduler.step_batch(params.batch_idx_train)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

        if rank == 0 and params.batch_idx_train % params.average_period == 0:
            update_averaged_model(params, model, model_avg)
        
        if rank == 0 and params.batch_idx_train % params.save_every_n == 0:
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir, global_batch_idx=params.batch_idx_train,
                model=model, model_avg=model_avg, params=params,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler, rank=rank,
            )
            remove_checkpoints(out_dir=params.exp_dir, topk=params.keep_last_k, rank=rank)

        if batch_idx % params.log_interval == 0:
            logging.info(f"Epoch {params.cur_epoch}, batch {batch_idx}, loss[{loss_info}], lr: {scheduler.get_last_lr()[0]:.2e}")


def run(rank, world_size, args):
    params = get_params(); params.update(vars(args))
    params.exp_dir = Path(params.exp_dir)
    fix_random_seed(params.seed)
    if world_size > 1: setup_dist(rank, world_size, params.master_port)
    setup_logger(f"{params.exp_dir}/log/log-train")
    
    sp = spm.SentencePieceProcessor(); sp.load(params.bpe_model)
    params.blank_id, params.vocab_size = sp.piece_to_id("<blk>"), sp.get_piece_size()

    model = get_transducer_model(params)
    
    enc_p = sum(p.numel() for p in model.encoder.parameters())
    dec_p = sum(p.numel() for p in model.decoder.parameters())
    joi_p = sum(p.numel() for p in model.joiner.parameters())
    tot_p = enc_p + dec_p + joi_p
    logging.info("="*40)
    logging.info(f"TINY ARCHITECTURE: {tot_p/1e6:.2f}M | Enc: {enc_p/1e6:.2f}M | Dec: {dec_p/1e6:.2f}M | Joi: {joi_p/1e6:.2f}M")
    logging.info("="*40)

    model_avg = copy.deepcopy(model).to(torch.float64) if rank == 0 else None
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    if world_size > 1: model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    parameters_names = [[name for name, _ in model.named_parameters()]]
    optimizer = ScaledAdam(model.parameters(), lr=params.base_lr, clipping_scale=2.0, parameters_names=parameters_names)
    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)
    scaler = create_grad_scaler(enabled=params.use_fp16)

    load_checkpoint_if_available(params=params, model=model, model_avg=model_avg, optimizer=optimizer, scheduler=scheduler, scaler=scaler)

    librispeech = LibriSpeechAsrDataModule(args)
    train_cuts = CutSet.from_file("data/fbank/bud500_cuts_train.jsonl.gz")
    valid_cuts = CutSet.from_file("data/fbank/bud500_cuts_dev.jsonl.gz")
    train_dl, valid_dl = librispeech.train_dataloaders(train_cuts), librispeech.valid_dataloaders(valid_cuts)

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        params.cur_epoch = epoch
        train_dl.sampler.set_epoch(epoch - 1)
        train_one_epoch(params, model, optimizer, scheduler, sp, train_dl, valid_dl, scaler, model_avg=model_avg, rank=rank)
        save_checkpoint(params=params, model=model, model_avg=model_avg, optimizer=optimizer, scheduler=scheduler, scaler=scaler, rank=rank)


def main():
    parser = get_parser(); LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args(); args.exp_dir = Path(args.exp_dir); args.enable_musan = False
    if args.world_size > 1: mp.spawn(run, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else: run(rank=0, world_size=1, args=args)

if __name__ == "__main__":
    main()
