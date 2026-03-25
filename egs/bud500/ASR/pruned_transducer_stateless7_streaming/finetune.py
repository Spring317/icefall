#!/usr/bin/env python3
# Copyright    2021-2022  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,)
#                                                       Zengwei Yao)
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
from lhotse.dataset.sampling.base import CutSampler
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
    parser.add_argument("--num-encoder-layers", type=str, default="2,4,3,2,4", help="Number of zipformer encoder layers, comma separated.")
    parser.add_argument("--feedforward-dims", type=str, default="1024,1024,2048,2048,1024", help="Feedforward dimension of the zipformer encoder layers.")
    parser.add_argument("--nhead", type=str, default="8,8,8,8,8", help="Number of attention heads.")
    parser.add_argument("--encoder-dims", type=str, default="384,384,384,384,384", help="Embedding dimension in the 2 blocks of zipformer encoder layers.")
    parser.add_argument("--attention-dims", type=str, default="192,192,192,192,192", help="Attention dimension.")
    parser.add_argument("--encoder-unmasked-dims", type=str, default="256,256,256,256,256", help="Unmasked dimensions in the encoders.")
    parser.add_argument("--zipformer-downsampling-factors", type=str, default="1,2,4,8,2", help="Downsampling factor.")
    parser.add_argument("--cnn-module-kernels", type=str, default="31,31,31,31,31", help="Sizes of kernels.")
    parser.add_argument("--decoder-dim", type=int, default=512, help="Embedding dimension in the decoder model.")
    parser.add_argument("--joiner-dim", type=int, default=512, help="Dimension used in the joiner model.")
    parser.add_argument("--short-chunk-size", type=int, default=50, help="Chunk length of dynamic training.")
    parser.add_argument("--num-left-chunks", type=int, default=4, help="How many left context can be seen.")
    parser.add_argument("--decode-chunk-len", type=int, default=32, help="The chunk size for decoding.")

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--world-size", type=int, default=1, help="Number of GPUs for DDP training.")
    parser.add_argument("--master-port", type=int, default=12354, help="Master port to use for DDP training.")
    parser.add_argument("--tensorboard", type=str2bool, default=True, help="Should various information be logged in tensorboard.")
    parser.add_argument("--num-epochs", type=int, default=30, help="Number of epochs to train.")
    parser.add_argument("--start-epoch", type=int, default=1, help="Resume training from this epoch.")
    parser.add_argument("--start-batch", type=int, default=0, help="If positive, load checkpoint from this batch.")
    parser.add_argument("--exp-dir", type=str, default="pruned_transducer_stateless7_streaming/exp_finetune", help="The experiment dir.")
    
    # --- MODIFIED: Fine-tuning argument ---
    parser.add_argument("--finetune-checkpoint", type=str, default=None, help="Path to a pretrained checkpoint to initialize weights from (resets optimizer).")
    # --------------------------------------

    parser.add_argument("--bpe-model", type=str, default="data/lang_bpe_500/bpe.model", help="Path to the BPE model")
    parser.add_argument("--base-lr", type=float, default=0.05, help="The base learning rate.")
    parser.add_argument("--lr-batches", type=float, default=5000, help="Number of steps that affects how rapidly the learning rate decreases.")
    parser.add_argument("--lr-epochs", type=float, default=3.5, help="Number of epochs that affects how rapidly the learning rate decreases.")
    parser.add_argument("--context-size", type=int, default=2, help="The context size in the decoder. 1 means bigram; 2 means tri-gram")
    parser.add_argument("--prune-range", type=int, default=5, help="The prune range for rnnt loss.")
    parser.add_argument("--lm-scale", type=float, default=0.25, help="The scale to smooth the loss with lm.")
    parser.add_argument("--am-scale", type=float, default=0.0, help="The scale to smooth the loss with am.")
    parser.add_argument("--simple-loss-scale", type=float, default=0.5, help="Simple loss scale.")
    parser.add_argument("--seed", type=int, default=42, help="The seed for random generators.")
    parser.add_argument("--print-diagnostics", type=str2bool, default=False, help="Accumulate stats on activations.")
    parser.add_argument("--inf-check", type=str2bool, default=False, help="Add hooks to check for infinite module outputs.")
    parser.add_argument("--save-every-n", type=int, default=2000, help="Save checkpoint after processing this number of batches.")
    parser.add_argument("--keep-last-k", type=int, default=30, help="Only keep this number of checkpoints on disk.")
    parser.add_argument("--average-period", type=int, default=200, help="Update the averaged model after processing this number of batches.")
    parser.add_argument("--use-fp16", type=str2bool, default=False, help="Whether to use half precision training.")
    add_model_arguments(parser)
    return parser

def get_params() -> AttributeDict:
    params = AttributeDict({
        "best_train_loss": float("inf"),
        "best_valid_loss": float("inf"),
        "best_train_epoch": -1,
        "best_valid_epoch": -1,
        "batch_idx_train": 0,
        "log_interval": 50,
        "reset_interval": 200,
        "valid_interval": 3000,
        "feature_dim": 80,
        "subsampling_factor": 4,
        "warm_step": 2000,
        "env_info": get_env_info(),
    })
    return params

def get_encoder_model(params: AttributeDict) -> nn.Module:
    def to_int_tuple(s: str): return tuple(map(int, s.split(",")))
    encoder = Zipformer(
        num_features=params.feature_dim,
        output_downsampling_factor=2,
        zipformer_downsampling_factors=to_int_tuple(params.zipformer_downsampling_factors),
        encoder_dims=to_int_tuple(params.encoder_dims),
        attention_dim=to_int_tuple(params.attention_dims),
        encoder_unmasked_dims=to_int_tuple(params.encoder_unmasked_dims),
        nhead=to_int_tuple(params.nhead),
        feedforward_dim=to_int_tuple(params.feedforward_dims),
        cnn_module_kernels=to_int_tuple(params.cnn_module_kernels),
        num_encoder_layers=to_int_tuple(params.num_encoder_layers),
        num_left_chunks=params.num_left_chunks,
        short_chunk_size=params.short_chunk_size,
        decode_chunk_size=params.decode_chunk_len // 2,
    )
    return encoder

def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(vocab_size=params.vocab_size, decoder_dim=params.decoder_dim, blank_id=params.blank_id, context_size=params.context_size)
    return decoder

def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(encoder_dim=int(params.encoder_dims.split(",")[-1]), decoder_dim=params.decoder_dim, joiner_dim=params.joiner_dim, vocab_size=params.vocab_size)
    return joiner

def get_transducer_model(params: AttributeDict) -> nn.Module:
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)
    model = Transducer(encoder=encoder, decoder=decoder, joiner=joiner, encoder_dim=int(params.encoder_dims.split(",")[-1]), decoder_dim=params.decoder_dim, joiner_dim=params.joiner_dim, vocab_size=params.vocab_size)
    return model

def load_checkpoint_if_available(params: AttributeDict, model: nn.Module, model_avg: nn.Module = None, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[LRSchedulerType] = None) -> Optional[Dict[str, Any]]:
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None
    assert filename.is_file(), f"{filename} does not exist!"
    saved_params = load_checkpoint(filename, model=model, model_avg=model_avg, optimizer=optimizer, scheduler=scheduler)
    keys = ["best_train_epoch", "best_valid_epoch", "batch_idx_train", "best_train_loss", "best_valid_loss"]
    for k in keys: params[k] = saved_params[k]
    if params.start_batch > 0:
        if "cur_epoch" in saved_params: params["start_epoch"] = saved_params["cur_epoch"]
    return saved_params

def save_checkpoint(params: AttributeDict, model: Union[nn.Module, DDP], model_avg: Optional[nn.Module] = None, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[LRSchedulerType] = None, sampler: Optional[CutSampler] = None, scaler: Optional["GradScaler"] = None, rank: int = 0) -> None:
    if rank != 0: return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(filename=filename, model=model, model_avg=model_avg, params=params, optimizer=optimizer, scheduler=scheduler, sampler=sampler, scaler=scaler, rank=rank)
    if params.best_train_epoch == params.cur_epoch:
        copyfile(src=filename, dst=params.exp_dir / "best-train-loss.pt")
    if params.best_valid_epoch == params.cur_epoch:
        copyfile(src=filename, dst=params.exp_dir / "best-valid-loss.pt")

def compute_loss(params: AttributeDict, model: Union[nn.Module, DDP], sp: spm.SentencePieceProcessor, batch: dict, is_training: bool) -> Tuple[Tensor, MetricsTracker]:
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)
    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)
    batch_idx_train = params.batch_idx_train
    warm_step = params.warm_step
    texts = batch["supervisions"]["text"]
    y = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(y).to(device)
    with torch.set_grad_enabled(is_training):
        simple_loss, pruned_loss = model(x=feature, x_lens=feature_lens, y=y, prune_range=params.prune_range, am_scale=params.am_scale, lm_scale=params.lm_scale)
        s = params.simple_loss_scale
        simple_loss_scale = s if batch_idx_train >= warm_step else 1.0 - (batch_idx_train / warm_step) * (1.0 - s)
        pruned_loss_scale = 1.0 if batch_idx_train >= warm_step else 0.1 + 0.9 * (batch_idx_train / warm_step)
        loss = simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss
    assert loss.requires_grad == is_training
    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()
    info["loss"] = loss.detach().cpu().item()
    info["simple_loss"] = simple_loss.detach().cpu().item()
    info["pruned_loss"] = pruned_loss.detach().cpu().item()
    return loss, info

def compute_validation_loss(params: AttributeDict, model: Union[nn.Module, DDP], sp: spm.SentencePieceProcessor, valid_dl: torch.utils.data.DataLoader, world_size: int = 1) -> MetricsTracker:
    model.eval()
    tot_loss = MetricsTracker()
    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(params=params, model=model, sp=sp, batch=batch, is_training=False)
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info
    if world_size > 1: tot_loss.reduce(loss.device)
    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value
    return tot_loss

def train_one_epoch(params: AttributeDict, model: Union[nn.Module, DDP], optimizer: torch.optim.Optimizer, scheduler: LRSchedulerType, sp: spm.SentencePieceProcessor, train_dl: torch.utils.data.DataLoader, valid_dl: torch.utils.data.DataLoader, scaler: "GradScaler", model_avg: Optional[nn.Module] = None, tb_writer: Optional[SummaryWriter] = None, world_size: int = 1, rank: int = 0) -> None:
    model.train()
    tot_loss = MetricsTracker()
    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])
        try:
            with torch_autocast(enabled=params.use_fp16):
                loss, loss_info = compute_loss(params=params, model=model, sp=sp, batch=batch, is_training=True)
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info
            scaler.scale(loss).backward()
            set_batch_count(model, params.batch_idx_train)
            scheduler.step_batch(params.batch_idx_train)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except:
            display_and_save_batch(batch, params=params, sp=sp)
            raise
        if params.print_diagnostics and batch_idx == 5: return
        if rank == 0 and params.batch_idx_train > 0 and params.batch_idx_train % params.average_period == 0:
            update_averaged_model(params=params, model_cur=model, model_avg=model_avg)
        if params.batch_idx_train > 0 and params.batch_idx_train % params.save_every_n == 0:
            save_checkpoint_with_global_batch_idx(out_dir=params.exp_dir, global_batch_idx=params.batch_idx_train, model=model, model_avg=model_avg, params=params, optimizer=optimizer, scheduler=scheduler, sampler=train_dl.sampler, scaler=scaler, rank=rank)
            remove_checkpoints(out_dir=params.exp_dir, topk=params.keep_last_k, rank=rank)
        if batch_idx % 100 == 0 and params.use_fp16:
            cur_grad_scale = scaler._scale.item()
            if cur_grad_scale < 1.0 or (cur_grad_scale < 8.0 and batch_idx % 400 == 0): scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01: logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05: raise_grad_scale_is_too_small_error(cur_grad_scale)
        if batch_idx % params.log_interval == 0:
            cur_lr = scheduler.get_last_lr()[0]
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0
            logging.info(f"Epoch {params.cur_epoch}, batch {batch_idx}, loss[{loss_info}], tot_loss[{tot_loss}], batch size: {batch_size}, lr: {cur_lr:.2e}, " + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else ""))
            if tb_writer is not None:
                tb_writer.add_scalar("train/learning_rate", cur_lr, params.batch_idx_train)
                loss_info.write_summary(tb_writer, "train/current_", params.batch_idx_train)
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16: tb_writer.add_scalar("train/grad_scale", cur_grad_scale, params.batch_idx_train)
        if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(params=params, model=model, sp=sp, valid_dl=valid_dl, world_size=world_size)
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB")
            if tb_writer is not None: valid_info.write_summary(tb_writer, "train/valid_", params.batch_idx_train)
    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss

def run(rank, world_size, args):
    params = get_params()
    params.update(vars(args))
    if params.full_libri is False: params.valid_interval = 1600
    fix_random_seed(params.seed)
    if world_size > 1: setup_dist(rank, world_size, params.master_port)
    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")
    if args.tensorboard and rank == 0: tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else: tb_writer = None
    device = torch.device("cpu")
    if torch.cuda.is_available(): device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")
    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()
    logging.info(params)
    logging.info("About to create model")
    model = get_transducer_model(params)
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")
    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0: model_avg = copy.deepcopy(model).to(torch.float64)
    assert params.start_epoch > 0, params.start_epoch
    
    # --- MODIFIED: Fine-tune checkpoint loading ---
    checkpoints = load_checkpoint_if_available(params=params, model=model, model_avg=model_avg)
    
    if checkpoints is None and params.finetune_checkpoint:
        logging.info(f"Finetuning: Loading pretrained model weights from {params.finetune_checkpoint}")
        if not Path(params.finetune_checkpoint).exists():
            raise FileNotFoundError(f"Finetune checkpoint not found: {params.finetune_checkpoint}")
        
        # Load the checkpoint manually
        checkpoint = torch.load(params.finetune_checkpoint, map_location="cpu")
        
        # Handle 'model' key usually found in icefall checkpoints
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            # Fallback if it's a raw state dict
            model.load_state_dict(checkpoint, strict=False)
            
        if rank == 0 and model_avg is not None:
             if "model_avg" in checkpoint:
                 model_avg.load_state_dict(checkpoint["model_avg"], strict=False)
             else:
                 model_avg.load_state_dict(model.state_dict())
        
        logging.info("Pretrained weights loaded. Optimizer/Scheduler reset for fine-tuning.")
    # ----------------------------------------------

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    parameters_names = []
    parameters_names.append([name_param_pair[0] for name_param_pair in model.named_parameters()])
    optimizer = ScaledAdam(model.parameters(), lr=params.base_lr, clipping_scale=2.0, parameters_names=parameters_names)
    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)
    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])
    if checkpoints and "scheduler" in checkpoints and checkpoints["scheduler"] is not None:
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])
    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(512)
        diagnostic = diagnostics.attach_diagnostics(model, opts)
    if params.inf_check: register_inf_check_hooks(model)
    librispeech = LibriSpeechAsrDataModule(args)
    assert not (params.mini_libri and params.full_libri), f"Cannot set both mini-libri and full-libri flags to True, now mini-libri {params.mini_libri} and full-libri {params.full_libri}"
    
    # --- MODIFIED: Load Combined Cuts ---
    logging.info("Loading Combined VLSP+VIVOS cuts...")
    try:
        # These paths must match what prepare_combined.sh output
        train_cuts = CutSet.from_file("data/fbank/combined_cuts_train.jsonl.gz")
        logging.info(f"Loaded train cuts: {len(train_cuts)}")
        
        valid_cuts = CutSet.from_file("data/fbank/combined_cuts_dev.jsonl.gz")
        logging.info(f"Loaded dev cuts: {len(valid_cuts)}")
    except Exception as e:
        logging.error(f"Failed to load cuts: {e}")
        raise
    
    # Channel Fix
    def fix_channel_mismatch(c: Cut):
        if c.has_features and c.features.channels != c.channel:
            c.features.channels = c.channel
        return c
    train_cuts = train_cuts.map(fix_channel_mismatch)
    valid_cuts = valid_cuts.map(fix_channel_mismatch)
    # ------------------------------------

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints: sampler_state_dict = checkpoints["sampler"]
    else: sampler_state_dict = None
    train_dl = librispeech.train_dataloaders(train_cuts, sampler_state_dict=sampler_state_dict)
    valid_dl = librispeech.valid_dataloaders(valid_cuts)
    scaler = create_grad_scaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])
    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)
        if tb_writer is not None: tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)
        params.cur_epoch = epoch
        train_one_epoch(params=params, model=model, model_avg=model_avg, optimizer=optimizer, scheduler=scheduler, sp=sp, train_dl=train_dl, valid_dl=valid_dl, scaler=scaler, tb_writer=tb_writer, world_size=world_size, rank=rank)
        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break
        save_checkpoint(params=params, model=model, model_avg=model_avg, optimizer=optimizer, scheduler=scheduler, sampler=train_dl.sampler, scaler=scaler, rank=rank)
    logging.info("Done!")
    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()

def display_and_save_batch(batch: dict, params: AttributeDict, sp: spm.SentencePieceProcessor) -> None:
    from lhotse.utils import uuid4
    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)
    supervisions = batch["supervisions"]
    features = batch["inputs"]
    logging.info(f"features shape: {features.shape}")
    y = sp.encode(supervisions["text"], out_type=int)
    num_tokens = sum(len(i) for i in y)
    logging.info(f"num tokens: {num_tokens}")

def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.enable_musan = False
    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1: mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else: run(rank=0, world_size=1, args=args)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
