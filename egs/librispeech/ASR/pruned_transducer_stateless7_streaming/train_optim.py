#!/usr/bin/env python3
# Copyright    2021-2022    Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang, Mingshuang Luo, Zengwei Yao)

import argparse
import copy
import logging
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Union, List

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from decoder import Decoder
from joiner import Joiner
from lhotse.cut import CutSet
from lhotse.utils import fix_random_seed
from model import Transducer
from optim import Eden, ScaledAdam
from torch.nn.parallel import DistributedDataParallel as DDP
from zipformer import Zipformer

from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    create_grad_scaler,
    setup_logger,
    str2bool,
    torch_autocast,
)

# --- FPGA Hardware-Friendly Components ---

class RMSNorm(nn.Module):
    """FPGA-native alternative to LayerNorm: Removes mean calculation overhead."""
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # RMSNorm: x * 1/sqrt(mean(x^2) + eps)
        norm_x = x.pow(2).mean(-1, keepdim=True)
        return self.scale * x * torch.rsqrt(norm_x + self.eps)

def optimize_for_fpga(model: nn.Module) -> int:
    """Recursively swaps Icefall's custom activations and norms for FPGA-friendly ones."""
    count = 0
    # Catch Icefall's proprietary activations
    act_names = ['SwooshL', 'SwooshR', 'Swish', 'SiLU', 'GELU', 'Activation']
    # Catch Icefall's proprietary normalizations
    norm_names = ['BiasNorm', 'LayerNorm', 'BasicNorm']

    for name, module in model.named_modules():
        for child_name, child_module in list(module.named_children()):
            class_name = child_module.__class__.__name__
            
            # Replace Activations
            if class_name in act_names:
                setattr(module, child_name, nn.ReLU(inplace=True))
                count += 1
                
            # Replace Normalizations
            elif class_name in norm_names:
                # Dynamically find the dimension of Icefall's custom norm layers
                dim = 256 # fallback
                if hasattr(child_module, 'num_channels'):
                    dim = child_module.num_channels
                elif hasattr(child_module, 'normalized_shape'):
                    dim = child_module.normalized_shape[0]
                elif hasattr(child_module, 'bias') and child_module.bias is not None:
                    dim = child_module.bias.shape[0]
                elif hasattr(child_module, 'weight') and child_module.weight is not None:
                    dim = child_module.weight.shape[0]
                    
                setattr(module, child_name, RMSNorm(dim))
                count += 1
                
    return count

# --- Core Training Setup ---

def add_model_arguments(parser):
    parser.add_argument("--num-encoder-layers", type=str, default="2,2,3,2,2")
    parser.add_argument("--feedforward-dims", type=str, default="512,768,768,512,512")
    parser.add_argument("--nhead", type=str, default="4,4,4,4,4")
    parser.add_argument("--encoder-dims", type=str, default="192,192,192,192,192")
    parser.add_argument("--attention-dims", type=str, default="192,192,192,192,192")
    parser.add_argument("--encoder-unmasked-dims", type=str, default="192,192,192,192,192")
    parser.add_argument("--zipformer-downsampling-factors", type=str, default="1,2,4,8,2")
    parser.add_argument("--cnn-module-kernels", type=str, default="31,31,31,31,31")
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--joiner-dim", type=int, default=256)
    parser.add_argument("--short-chunk-size", type=int, default=50)
    parser.add_argument("--num-left-chunks", type=int, default=4)
    parser.add_argument("--decode-chunk-len", type=int, default=32)

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=12354)
    parser.add_argument("--num-epochs", type=int, default=45)
    parser.add_argument("--exp-dir", type=str, default="pruned_transducer_stateless7_streaming/exp_micro_fpga")
    parser.add_argument("--bpe-model", type=str, default="data/lang_bpe_500/bpe.model")
    parser.add_argument("--base-lr", type=float, default=0.045)
    parser.add_argument("--lr-epochs", type=float, default=4.5)
    parser.add_argument("--context-size", type=int, default=2)
    parser.add_argument("--prune-range", type=int, default=5)
    parser.add_argument("--lm-scale", type=float, default=0.25)
    parser.add_argument("--am-scale", type=float, default=0.0)
    parser.add_argument("--simple-loss-scale", type=float, default=0.5)
    parser.add_argument("--use-fp16", type=str2bool, default=True)
    add_model_arguments(parser)
    return parser

def run(rank, world_size, args):
    params = AttributeDict({
        "best_train_loss": float("inf"), "best_valid_loss": float("inf"),
        "batch_idx_train": 0, "log_interval": 50, "valid_interval": 3000,
        "feature_dim": 80, "subsampling_factor": 4, "warm_step": 2000,
        "env_info": get_env_info(),
    })
    params.update(vars(args))
    params.exp_dir = Path(params.exp_dir)
    fix_random_seed(42)
    
    if world_size > 1: setup_dist(rank, world_size, params.master_port)
    setup_logger(f"{params.exp_dir}/log/log-train")
    
    sp = spm.SentencePieceProcessor(); sp.load(params.bpe_model)
    params.blank_id, params.vocab_size = sp.piece_to_id("<blk>"), sp.get_piece_size()

    # Model Initialization
    def to_int_tuple(s): return tuple(map(int, s.split(",")))
    encoder = Zipformer(
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
    decoder = Decoder(params.vocab_size, params.decoder_dim, params.blank_id, params.context_size)
    joiner = Joiner(int(params.encoder_dims.split(",")[-1]), params.decoder_dim, params.joiner_dim, params.vocab_size)
    model = Transducer(encoder, decoder, joiner, int(params.encoder_dims.split(",")[-1]), params.decoder_dim, params.joiner_dim, params.vocab_size)

    # --- FPGA OPTIMIZATIONS ---
    logging.info(f"Applying FPGA Hardware Substitutions (ReLU & RMSNorm)...")
    replacements = optimize_for_fpga(model)
    logging.info(f"FPGA optimization complete. Replaced {replacements} modules.")
    
    # --- PARAMETER CALCULATOR ---
    num_param_encoder = sum(p.numel() for p in model.encoder.parameters())
    num_param_decoder = sum(p.numel() for p in model.decoder.parameters())
    num_param_joiner = sum(p.numel() for p in model.joiner.parameters())
    num_param_total = sum(p.numel() for p in model.parameters())

    logging.info("=" * 50)
    logging.info(f"Model Parameters Breakdown:")
    logging.info(f"Encoder:  {num_param_encoder / 1e6:.2f} M")
    logging.info(f"Decoder:  {num_param_decoder / 1e6:.2f} M")
    logging.info(f"Joiner:   {num_param_joiner / 1e6:.2f} M")
    logging.info(f"Total:    {num_param_total / 1e6:.2f} M")
    logging.info("=" * 50)
    # ----------------------------

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    # --- FIX ScaledAdam: Extract parameter names ---
    parameters_names = []
    for name, _ in model.named_parameters():
        parameters_names.append(name)
    
    optimizer = ScaledAdam(
        model.parameters(), 
        lr=params.base_lr, 
        clipping_scale=2.0,
        parameters_names=[parameters_names]
    )
    
    scheduler = Eden(optimizer, 5000, params.lr_epochs)
    scaler = create_grad_scaler(enabled=params.use_fp16)

    # Dataloader setup
    train_cuts = CutSet.from_file("data/fbank/bud500_cuts_train.jsonl.gz")
    valid_cuts = CutSet.from_file("data/fbank/bud500_cuts_dev.jsonl.gz")
    datamodule = LibriSpeechAsrDataModule(args)

    for epoch in range(1, params.num_epochs + 1):
        model.train()
        train_dl = datamodule.train_dataloaders(train_cuts)
        for batch_idx, batch in enumerate(train_dl):
            params.batch_idx_train += 1
            with torch_autocast(enabled=params.use_fp16):
                feature = batch["inputs"].to(device)
                feature_lens = batch["supervisions"]["num_frames"].to(device)
                y = k2.RaggedTensor(sp.encode(batch["supervisions"]["text"], out_type=int)).to(device)
                simple_loss, pruned_loss = model(x=feature, x_lens=feature_lens, y=y)
                loss = 0.5 * simple_loss + pruned_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            scheduler.step_batch(params.batch_idx_train)

            if batch_idx % params.log_interval == 0:
                logging.info(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint after each epoch
        save_checkpoint_impl(
            filename=params.exp_dir / f"epoch-{epoch}.pt",
            model=model, params=params, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )

if __name__ == "__main__":
    parser = get_parser(); LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args(); run(0, 1, args)
