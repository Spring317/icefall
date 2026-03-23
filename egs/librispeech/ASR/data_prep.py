#!/usr/bin/env python3
# Copyright 2025 (Standard Icefall/Lhotse Preprocessing)

import argparse
import logging
import os
import io
import shutil
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio
import sentencepiece as spm

# ADDED: load_from_disk
from datasets import load_dataset, Audio, load_from_disk 

from lhotse import (
    Recording, RecordingSet, 
    SupervisionSegment, SupervisionSet, 
    CutSet, Fbank, FbankConfig, LilcomFilesWriter,
    MonoCut, Features
)
from lhotse.utils import compute_num_samples

# Setup Logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", 
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_args():
    parser = argparse.ArgumentParser(description="Prepare Bud500 Dataset for Icefall")
    parser.add_argument("--output-dir", type=Path, default="data", help="Root output directory")
    parser.add_argument("--vocab-size", type=int, default=4000, help="Size of BPE vocabulary")
    parser.add_argument("--overwrite", action="store_true", help="Force re-processing even if output exists")
    
    # ADDED: Dataset path argument defaulting to your local directory
    parser.add_argument("--dataset-path", type=str, default="/home/datasets/viet_bud500", 
                        help="Local path to the saved Hugging Face dataset")
    return parser.parse_args()

def process_dataset_in_memory(args):
    """
    Loads local data, extracts features in memory, and builds CutSets.
    Merges Manifest Creation + Feature Extraction into one efficient pass.
    """
    logging.info(f"Step 1: Loading Bud500 from local disk at {args.dataset_path}...")
    
    # CHANGED: Load the pre-processed arrow files instantly from disk
    hf_dataset = load_from_disk(args.dataset_path)
    
    # Cast to Audio(decode=False) to handle raw bytes/paths manually
    hf_dataset = hf_dataset.cast_column("audio", Audio(decode=False))
    
    splits_map = {
        "train": "train",
        "validation": "dev",
        "test": "test"
    }
    
    # Prepare directories
    manifest_dir = args.output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    fbank_dir = args.output_dir / "fbank"
    fbank_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Feature Extractor (Standard Icefall 80-mel Fbank)
    extractor = Fbank(FbankConfig(num_mel_bins=80))
    target_sampling_rate = 16000
    
    generated_cuts = {}

    for split, icefall_split in splits_map.items():
        cut_set_path = fbank_dir / f"bud500_cuts_{icefall_split}.jsonl.gz"
        
        # Check if output exists
        if cut_set_path.exists() and not args.overwrite:
            logging.info(f"Split '{icefall_split}' already exists at {cut_set_path}. Skipping.")
            try:
                generated_cuts[icefall_split] = CutSet.from_file(cut_set_path)
            except Exception as e:
                logging.warning(f"Failed to load existing cutset for {icefall_split}: {e}. Consider using --overwrite.")
            continue

        if split not in hf_dataset:
            logging.warning(f"Split {split} not found in local dataset, skipping.")
            continue
            
        logging.info(f"Processing split: {split} -> {icefall_split}")
        dataset_part = hf_dataset[split]
        
        cuts = []
        
        storage_path = fbank_dir / f"feats_{icefall_split}"
        
        with LilcomFilesWriter(storage_path) as writer:
            for i, item in tqdm(enumerate(dataset_part), total=len(dataset_part), desc=f"Extracting {icefall_split}"):
                unique_id = f"bud500_{icefall_split}_{i:06d}"
                
                try:
                    # --- Robust Audio Loading ---
                    audio_info = item["audio"]
                    audio_path = audio_info.get("path", None)
                    audio_bytes = audio_info.get("bytes", None)
                    
                    waveform = None
                    sample_rate = None

                    if audio_path is not None:
                        waveform, sample_rate = torchaudio.load(audio_path)
                    elif audio_bytes is not None:
                        buffer = io.BytesIO(audio_bytes)
                        waveform, sample_rate = torchaudio.load(buffer)
                    else:
                        logging.warning(f"Audio sample {unique_id} has no path or bytes, skipping!")
                        continue

                    if waveform.shape[0] > 1:
                         waveform = waveform[0:1, :] 

                    if sample_rate != target_sampling_rate:
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=sample_rate,
                            new_freq=target_sampling_rate
                        )
                        waveform = resampler(waveform)
                        sample_rate = target_sampling_rate
                    
                    audio_array = waveform.squeeze().numpy()
                    
                    # 2. Compute Features
                    features = extractor.extract(audio_array, sample_rate)
                    
                    # 3. Write Features to Disk
                    storage_key = writer.write(unique_id, features)
                    
                    # 4. Create Supervision
                    text = item.get("transcription", "")
                    duration = features.shape[0] * extractor.frame_shift
                    
                    sup = SupervisionSegment(
                        id=unique_id,
                        recording_id=unique_id,
                        start=0.0,
                        duration=duration,
                        channel=0,
                        text=text,
                        language="vi"
                    )
                    
                    # Create Features Manifest
                    features_manifest = Features(
                        type="fbank",
                        num_frames=features.shape[0],
                        num_features=features.shape[1],
                        frame_shift=extractor.frame_shift,
                        sampling_rate=target_sampling_rate,
                        start=0,
                        duration=duration,
                        storage_type=writer.name,
                        storage_path=str(writer.storage_path),
                        storage_key=storage_key,
                        channels=0, # FIX 1: Explicit Channel 0
                        recording_id=unique_id # FIX 2: Pass recording_id here instead of MonoCut
                    )

                    # 5. Create Cut
                    cut = MonoCut(
                        id=unique_id,
                        start=0.0,
                        duration=duration,
                        channel=0,
                        supervisions=[sup],
                        features=features_manifest
                    )

                    cuts.append(cut)
                    
                except Exception as e:
                    logging.error(f"Failed to process {unique_id}: {e}")
                    continue

        # Save CutSet
        logging.info(f"Saving CutSet for {icefall_split}...")
        cut_set = CutSet.from_cuts(cuts)
        cut_set_path = fbank_dir / f"bud500_cuts_{icefall_split}.jsonl.gz"
        cut_set.to_file(cut_set_path)
        
        generated_cuts[icefall_split] = cut_set

    return generated_cuts

def train_bpe(args, train_cut_set):
    """
    Train SentencePiece BPE model.
    """
    logging.info(f"Step 2: Preparing BPE Model (Vocab Size: {args.vocab_size})...")
    
    lang_dir = args.output_dir / f"lang_bpe_{args.vocab_size}"
    lang_dir.mkdir(parents=True, exist_ok=True)
    
    script_dir = Path(__file__).resolve().parent
    local_bpe_model = script_dir / "bpe.model"
    local_bpe_vocab = script_dir / "bpe.vocab"

    if local_bpe_model.exists() and local_bpe_vocab.exists():
        logging.info(f"Found existing BPE model in {script_dir}. Copying to {lang_dir}...")
        shutil.copy(local_bpe_model, lang_dir / "bpe.model")
        shutil.copy(local_bpe_vocab, lang_dir / "bpe.vocab")
        return

    model_prefix = lang_dir / "bpe"
    model_path = lang_dir / "bpe.model"
    if model_path.exists() and not args.overwrite:
        logging.info(f"BPE model already exists at {model_path}. Skipping training.")
        return

    txt_file = lang_dir / "transcripts.txt"
    logging.info("Extracting text...")
    
    with open(txt_file, "w", encoding="utf-8") as f:
        for cut in train_cut_set:
            text = " ".join(s.text for s in cut.supervisions)
            f.write(text + "\n")
            
    logging.info(f"Training SentencePiece model: {model_prefix}")
    
    spm.SentencePieceTrainer.train(
        input=str(txt_file),
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        input_sentence_size=10000000,
        user_defined_symbols=["<blk>"],
        pad_id=2,
        unk_id=1,
        bos_id=-1,
        eos_id=-1
    )
    
    logging.info("BPE Training Complete.")

def main():
    args = get_args()
    cut_sets = process_dataset_in_memory(args)
    if "train" in cut_sets:
        train_bpe(args, cut_sets["train"])
    else:
        train_bpe(args, []) 

    logging.info("=" * 50)
    logging.info(f"Preparation Done! Data is in: {args.output_dir}")
    logging.info("=" * 50)

if __name__ == "__main__":
    main()
