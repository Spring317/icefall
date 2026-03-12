#!/usr/bin/env python3
# Copyright 2025 (Icefall/Lhotse Preprocessing for Combined Dataset)

import argparse
import logging
import os
import io
import shutil
import glob
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio
import sentencepiece as spm
from datasets import load_dataset, Audio
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

# Fix for protobuf environment issues
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def get_args():
    parser = argparse.ArgumentParser(description="Prepare Combined VLSP+VIVOS Dataset")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to the dataset root (containing train/test/validation folders)")
    parser.add_argument("--output-dir", type=Path, default="data", help="Root output directory")
    parser.add_argument("--vocab-size", type=int, default=4000, help="Size of BPE vocabulary")
    parser.add_argument("--overwrite", action="store_true", help="Force re-processing")
    return parser.parse_args()

def load_custom_arrow_dataset(dataset_dir: Path):
    """
    Loads the dataset by globbing .arrow files as requested.
    """
    logging.info(f"Scanning for arrow files in {dataset_dir}...")
    
    data_files = {}
    
    # Map typical folder names to split names
    # Adjust glob patterns if your folder structure differs slightly
    splits = {
        "train": "train",
        "validation": "validation", # Will map to 'dev' later
        "test": "test"
    }
    
    found_splits = False
    
    for folder_name, split_key in splits.items():
        folder_path = dataset_dir / folder_name
        if folder_path.exists():
            # Get all .arrow files in the directory
            arrow_files = sorted(glob.glob(str(folder_path / "*.arrow")))
            if arrow_files:
                data_files[split_key] = arrow_files
                logging.info(f"Found {len(arrow_files)} arrow files for split '{split_key}'")
                found_splits = True
            else:
                logging.warning(f"Directory {folder_path} exists but contains no .arrow files.")
        else:
            logging.warning(f"Directory {folder_path} not found.")

    if not found_splits:
        raise ValueError(f"No data found in {dataset_dir}. Structure expected: root/{{train,test,validation}}/*.arrow")

    logging.info("Loading dataset using 'arrow' loader...")
    # Load using the specific "arrow" format as requested
    dataset = load_dataset("arrow", data_files=data_files)
    
    return dataset

def process_dataset(args):
    dataset = load_custom_arrow_dataset(args.dataset_dir)
    
    # Cast to Audio(decode=False) to try and get raw bytes if possible.
    # If the arrow files were saved with decoded audio, this might check for compatibility.
    # However, 'arrow' loading often loads what's on disk. We will robustly handle bytes vs arrays below.
    try:
        logging.info("Attempting to cast audio column to raw bytes for efficiency...")
        dataset = dataset.cast_column("audio", Audio(decode=False))
    except Exception as e:
        logging.warning(f"Could not cast to Audio(decode=False): {e}. Proceeding with default format.")

    # Icefall Split Mapping
    splits_map = {
        "train": "train",
        "validation": "dev",
        "test": "test"
    }

    # Prepare directories
    fbank_dir = args.output_dir / "fbank"
    fbank_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Feature Extractor
    extractor = Fbank(FbankConfig(num_mel_bins=80))
    target_sampling_rate = 16000
    
    generated_cuts = {}

    for hf_split, icefall_split in splits_map.items():
        if hf_split not in dataset:
            logging.info(f"Split '{hf_split}' not in dataset. Skipping.")
            continue
            
        cut_set_path = fbank_dir / f"combined_cuts_{icefall_split}.jsonl.gz"
        
        if cut_set_path.exists() and not args.overwrite:
            logging.info(f"Split '{icefall_split}' already processed. Loading from disk.")
            generated_cuts[icefall_split] = CutSet.from_file(cut_set_path)
            continue

        logging.info(f"Processing split: {hf_split} -> {icefall_split}")
        dataset_part = dataset[hf_split]
        
        cuts = []
        storage_path = fbank_dir / f"feats_{icefall_split}"
        
        with LilcomFilesWriter(storage_path) as writer:
            for i, item in tqdm(enumerate(dataset_part), total=len(dataset_part), desc=f"Extracting {icefall_split}"):
                unique_id = f"combined_{icefall_split}_{i:07d}"
                
                try:
                    # --- Robust Audio Loading ---
                    audio_info = item["audio"]
                    
                    waveform = None
                    sample_rate = None

                    # Case 1: Raw Bytes (Preferred)
                    if isinstance(audio_info, dict) and "bytes" in audio_info and audio_info["bytes"] is not None:
                        buffer = io.BytesIO(audio_info["bytes"])
                        waveform, sample_rate = torchaudio.load(buffer)
                    
                    # Case 2: File Path inside dictionary
                    elif isinstance(audio_info, dict) and "path" in audio_info and audio_info["path"] is not None:
                         waveform, sample_rate = torchaudio.load(audio_info["path"])

                    # Case 3: Decoded Array (if cast_column failed or wasn't used)
                    elif isinstance(audio_info, dict) and "array" in audio_info:
                        waveform = torch.from_numpy(audio_info["array"]).float()
                        sample_rate = audio_info["sampling_rate"]
                        if waveform.dim() == 1:
                            waveform = waveform.unsqueeze(0)
                            
                    if waveform is None:
                        logging.warning(f"Skipping {unique_id}: Could not resolve audio data.")
                        continue

                    # Ensure Mono
                    if waveform.shape[0] > 1:
                        waveform = waveform[0:1, :]

                    # Resample
                    if sample_rate != target_sampling_rate:
                        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sampling_rate)
                        waveform = resampler(waveform)
                        sample_rate = target_sampling_rate
                    
                    audio_array = waveform.squeeze().numpy()
                    
                    # Compute Features
                    features = extractor.extract(audio_array, sample_rate)
                    
                    # Check for empty features
                    if features.shape[0] == 0:
                        logging.warning(f"Skipping {unique_id}: Audio too short, 0 features extracted.")
                        continue

                    # Write to disk
                    storage_key = writer.write(unique_id, features)
                    
                    # Supervision
                    text = item.get("transcription", "")
                    # Fallback keys commonly found in Vivos/VLSP
                    if not text: text = item.get("sentence", "")
                    if not text: text = item.get("text", "")
                    
                    if text is None: text = ""
                    text = text.lower().strip()

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
                        channels=0,
                        recording_id=unique_id
                    )

                    cut = MonoCut(
                        id=unique_id,
                        start=0.0,
                        duration=duration,
                        channel=0,
                        supervisions=[sup],
                        features=features_manifest,
                        recording=None
                    )
                    cuts.append(cut)
                    
                except Exception as e:
                    # logging.error(f"Error on {unique_id}: {e}")
                    continue

        logging.info(f"Saving CutSet: {cut_set_path}")
        cut_set = CutSet.from_cuts(cuts)
        cut_set.to_file(cut_set_path)
        generated_cuts[icefall_split] = cut_set

    return generated_cuts

def train_bpe(args, train_cut_set):
    logging.info(f"Preparing BPE Model (Vocab Size: {args.vocab_size})...")
    
    lang_dir = args.output_dir / f"lang_bpe_{args.vocab_size}"
    lang_dir.mkdir(parents=True, exist_ok=True)
    
    model_prefix = lang_dir / "bpe"
    model_path = lang_dir / "bpe.model"
    
    if model_path.exists() and not args.overwrite:
        logging.info("BPE model exists. Skipping training.")
        return

    txt_file = lang_dir / "transcripts.txt"
    logging.info("Extracting text for BPE training...")
    
    count = 0
    with open(txt_file, "w", encoding="utf-8") as f:
        for cut in train_cut_set:
            text = " ".join(s.text for s in cut.supervisions if s.text)
            if text:
                f.write(text + "\n")
                count += 1
    
    if count == 0:
        logging.error("No text found for BPE training! Check your dataset keys (transcription/sentence/text).")
        return

    logging.info(f"Training SentencePiece model on {count} lines...")
    
    spm.SentencePieceTrainer.train(
        input=str(txt_file),
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=1.0, # Vietnamese has latin script, 1.0 is usually safe
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
    cut_sets = process_dataset(args)
    
    if "train" in cut_sets:
        train_bpe(args, cut_sets["train"])
    else:
        logging.warning("No 'train' split found. Skipping BPE training.")

    logging.info(f"Combined preparation done! Output: {args.output_dir}")

if __name__ == "__main__":
    main()
