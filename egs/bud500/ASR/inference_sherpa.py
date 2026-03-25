import argparse
import logging
import os
import time
import json

import numpy as np
import sherpa_onnx
import soundfile as sf
import jiwer

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate streaming Zipformer with sherpa-onnx natively (with padding fix)")
    parser.add_argument("--encoder", type=str, required=True, help="Path to encoder.onnx")
    parser.add_argument("--decoder", type=str, required=True, help="Path to decoder.onnx")
    parser.add_argument("--joiner", type=str, required=True, help="Path to joiner.onnx")
    parser.add_argument("--tokens", type=str, required=True, help="Path to tokens.txt")
    parser.add_argument("--manifest", type=str, required=True, help="Path to transcripts.jsonl")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples to process")
    parser.add_argument("--decoding-method", type=str, default="greedy_search", choices=["greedy_search", "modified_beam_search"])
    parser.add_argument("--num-threads", type=int, default=1)
    return parser.parse_args()

def main():
    args = get_args()
    
    # Validation
    for p in [args.encoder, args.decoder, args.joiner, args.tokens, args.manifest]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")
            
    # 1. Initialize Sherpa-ONNX Recognizer
    logging.info("Initializing sherpa-onnx OnlineRecognizer...")
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        tokens=args.tokens,
        num_threads=args.num_threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=args.decoding_method,
        max_active_paths=4
    )
    
    # 2. Load Manifest
    logging.info(f"Loading metadata from {args.manifest}...")
    with open(args.manifest, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    if args.max_samples:
        lines = lines[:args.max_samples]
        
    logging.info("-" * 60)
    logging.info(f"Starting Evaluation on {len(lines)} samples using sherpa-onnx...")
    logging.info("-" * 60)
    
    all_refs = []
    all_hyps = []
    total_time = 0.0
    total_audio_duration = 0.0
    
    # 3. Iterate through JSONL and decode directly from WAVs
    for i, line in enumerate(lines):
        meta = json.loads(line)
        utt_id = meta["id"]
        audio_path = meta["audio_filepath"]
        ref_text = meta["text"]
        
        logging.info(f"Processing audio {i+1}/{len(lines)}: {utt_id}")
        
        try:
            # Load raw audio using soundfile as float32 natively
            audio, sample_rate = sf.read(audio_path, dtype='float32')
            
            # Ensure mono audio just in case
            if audio.ndim > 1:
                audio = audio[:, 0]
                
            # --- THE FIX: ADD 0.5 SECONDS OF SILENCE PADDING ---
            # This flushes the final tokens out of the streaming model
            tail_padding = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
            audio = np.concatenate([audio, tail_padding])
            # ---------------------------------------------------
                
            samples = np.ascontiguousarray(audio)
            
            # We calculate duration based on the actual processed array (including padding)
            # so the RTF accurately reflects the compute time spent.
            audio_dur = len(samples) / sample_rate 
            total_audio_duration += audio_dur
            
            # --- Inference with Sherpa-ONNX ---
            t0 = time.time()
            
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            stream.input_finished()
            
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
                
            result = recognizer.get_result(stream)
            hyp_text = getattr(result, 'text', str(result))
            
            infer_dur = time.time() - t0
            total_time += infer_dur
            
            logging.info(f"  Ref : {ref_text}")
            logging.info(f"  Hyp : {hyp_text}")
            logging.info(f"  Time: {infer_dur*1000:.0f} ms")
            
            if ref_text:
                all_refs.append(ref_text)
                all_hyps.append(hyp_text)
                
        except Exception as e:
            logging.error(f"Error processing {utt_id}: {e}")
            
    # 4. Final Report
    if all_refs:
        logging.info("-" * 60)
        logging.info(f"Calculating Final Metrics for {len(all_refs)} sentences...")
        
        # Vietnamese text normalization for Jiwer
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ])
        
        wer = jiwer.wer(
            all_refs,  
            all_hyps,  
            reference_transform=transformation,  
            hypothesis_transform=transformation
        )
        
        avg_infer_time = total_time / len(all_refs)
        rtf = total_time / total_audio_duration
        
        logging.info(f"Overall WER: {wer:.2%}")
        logging.info(f"Total Audio Duration (inc. padding): {total_audio_duration:.2f} s")
        logging.info(f"Total Inference Time: {total_time:.2f} s")
        logging.info(f"Average Inference Time: {avg_infer_time*1000:.0f} ms/utterance")
        logging.info(f"Real-Time Factor (RTF): {rtf:.4f}")
        logging.info("-" * 60)
    else:
        logging.warning("No reference text found to compute WER.")

if __name__ == "__main__":
    main()
