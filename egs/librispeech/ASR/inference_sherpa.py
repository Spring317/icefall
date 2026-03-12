import argparse
import logging
import os
import time

import numpy as np
import sherpa_onnx
from lhotse import CutSet
import jiwer

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate streaming Zipformer with sherpa-onnx on Lhotse cuts")
    parser.add_argument("--encoder", type=str, required=True, help="Path to encoder.onnx")
    parser.add_argument("--decoder", type=str, required=True, help="Path to decoder.onnx")
    parser.add_argument("--joiner", type=str, required=True, help="Path to joiner.onnx")
    parser.add_argument("--tokens", type=str, required=True, help="Path to tokens.txt")
    parser.add_argument("--cuts", type=str, required=True, help="Path to cuts jsonl.gz")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--decoding-method", type=str, default="greedy_search", choices=["greedy_search", "modified_beam_search"])
    parser.add_argument("--num-threads", type=int, default=1)
    return parser.parse_args()

def main():
    args = get_args()
    
    # Validation
    for p in [args.encoder, args.decoder, args.joiner, args.tokens]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")
            
    # 1. Initialize Sherpa-ONNX Recognizer
    # This automatically handles C++ level ONNX states, BPE decoding, and Fbank extraction!
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
    
    # 2. Load Cuts
    logging.info(f"Loading cuts from {args.cuts}...")
    try:
        cuts = CutSet.from_file(args.cuts)
    except Exception as e:
        logging.error(f"Failed to load cuts: {e}")
        return
        
    if args.max_samples:
        cuts = cuts.subset(first=args.max_samples)
        
    logging.info("-" * 60)
    logging.info(f"Starting Evaluation on {len(cuts)} samples using sherpa-onnx...")
    logging.info("-" * 60)
    
    all_refs = []
    all_hyps = []
    total_time = 0.0
    
    for i, cut in enumerate(cuts):
        logging.info(f"Processing cut {i+1}: {cut.id}")
        try:
            # Load raw audio: sherpa-onnx wants float32 arrays in the [-1.0, 1.0] range
            # Lhotse provides exactly this natively. No manual fbank needed!
            audio = cut.load_audio()
            if audio is None:
                logging.warning(f"Skipping {cut.id}: No audio found.")
                continue
                
            if audio.ndim > 1:
                audio = audio[0] # Take the first channel
            
            samples = np.ascontiguousarray(audio, dtype=np.float32)
            ref_text = cut.supervisions[0].text if cut.supervisions else ""
            
            # --- Inference with Sherpa-ONNX ---
            t0 = time.time()
            
            # Create a new stream for the utterance
            stream = recognizer.create_stream()
            
            # Feed the raw waveform. Sherpa-onnx handles resampling internally if needed.
            stream.accept_waveform(cut.sampling_rate, samples)
            
            # Signal the stream is finished
            stream.input_finished()
            
            # Decode the stream chunk by chunk
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
                
            # Get the final text result
            result = recognizer.get_result(stream)
            hyp_text = getattr(result, 'text', str(result))
            
            infer_dur = time.time() - t0
            total_time += infer_dur
            
            logging.info(f"  Ref : {ref_text}")
            logging.info(f"  Hyp : {hyp_text}")
            logging.info(f"  Time: {infer_dur*1000:.0f}ms")
            
            if ref_text:
                all_refs.append(ref_text)
                all_hyps.append(hyp_text)
                
        except Exception as e:
            logging.error(f"Error processing cut {cut.id}: {e}", exc_info=True)
            
    # 3. Final WER Report using jiwer
    if all_refs:
        logging.info("-" * 60)
        logging.info(f"Calculating WER for {len(all_refs)} sentences...")
        
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
        
        logging.info(f"Overall WER: {wer:.2%}")
        logging.info(f"Total Inference Time: {total_time:.2f}s")
    else:
        logging.warning("No reference text found to compute WER.")

if __name__ == "__main__":
    main()
