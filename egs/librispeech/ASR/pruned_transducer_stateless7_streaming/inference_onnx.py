import argparse
import logging
import math
import os
from typing import List, Tuple, Dict, Any

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import torch
import torchaudio
from lhotse import CutSet, load_manifest_lazy
import jiwer 

logging.basicConfig(format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)

def get_args():
    parser = argparse.ArgumentParser(description="Inference with ONNX Transducer models (Encoder/Decoder/Joiner) and Lhotse cuts")
    parser.add_argument("--encoder", type=str, required=True, help="Path to the Encoder ONNX file")
    parser.add_argument("--decoder", type=str, required=True, help="Path to the Decoder ONNX file")
    parser.add_argument("--joiner", type=str, required=True, help="Path to the Joiner ONNX file")
    parser.add_argument("--bpe-model", type=str, required=True, help="Path to bpe.model")
    parser.add_argument("--cuts-jsonl", type=str, required=True, help="Path to cuts.jsonl.gz")
    parser.add_argument("--segment", type=int, default=39, help="Chunk size in frames for streaming")
    parser.add_argument("--offset", type=int, default=32, help="Stride/Offset in frames for streaming")
    parser.add_argument("--context-size", type=int, default=2, help="Decoder context size (2 for Zipformer/Pruned Stateless)")
    return parser.parse_args()

def compute_features(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Compute log-mel fbank features using torchaudio."""
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16000)
    
    features = torchaudio.compliance.kaldi.fbank(
        audio.unsqueeze(0),
        frame_length=25.0,
        frame_shift=10.0,
        num_mel_bins=80,
        dither=0.0,
    )
    return features.unsqueeze(0) # [1, T, 80]

class OnnxBase:
    def __init__(self, model_filename: str):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1
        
        self.session = ort.InferenceSession(
            model_filename,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [meta.name for meta in self.session.get_inputs()]
        self.output_names = [meta.name for meta in self.session.get_outputs()]

class OnnxEncoder(OnnxBase):
    def __init__(self, model_filename: str):
        super().__init__(model_filename)
        self.states = []
        self.state_info = []
        self._init_states_metadata()

    def _init_states_metadata(self):
        for meta in self.session.get_inputs():
            if meta.name not in ["x", "x_lens", "speech", "speech_lengths"]:
                shape = meta.shape
                fixed_shape = [1 if isinstance(s, str) else s for s in shape]
                dtype = np.float32
                if meta.type == 'tensor(int64)':
                    dtype = np.int64
                
                self.state_info.append({
                    "name": meta.name,
                    "shape": fixed_shape,
                    "dtype": dtype
                })

    def init_states(self):
        self.states = []
        for info in self.state_info:
            self.states.append(np.zeros(info["shape"], dtype=info["dtype"]))

    def __call__(self, x: np.ndarray, x_lens: np.ndarray) -> np.ndarray:
        input_feed = {}
        if "x" in self.input_names:
            input_feed["x"] = x
            if "x_lens" in self.input_names:
                input_feed["x_lens"] = x_lens
        elif "speech" in self.input_names:
            input_feed["speech"] = x
            if "speech_lengths" in self.input_names:
                input_feed["speech_lengths"] = x_lens
            
        for i, info in enumerate(self.state_info):
            input_feed[info["name"]] = self.states[i]

        outputs = self.session.run(self.output_names, input_feed)
        encoder_out = outputs[0]
        
        num_states = len(self.state_info)
        if num_states > 0:
            state_start_idx = len(outputs) - num_states
            self.states = outputs[state_start_idx:]
            
        return encoder_out

class OnnxDecoder(OnnxBase):
    def __call__(self, y: np.ndarray) -> np.ndarray:
        # y: [B, Context] -> returns [B, 1, D]
        input_name = self.input_names[0]
        outputs = self.session.run(self.output_names, {input_name: y})
        return outputs[0]

class OnnxJoiner(OnnxBase):
    def __call__(self, encoder_out: np.ndarray, decoder_out: np.ndarray) -> np.ndarray:
        """
        Args:
            encoder_out: [N, D_enc] (Flattened batch of frames)
            decoder_out: [N, D_dec] (Flattened batch of decoder states)
        Returns:
            logits: [N, Vocab]
        """
        # Ensure inputs are Rank 2 [N, D] for the vectorized call
        if encoder_out.ndim == 3: encoder_out = encoder_out.squeeze(1)
        if decoder_out.ndim == 3: decoder_out = decoder_out.squeeze(1)

        input_feed = {
            self.input_names[0]: encoder_out,
            self.input_names[1]: decoder_out
        }
        outputs = self.session.run(self.output_names, input_feed)
        return outputs[0]

def main():
    args = get_args()
    
    # Validation
    for p in [args.encoder, args.decoder, args.joiner, args.bpe_model]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    logging.info("Loading models...")
    encoder_model = OnnxEncoder(args.encoder)
    decoder_model = OnnxDecoder(args.decoder)
    joiner_model = OnnxJoiner(args.joiner)
    
    logging.info(f"Loading BPE model: {args.bpe_model}")
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)
    
    logging.info(f"Loading cuts: {args.cuts_jsonl}")
    try:
        cuts = load_manifest_lazy(args.cuts_jsonl)
    except Exception as e:
        logging.error(f"Failed to load cuts: {e}")
        return

    all_refs = []
    all_hyps = []

    for i, cut in enumerate(cuts):
        try:
            logging.info(f"------------------------------------------------")
            logging.info(f"Processing cut {i}: {cut.id}")
            
            # 1. Feature Extraction
            encoder_model.init_states()
            features = None
            
            if cut.has_features:
                try:
                    feats = torch.from_numpy(cut.load_features()).float()
                    # Ensure 3D: [1, T, 80]
                    if feats.dim() == 2:
                        features = feats.unsqueeze(0)
                    else:
                        features = feats
                except Exception as e:
                    logging.warning(f"Failed to load features for {cut.id}: {e}")

            if features is None:
                audio = cut.load_audio()
                if audio is None: 
                    logging.warning(f"Skipping {cut.id}: No audio found.")
                    continue
                if isinstance(audio, np.ndarray): 
                    audio = torch.from_numpy(audio)
                if audio.ndim > 1: 
                    audio = audio[0]
                features = compute_features(audio, cut.sampling_rate)

            # 2. Streaming Transducer Decoding (Vectorized)
            T = features.size(1)
            num_processed_frames = 0
            
            blank_id = 0
            context_size = args.context_size
            decoder_input = [blank_id] * context_size
            
            hyp_tokens = []
            
            # Initial decoder state
            decoder_input_np = np.array([decoder_input], dtype=np.int64)
            current_decoder_out = decoder_model(decoder_input_np) 

            while num_processed_frames < T:
                # Prepare Chunk
                if num_processed_frames + args.segment < T:
                    chunk = features[:, num_processed_frames : num_processed_frames + args.segment, :]
                    num_processed_frames += args.offset
                else:
                    chunk = features[:, num_processed_frames:, :]
                    num_processed_frames = T
                
                # Padding logic
                original_chunk_len = chunk.size(1)
                if chunk.size(1) < args.segment:
                    pad_amount = args.segment - chunk.size(1)
                    chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad_amount))
                
                chunk_len = np.array([original_chunk_len], dtype=np.int64)
                chunk_np = chunk.numpy()
                
                # Run Encoder
                encoder_out = encoder_model(chunk_np, chunk_len)
                
                # --- START VECTORIZED GREEDY SEARCH ---
                encoder_view = encoder_out.squeeze(0)
                encoder_view = encoder_view[:original_chunk_len] # Valid frames only
                
                T_chunk = encoder_view.shape[0]
                t_offset = 0
                
                while t_offset < T_chunk:
                    # Prepare blocks
                    current_encoder_block = encoder_view[t_offset:] 
                    block_size = current_encoder_block.shape[0]
                    
                    current_decoder_block = np.tile(
                        current_decoder_out.reshape(1, -1), 
                        (block_size, 1)
                    )
                    
                    # Run Joiner
                    logits = joiner_model(current_encoder_block, current_decoder_block)
                    max_ids = np.argmax(logits, axis=-1)
                    non_blank_indices = np.nonzero(max_ids != blank_id)[0]
                    
                    if len(non_blank_indices) == 0:
                        t_offset = T_chunk 
                    else:
                        k = non_blank_indices[0]
                        
                        # --- FIX HERE: Convert numpy int to python int ---
                        y = max_ids[k].item() 
                        
                        hyp_tokens.append(y)
                        
                        # Update Decoder
                        decoder_input = decoder_input[1:] + [y]
                        decoder_input_np = np.array([decoder_input], dtype=np.int64)
                        current_decoder_out = decoder_model(decoder_input_np)
                        
                        t_offset += (k + 1)
                # --- END VECTORIZED GREEDY SEARCH ---

            # Decode & Print
            # Now hyp_tokens contains standard Python ints, so sp.decode works
            text = sp.decode(hyp_tokens)
            ref_text = cut.supervisions[0].text if cut.supervisions else ""
            
            logging.info(f"Ref Text: {ref_text}")
            logging.info(f"Hyp Text: {text}")
            
            if ref_text:
                all_refs.append(ref_text)
                all_hyps.append(text)

        except Exception as e:
            logging.error(f"Error processing cut {cut.id}: {e}", exc_info=True)
            
    # Final WER Report
    if all_refs:
        logging.info(f"------------------------------------------------")
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

if __name__ == "__main__":
    main()
