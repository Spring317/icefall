import argparse
import logging
import math
import os
import time
import psutil  # Added for RAM monitoring
import onnx    # Added for parameter counting
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

class ModelProfiler:
    """Calculates parameters and monitors system RAM usage."""
    @staticmethod
    def count_parameters(onnx_model_path: str) -> int:
        model = onnx.load(onnx_model_path)
        return sum(np.prod(tensor.dims) for tensor in model.graph.initializer)

    @staticmethod
    def get_mem_usage() -> float:
        """Returns current process Resident Set Size (RSS) in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)

def compute_features(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
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
        # Profiling logic
        self.num_params = ModelProfiler.count_parameters(model_filename)
        
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
                dtype = np.int64 if meta.type == 'tensor(int64)' else np.float32
                self.state_info.append({"name": meta.name, "shape": fixed_shape, "dtype": dtype})

    def init_states(self):
        self.states = [np.zeros(info["shape"], dtype=info["dtype"]) for info in self.state_info]

    def __call__(self, x: np.ndarray, x_lens: np.ndarray) -> np.ndarray:
        input_feed = {}
        if "x" in self.input_names:
            input_feed["x"] = x
            if "x_lens" in self.input_names: input_feed["x_lens"] = x_lens
        elif "speech" in self.input_names:
            input_feed["speech"] = x
            if "speech_lengths" in self.input_names: input_feed["speech_lengths"] = x_lens
            
        for i, info in enumerate(self.state_info):
            input_feed[info["name"]] = self.states[i]

        outputs = self.session.run(self.output_names, input_feed)
        encoder_out = outputs[0]
        num_states = len(self.state_info)
        if num_states > 0:
            self.states = outputs[-num_states:]
        return encoder_out

class OnnxDecoder(OnnxBase):
    def __call__(self, y: np.ndarray) -> np.ndarray:
        input_name = self.input_names[0]
        outputs = self.session.run(self.output_names, {input_name: y})
        return outputs[0]

class OnnxJoiner(OnnxBase):
    def __call__(self, encoder_out: np.ndarray, decoder_out: np.ndarray) -> np.ndarray:
        if encoder_out.ndim == 3 and encoder_out.shape[1] == 1:
             encoder_out = encoder_out.squeeze(1)
        if decoder_out.ndim == 3 and decoder_out.shape[1] == 1:
            decoder_out = decoder_out.squeeze(1)

        input_feed = {self.input_names[0]: encoder_out, self.input_names[1]: decoder_out}
        outputs = self.session.run(self.output_names, input_feed)
        return outputs[0]

def main():
    args = get_args()
    
    # Measure RAM before loading
    mem_start = ModelProfiler.get_mem_usage()

    logging.info("Loading models...")
    encoder_model = OnnxEncoder(args.encoder)
    decoder_model = OnnxDecoder(args.decoder)
    joiner_model = OnnxJoiner(args.joiner)
    
    # Measure RAM after loading
    mem_after_load = ModelProfiler.get_mem_usage()
    
    total_params = encoder_model.num_params + decoder_model.num_params + joiner_model.num_params
    
    # PRINT PROFILING REPORT
    logging.info("="*50)
    logging.info(f"PROFILING REPORT:")
    logging.info(f"Total Parameters: {total_params / 1e6:.2f} M")
    logging.info(f"  - Encoder: {encoder_model.num_params / 1e6:.2f} M")
    logging.info(f"  - Decoder: {decoder_model.num_params / 1e6:.2f} M")
    logging.info(f"  - Joiner:  {joiner_model.num_params / 1e6:.2f} M")
    logging.info(f"Static RAM Usage (Models): {mem_after_load - mem_start:.2f} MB")
    logging.info(f"Total Current RAM: {mem_after_load:.2f} MB")
    logging.info("="*50)

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)
    cuts = load_manifest_lazy(args.cuts_jsonl)

    all_refs, all_hyps = [], []
    total_inference_time, total_audio_duration, num_processed_cuts = 0.0, 0.0, 0
    
    for i, cut in enumerate(cuts):
        try:
            logging.info(f"------------------------------------------------")
            logging.info(f"Processing cut {i}: {cut.id}")
            
            encoder_model.init_states()
            features = None
            
            if cut.has_features:
                try:
                    feats = torch.from_numpy(cut.load_features()).float()
                    if feats.dim() == 2: features = feats.unsqueeze(0)
                except Exception as e: logging.warning(f"Could not load features: {e}")

            if features is None:
                audio = cut.load_audio()
                if audio is None: continue
                if isinstance(audio, np.ndarray): audio = torch.from_numpy(audio)
                if audio.ndim > 1: audio = audio[0]
                features = compute_features(audio, cut.sampling_rate)

            cut_start_time = time.perf_counter()
            T = features.size(1)
            num_processed_frames = 0
            blank_id, context_size = 0, args.context_size
            decoder_input = [blank_id] * context_size
            hyp_tokens = []
            
            current_decoder_out = decoder_model(np.array([decoder_input], dtype=np.int64))

            while num_processed_frames < T:
                if num_processed_frames + args.segment < T:
                    chunk = features[:, num_processed_frames : num_processed_frames + args.segment, :]
                    num_processed_frames += args.offset
                else:
                    chunk = features[:, num_processed_frames:, :]
                    num_processed_frames = T
                    if chunk.size(1) < args.segment:
                        chunk = torch.nn.functional.pad(chunk, (0, 0, 0, args.segment - chunk.size(1)))
                
                encoder_out = encoder_model(chunk.numpy(), np.array([chunk.size(1)], dtype=np.int64))
                
                for t in range(encoder_out.shape[1]):
                    current_frame = encoder_out[:, t:t+1, :]
                    for _ in range(5):
                        logits = joiner_model(current_frame, current_decoder_out)
                        y = np.argmax(logits, axis=-1).item()
                        if y != blank_id:
                            hyp_tokens.append(y)
                            decoder_input = decoder_input[1:] + [y]
                            current_decoder_out = decoder_model(np.array([decoder_input], dtype=np.int64))
                        else: break
            
            cut_inference_time = time.perf_counter() - cut_start_time
            total_inference_time += cut_inference_time
            total_audio_duration += cut.duration
            num_processed_cuts += 1
            
            logging.info(f"Timing - RTF: {cut_inference_time / cut.duration:.2f} | Current RAM: {ModelProfiler.get_mem_usage():.2f} MB")

            text = sp.decode(hyp_tokens)
            ref_text = cut.supervisions[0].text if cut.supervisions else ""
            logging.info(f"Ref: {ref_text} | Hyp: {text}")
            
            if ref_text:
                all_refs.append(ref_text)
                all_hyps.append(text)

        except Exception as e:
            logging.error(f"Error processing cut {cut.id}: {e}", exc_info=True)
            
    if all_refs:
        logging.info(f"SUMMARY | Overall RTF: {total_inference_time / total_audio_duration:.2f} | WER: {jiwer.wer(all_refs, all_hyps):.2%}")

if __name__ == "__main__":
    main()
