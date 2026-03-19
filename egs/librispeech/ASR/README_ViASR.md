# Vietnamese Streaming ASR Training with pruned_transducer_stateless7_streaming_multi

This guide walks you through training a **streaming Zipformer transducer** model for Vietnamese ASR using the `pruned_transducer_stateless7_streaming_multi` recipe.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [BPE Model & Feature Extraction](#bpe-model--feature-extraction)
5. [Training](#training)
6. [Decoding](#decoding)
7. [Model Export](#model-export)
8. [Pretrained Inference](#pretrained-inference)
9. [ONNX Export & Deployment](#onnx-export--deployment)

---

## Prerequisites

- **Hardware**: At least 1 GPU (recommended: 4× V100/A100 GPUs)
- **Software**:
  - Python 3.8+
  - PyTorch 1.12+ with CUDA
  - [k2](https://github.com/k2-fsa/k2) installed
  - [lhotse](https://github.com/lhotse-speech/lhotse) installed
  - [icefall](https://github.com/k2-fsa/icefall) installed

## Environment Setup

```bash
# 1. Clone icefall and install dependencies
git clone https://github.com/k2-fsa/icefall.git
cd icefall
pip install -r requirements.txt

# 2. Install icefall
pip install -e .

# 3. Verify k2 and lhotse
python -c "import k2; print(k2.__version__)"
python -c "import lhotse; print(lhotse.__version__)"
```

## Data Preparation

The [`data_prep.py`](egs/librispeech/ASR/data_prep.py) script handles **everything automatically**:

1. Downloads the [Bud500](https://huggingface.co/datasets/linhtran92/viet_bud500) Vietnamese dataset from Hugging Face (if not already cached locally)
2. Extracts **80-dim log-mel fbank features** in-memory and writes them to disk
3. Creates Lhotse CutSets for train/dev/test splits
4. Trains a **SentencePiece BPE model** (unigram, vocab size 4000)

### Step 1: Create output directory and run data preparation

```bash
cd egs/librispeech/ASR

# Create the data directory
mkdir -p data

# Run the all-in-one data preparation script
# This will auto-download Bud500 from Hugging Face if not available locally
python data_prep.py \
  --output-dir data \
  --vocab-size 4000
```

This produces the following structure:

```
data/
├── fbank/
│   ├── bud500_cuts_train.jsonl.gz   # Training CutSet with pre-computed features
│   ├── bud500_cuts_dev.jsonl.gz     # Validation CutSet
│   ├── bud500_cuts_test.jsonl.gz    # Test CutSet
│   ├── feats_train/                 # Fbank feature files (LilcomFiles)
│   ├── feats_dev/
│   └── feats_test/
└── lang_bpe_4000/
    ├── bpe.model                    # SentencePiece BPE model
    ├── bpe.vocab                    # BPE vocabulary
    └── transcripts.txt              # Extracted training transcripts
```

### Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--output-dir` | Root output directory | `data` |
| `--vocab-size` | BPE vocabulary size | `4000` |
| `--overwrite` | Force re-processing even if outputs exist | `False` |

> **Note**: If you already have the Bud500 dataset cached by Hugging Face `datasets`, it will be loaded from cache without re-downloading. The script also skips splits that have already been processed (use `--overwrite` to force re-processing).

> **Tip**: If you have a local `bpe.model` and `bpe.vocab` in the same directory as `data_prep.py`, the script will copy them directly instead of re-training.

## BPE Model & Feature Extraction

> **Both are already handled by `data_prep.py`** in the previous step. No separate action needed.

- **BPE model**: A unigram SentencePiece model (vocab size 4000) is saved to `data/lang_bpe_4000/bpe.model`
- **Fbank features**: 80-dim log-mel filterbank features are extracted in-memory and stored as LilcomFiles under `data/fbank/`

If you want to re-train the BPE model with a different vocab size:

```bash
python data_prep.py --output-dir data --vocab-size 2000 --overwrite
# This will create data/lang_bpe_2000/ with the new BPE model
```

## Training

### Step 2: Adapt the training script

The `train.py` in the recipe needs to load Bud500 CutSets instead of LibriSpeech + GigaSpeech. Modify the [`train.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/train.py) `run()` function to load Bud500 cuts directly:

```python
# In train.py run() function, replace the LibriSpeech/GigaSpeech data loading with:
from lhotse import CutSet

# Load Bud500 cuts (already have pre-computed fbank features from data_prep.py)
train_cuts = CutSet.from_file("data/fbank/bud500_cuts_train.jsonl.gz")
valid_cuts = CutSet.from_file("data/fbank/bud500_cuts_dev.jsonl.gz")
```

> **Reference**: See how [`pruned_transducer_stateless7_streaming/train.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming/train.py) already loads Bud500 cuts for an example of this pattern.

### Step 3: Run training

#### Single-dataset training (Bud500 Vietnamese)

```bash
cd egs/librispeech/ASR

export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless7_streaming_multi/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7_streaming_multi/exp \
  --max-duration 550 \
  --manifest-dir ./data/fbank \
  --bpe-model data/lang_bpe_4000/bpe.model \
  --master-port 12345
```

#### Key training arguments

Refer to [`train.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/train.py) for all options. Important ones:

| Argument | Description | Default |
|----------|-------------|---------|
| `--world-size` | Number of GPUs for DDP training | 1 |
| `--num-epochs` | Total training epochs | 30 |
| `--start-epoch` | Resume from this epoch | 1 |
| `--use-fp16` | Mixed precision training (recommended) | 0 |
| `--exp-dir` | Directory for checkpoints and logs | — |
| `--max-duration` | Max audio duration (seconds) per batch | 300 |
| `--manifest-dir` | Directory with cut manifests | `data/fbank` |
| `--bpe-model` | Path to BPE model | — |
| `--num-encoder-layers` | Encoder layer config per stack | `"2,4,3,2,4"` |
| `--feedforward-dims` | FFN dims per stack | `"1024,1024,2048,2048,1024"` |
| `--encoder-dims` | Encoder dims per stack | `"384,384,384,384,384"` |
| `--enable-musan` | Enable MUSAN noise augmentation | True |
| `--save-every-n` | Save checkpoint every N batches | 4000 |

#### Smaller model variant (for limited resources)

For a smaller model (~6.1M params), use:

```bash
./pruned_transducer_stateless7_streaming_multi/train.py \
  --world-size 2 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7_streaming_multi/exp_small \
  --max-duration 550 \
  --manifest-dir ./data/fbank \
  --bpe-model data/lang_bpe_4000/bpe.model \
  --num-encoder-layers "2,2,2,2,2" \
  --feedforward-dims "256,256,512,512,256" \
  --nhead "4,4,4,4,4" \
  --encoder-dims "128,128,128,128,128" \
  --attention-dims "96,96,96,96,96" \
  --encoder-unmasked-dims "96,96,96,96,96" \
  --zipformer-downsampling-factors "1,2,4,8,2" \
  --master-port 12345
```

#### Resume training

```bash
# Resume from epoch checkpoint
./pruned_transducer_stateless7_streaming_multi/train.py \
  --start-epoch 11 \
  --exp-dir pruned_transducer_stateless7_streaming_multi/exp \
  ... (same args as above)

# Resume from batch checkpoint
./pruned_transducer_stateless7_streaming_multi/train.py \
  --start-batch 436000 \
  --exp-dir pruned_transducer_stateless7_streaming_multi/exp \
  ... (same args as above)
```

### Monitoring training

```bash
# View tensorboard logs
tensorboard --logdir pruned_transducer_stateless7_streaming_multi/exp/tensorboard
```

## Decoding

After training, decode with [`decode.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/decode.py). The script supports these decoding methods from [`beam_search.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/beam_search.py):

- `greedy_search` / `greedy_search_batch`
- `beam_search`
- `modified_beam_search`
- `fast_beam_search_one_best`
- `fast_beam_search_nbest`
- `fast_beam_search_nbest_oracle`
- `fast_beam_search_nbest_LG`

### Simulated streaming decoding

```bash
# Greedy search
./pruned_transducer_stateless7_streaming_multi/decode.py \
  --epoch 30 \
  --avg 9 \
  --exp-dir ./pruned_transducer_stateless7_streaming_multi/exp \
  --max-duration 600 \
  --decode-chunk-len 32 \
  --decoding-method greedy_search \
  --manifest-dir ./data/fbank \
  --bpe-model data/lang_bpe_4000/bpe.model

# Modified beam search
./pruned_transducer_stateless7_streaming_multi/decode.py \
  --epoch 30 \
  --avg 9 \
  --exp-dir ./pruned_transducer_stateless7_streaming_multi/exp \
  --max-duration 600 \
  --decode-chunk-len 32 \
  --decoding-method modified_beam_search \
  --beam-size 4 \
  --manifest-dir ./data/fbank \
  --bpe-model data/lang_bpe_4000/bpe.model

# Fast beam search
./pruned_transducer_stateless7_streaming_multi/decode.py \
  --epoch 30 \
  --avg 9 \
  --exp-dir ./pruned_transducer_stateless7_streaming_multi/exp \
  --max-duration 600 \
  --decode-chunk-len 32 \
  --decoding-method fast_beam_search \
  --beam 20.0 \
  --max-contexts 8 \
  --max-states 64 \
  --manifest-dir ./data/fbank \
  --bpe-model data/lang_bpe_4000/bpe.model
```

### Real chunk-wise streaming decoding

Use [`streaming_decode.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/streaming_decode.py) for true streaming evaluation:

```bash
for m in greedy_search modified_beam_search fast_beam_search; do
  ./pruned_transducer_stateless7_streaming_multi/streaming_decode.py \
    --epoch 30 \
    --avg 9 \
    --exp-dir ./pruned_transducer_stateless7_streaming_multi/exp \
    --decoding-method $m \
    --decode-chunk-len 32 \
    --num-decode-streams 2000 \
    --manifest-dir ./data/fbank \
    --bpe-model data/lang_bpe_4000/bpe.model
done
```

> **Note**: `decode.py` processes the whole utterance at once with masking (simulated streaming), while `streaming_decode.py` processes frames chunk-by-chunk (real streaming). See [`decode_stream.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/decode_stream.py) and [`streaming_beam_search.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/streaming_beam_search.py) for the streaming decoding internals.

### Sweep for best checkpoint

```bash
for m in greedy_search fast_beam_search modified_beam_search; do
  for epoch in $(seq 30 -1 20); do
    for avg in $(seq 9 -1 1); do
      ./pruned_transducer_stateless7_streaming_multi/decode.py \
        --epoch $epoch \
        --avg $avg \
        --exp-dir ./pruned_transducer_stateless7_streaming_multi/exp \
        --max-duration 600 \
        --decode-chunk-len 32 \
        --decoding-method $m \
        --manifest-dir ./data/fbank \
        --bpe-model data/lang_bpe_4000/bpe.model
    done
  done
done
```

## Model Export

Use [`export.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/export.py) to export the trained model.

### Export `model.state_dict()`

```bash
epoch=30
avg=9

./pruned_transducer_stateless7_streaming_multi/export.py \
  --exp-dir ./pruned_transducer_stateless7_streaming_multi/exp \
  --tokens data/lang_bpe_4000/tokens.txt \
  --epoch $epoch \
  --avg $avg

# This generates: exp/pretrained.pt
```

### Export as TorchScript (JIT)

```bash
./pruned_transducer_stateless7_streaming_multi/export.py \
  --exp-dir ./pruned_transducer_stateless7_streaming_multi/exp \
  --tokens data/lang_bpe_4000/tokens.txt \
  --epoch $epoch \
  --avg $avg \
  --jit 1

# This generates: exp/cpu_jit.pt
```

### Export via `torch.jit.trace()`

Use [`jit_trace_export.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/jit_trace_export.py):

```bash
./pruned_transducer_stateless7_streaming_multi/jit_trace_export.py \
  --bpe-model data/lang_bpe_4000/bpe.model \
  --use-averaged-model True \
  --decode-chunk-len 32 \
  --exp-dir ./pruned_transducer_stateless7_streaming_multi/exp \
  --epoch $epoch \
  --avg $avg

# This generates:
#   exp/encoder_jit_trace.pt
#   exp/decoder_jit_trace.pt
#   exp/joiner_jit_trace.pt
```

## Pretrained Inference

Use [`pretrained.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/pretrained.py) to transcribe audio files:

```bash
./pruned_transducer_stateless7_streaming_multi/pretrained.py \
  --checkpoint ./pruned_transducer_stateless7_streaming_multi/exp/pretrained.pt \
  --tokens data/lang_bpe_4000/tokens.txt \
  --method greedy_search \
  /path/to/vietnamese_audio.wav
```

Use [`jit_trace_pretrained.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/jit_trace_pretrained.py) for torchscript models:

```bash
./pruned_transducer_stateless7_streaming_multi/jit_trace_pretrained.py \
  --encoder-model-filename ./pruned_transducer_stateless7_streaming_multi/exp/encoder_jit_trace.pt \
  --decoder-model-filename ./pruned_transducer_stateless7_streaming_multi/exp/decoder_jit_trace.pt \
  --joiner-model-filename ./pruned_transducer_stateless7_streaming_multi/exp/joiner_jit_trace.pt \
  --bpe-model ./data/lang_bpe_4000/bpe.model \
  --decode-chunk-len 32 \
  /path/to/vietnamese_audio.wav
```

## ONNX Export & Deployment

### Export to ONNX

Use [`export-onnx.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/export-onnx.py):

```bash
./pruned_transducer_stateless7_streaming_multi/export-onnx.py \
  --bpe-model data/lang_bpe_4000/bpe.model \
  --use-averaged-model True \
  --epoch $epoch \
  --avg $avg \
  --decode-chunk-len 32 \
  --exp-dir ./pruned_transducer_stateless7_streaming_multi/exp

# This generates:
#   exp/encoder-epoch-{epoch}-avg-{avg}.onnx
#   exp/decoder-epoch-{epoch}-avg-{avg}.onnx
#   exp/joiner-epoch-{epoch}-avg-{avg}.onnx
```

### Verify ONNX models

Use [`onnx_check.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/onnx_check.py):

```bash
./pruned_transducer_stateless7_streaming_multi/onnx_check.py \
  --bpe-model data/lang_bpe_4000/bpe.model \
  --encoder-model-filename ./exp/encoder-epoch-30-avg-9.onnx \
  --decoder-model-filename ./exp/decoder-epoch-30-avg-9.onnx \
  --joiner-model-filename ./exp/joiner-epoch-30-avg-9.onnx \
  --jit-filename ./exp/encoder_jit_trace.pt
```

### Run ONNX inference

Use [`onnx_pretrained.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/onnx_pretrained.py):

```bash
./pruned_transducer_stateless7_streaming_multi/onnx_pretrained.py \
  --encoder-model-filename ./exp/encoder-epoch-30-avg-9.onnx \
  --decoder-model-filename ./exp/decoder-epoch-30-avg-9.onnx \
  --joiner-model-filename ./exp/joiner-epoch-30-avg-9.onnx \
  --tokens data/lang_bpe_4000/tokens.txt \
  /path/to/vietnamese_audio.wav
```

### Export to NCNN (for mobile deployment)

Use [`export-for-ncnn.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/export-for-ncnn.py):

```bash
./pruned_transducer_stateless7_streaming_multi/export-for-ncnn.py \
  --tokens data/lang_bpe_4000/tokens.txt \
  --exp-dir ./pruned_transducer_stateless7_streaming_multi/exp \
  --use-averaged-model True \
  --epoch $epoch \
  --avg $avg \
  --decode-chunk-len 32 \
  --num-encoder-layers "2,4,3,2,4" \
  --feedforward-dims "1024,1024,2048,2048,1024" \
  --nhead "8,8,8,8,8" \
  --encoder-dims "384,384,384,384,384" \
  --attention-dims "192,192,192,192,192" \
  --encoder-unmasked-dims "256,256,256,256,256" \
  --zipformer-downsampling-factors "1,2,4,8,2"
```

Then decode with [`streaming-ncnn-decode.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/streaming-ncnn-decode.py).

### Deploy with sherpa-onnx

For production deployment, use [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with the exported ONNX models.

---

## Model Architecture Summary

The model consists of three components (defined in the recipe):

| Component | File | Description |
|-----------|------|-------------|
| Encoder | [`zipformer.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/zipformer.py) | Streaming Zipformer with configurable stacks |
| Decoder | [`decoder.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/decoder.py) | Stateless prediction network (Embedding + Conv1d) |
| Joiner | [`joiner.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/joiner.py) | Combines encoder and decoder outputs |
| Full Model | [`model.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/model.py) | Transducer model with pruned RNN-T loss |

Default configuration: **~70.37M parameters**

## File Reference

| File | Purpose |
|------|---------|
| [`data_prep.py`](egs/librispeech/ASR/data_prep.py) | All-in-one data preparation (download Bud500 + fbank + BPE) |
| [`train.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/train.py) | Main training script |
| [`decode.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/decode.py) | Simulated streaming decoding |
| [`streaming_decode.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/streaming_decode.py) | Real chunk-wise streaming decoding |
| [`export.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/export.py) | Export state_dict / JIT / ONNX |
| [`export-onnx.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/export-onnx.py) | ONNX-specific export |
| [`export-for-ncnn.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/export-for-ncnn.py) | NCNN export for mobile |
| [`pretrained.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/pretrained.py) | Inference with pretrained checkpoint |
| [`jit_trace_export.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/jit_trace_export.py) | Export via torch.jit.trace |
| [`jit_trace_pretrained.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/jit_trace_pretrained.py) | Inference with traced models |
| [`onnx_pretrained.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/onnx_pretrained.py) | ONNX inference |
| [`onnx_check.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/onnx_check.py) | Validate ONNX export correctness |
| [`beam_search.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/beam_search.py) | All decoding algorithms |
| [`streaming_beam_search.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/streaming_beam_search.py) | Streaming-specific beam search |
| [`decode_stream.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/decode_stream.py) | Stream state management for real streaming |
| [`asr_datamodule.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/asr_datamodule.py) | Data loading and batching |
| [`scaling.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/scaling.py) | Custom scaling layers |
| [`optim.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/optim.py) | ScaledAdam and Eden optimizers |

## Tips

- **Bud500 dataset**: The [Bud500](https://huggingface.co/datasets/linhtran92/viet_bud500) dataset is ~500 hours of Vietnamese speech. It is automatically downloaded from Hugging Face by `data_prep.py` and cached locally by the `datasets` library.
- **Vietnamese text normalization**: Ensure consistent Unicode normalization (NFC) for all Vietnamese text. The Bud500 transcripts are already normalized.
- **Vocab size**: The default is 4000 (unigram). For Vietnamese, experiment with BPE vocab sizes of 2000, 4000, and 8000. Vietnamese has many diacritical marks that increase the effective character set.
- **Chunk size**: `--decode-chunk-len 32` corresponds to 320ms latency. You can try 16 (160ms) for lower latency or 64 (640ms) for better accuracy.
- **Speed perturbation**: The data pipeline automatically applies speed perturbation (0.9x, 1.0x, 1.1x) if configured in the cuts.
- **Multi-dataset**: If you have multiple Vietnamese datasets, follow the pattern in [`train.py`](egs/librispeech/ASR/pruned_transducer_stateless7_streaming_multi/train.py) which uses `--giga-prob` to mix LibriSpeech with GigaSpeech. Create similar mixing for your datasets.
- **Re-running data_prep.py**: The script is idempotent — it skips splits that already have output files. Use `--overwrite` to force re-processing.
