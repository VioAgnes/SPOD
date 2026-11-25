# SPOD: Recovering Suppressed Temporal Evidence in Video-LLMs via Single-Pass Orthogonal Decomposition

[Chinese README](README.zh-CN.md)

## 1. Introduction

Video Large Language Models (Video-LLMs) can generate temporally inconsistent responses when static visual cues dominate dynamic temporal evidence. **SPOD** aims to recover temporal evidence from hidden representations through orthogonal subspace decomposition during inference.

| Component | Description |
| :--- | :--- |
| **FDA** | **Frame Distinctiveness Analysis** identifies temporally salient frames by measuring hidden-state discriminability across middle layers. |
| **TDS** | **Temporal Delta Subspace** constructs an orthogonal temporal subspace `T` and prior subspace `P` from weighted inter-frame differences. |

Hidden representations are decomposed into visual evidence, question-prior, and residual components, then recombined with configurable gains before the language-model head. Interventions cover prefill tokens and valid decoding steps without rewriting historical KV entries.

## 2. Setup

### 2.1 Environment Setup

Use Linux with CUDA and run the following commands from the repository root:

```bash
conda create -n spod python=3.10 -y
conda activate spod

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e . --no-deps
```

The reference environment uses PyTorch 2.4.0+cu124, Transformers 4.51.3, and NumPy 1.26.4.

### 2.2 Dataset

Download **VidHalluc** from its [official Hugging Face repository](https://huggingface.co/datasets/chaoyuli/VidHalluc), linked from the [project website](https://vid-halluc.github.io/).

| Files | Download |
| :--- | :--- |
| Annotations | [ach_binaryqa.json](https://huggingface.co/datasets/chaoyuli/VidHalluc/resolve/main/ach_binaryqa.json?download=true), [ach_mcq.json](https://huggingface.co/datasets/chaoyuli/VidHalluc/resolve/main/ach_mcq.json?download=true), [sth.json](https://huggingface.co/datasets/chaoyuli/VidHalluc/resolve/main/sth.json?download=true), [tsh.json](https://huggingface.co/datasets/chaoyuli/VidHalluc/resolve/main/tsh.json?download=true) |
| Action videos | [ACH_videos.zip](https://huggingface.co/datasets/chaoyuli/VidHalluc/resolve/main/data/ACH_videos.zip?download=true) |
| Scene-transition videos | [STH_videos.zip](https://huggingface.co/datasets/chaoyuli/VidHalluc/resolve/main/data/STH_videos.zip?download=true) |
| Temporal-sequence videos | [TSH_videos.zip](https://huggingface.co/datasets/chaoyuli/VidHalluc/resolve/main/data/TSH_videos.zip?download=true) |

Extract the video archives and arrange the annotations and videos as follows:

```text
data/VidHalluc/
  ach_binaryqa.json
  ach_mcq.json
  sth.json
  tsh.json
  videos/
    ACH/<video_id>.mp4
    STH/<video_id>.mp4
    TSH/<video_id>.mp4
```

For an external dataset location:

```bash
export VIDHALLUC_DATA_DIR=/path/to/datasets/VidHalluc
```

### 2.3 Model Download

| Model | Purpose | Download |
| :--- | :--- | :--- |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Video understanding and generation | [Checkpoint files](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main) |
| [sup-simcse-roberta-large](https://huggingface.co/princeton-nlp/sup-simcse-roberta-large) | Sentence embeddings for STH evaluation | [Checkpoint files](https://huggingface.co/princeton-nlp/sup-simcse-roberta-large/tree/main) |

Download the complete checkpoints, including configuration, tokenizer, and processor files, into:

```text
models/
  Qwen2.5-VL-7B-Instruct/
  sup-simcse-roberta-large/
```

For an external model location:

```bash
export VIDHALLUC_MODEL_DIR=/path/to/models
```

## 3. Experiments

### VidHalluc Inference

```bash
# Quick run: eight inputs per subset
SMOKE=1 OUTPUT_DIR=outputs/demo bash scripts/inference.sh

# Full inference on one GPU
bash scripts/inference.sh

# Full inference on four GPUs
NUM_GPUS=4 OUTPUT_DIR=outputs/vidhalluc_4gpu bash scripts/inference.sh
```

### VidHalluc Evaluation

```bash
# Evaluate the single-GPU run
bash scripts/evaluate.sh

# Evaluate the four-GPU run
OUTPUT_DIR=outputs/vidhalluc_4gpu bash scripts/evaluate.sh
```

## 4. Repository Structure

```text
configs/
  spod.json                 # Rank-8 method configuration
src/vidhalluc/
  interventions/            # Evidence decomposition and attention intervention
  decoding/                 # Video decoding and fallback
  evaluators/               # Official VidHalluc scoring functions
  diagnostics/              # Answer-format analysis
  resources/                # Benchmark input IDs and prompts
  inference.py              # Model inference
  evaluation.py             # Benchmark evaluation
  cli.py                    # Command-line interface
scripts/
  inference.sh              # Single- and multi-GPU inference
  evaluate.sh               # Official evaluation
  check_environment.py      # Environment and asset checks
tests/                      # CPU-only checks
pyproject.toml              # Package configuration
requirements.txt            # Python dependencies
```
