# SPOD: Recovering Suppressed Temporal Evidence in Video-LLMs via Single-Pass Orthogonal Decomposition

# 📖 Overview

Video Large Language Models (Video-LLMs) frequently generate temporally inconsistent responses due to an inherent bias toward static visual cues over dynamic temporal information—a failure mode termed **temporal hallucination**. Our key insight reveals that temporal information already exists within hidden representations but is systematically suppressed during standard inference.

SPOD recovers suppressed temporal signals through orthogonal subspace decomposition within a **single inference pass**:

| Component | Description                                                  |
| --------- | ------------------------------------------------------------ |
| **FDA**   | Frame Distinctiveness Analysis — identifies temporally salient frames by measuring hidden-state discriminability across middle layers |
| **TDS**   | Temporal Delta Subspace — constructs an orthogonal temporal subspace $\mathcal{T}$ and non-temporal subspace $\mathcal{P}$ from weighted inter-frame differences |



## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n spod python=3.10 -y
conda activate spod

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Download

Please download Qwen2.5-VL-7B, VideoLLaMA2-7B, and Video-LLaVA-7B from their official release pages.

### 3. Dataset Download

Please download VideoMMMU, EventHallusion, and VidHalluc from their official sources.




## 📊 Running Experiments

### Theory Analysis

```bash
bash run_theory.sh perception 30
bash run_theory.sh all 100
```

### VideoMMMU Evaluation

```bash
# SPOD
bash run_videommmu.sh spod all

# Baseline comparison
bash run_videommmu.sh baseline all
```

### EventHallusion Evaluation

```bash
# SPOD
bash run_eventhallusion.sh spod all

# Baseline
bash run_eventhallusion.sh baseline entire
```

### VidHalluc Evaluation

```bash
# SPOD
bash run_vidhalluc.sh spod all

# Baseline
bash run_vidhalluc.sh baseline all
```