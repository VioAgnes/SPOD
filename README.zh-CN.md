# SPOD: Recovering Suppressed Temporal Evidence in Video-LLMs via Single-Pass Orthogonal Decomposition

[English README](README.md)

## 1. 介绍

视频大语言模型容易偏向静态视觉线索，忽略动态时序证据，从而生成时间关系不一致的回答。**SPOD** 通过对隐藏表示进行正交子空间分解，恢复受到抑制的时序证据。

| 组件 | 说明 |
| :--- | :--- |
| **FDA** | 帧区分性分析：通过衡量中间层隐藏状态的可区分性，识别时序显著帧。 |
| **TDS** | 时序差分子空间：基于加权帧间差分，构建正交的时序子空间 `T` 与先验子空间 `P`。 |

隐藏表示被分解为视觉证据、问题先验和残差分量，通过可配置增益重新组合后送入语言模型输出层。干预覆盖预填充位置和有效解码步骤，无需重写历史 KV。

## 2. Setup

### 2.1 环境配置

使用 Linux 与 CUDA，在仓库根目录执行：

```bash
conda create -n spod python=3.10 -y
conda activate spod

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e . --no-deps
```

参考环境为 PyTorch 2.4.0+cu124、Transformers 4.51.3 和 NumPy 1.26.4。

### 2.2 数据集

从 VidHalluc [官方 Hugging Face 仓库](https://huggingface.co/datasets/chaoyuli/VidHalluc) 下载数据。该仓库由 [项目主页](https://vid-halluc.github.io/) 提供。

| 文件 | 下载链接 |
| :--- | :--- |
| 标注文件 | [ach_binaryqa.json](https://huggingface.co/datasets/chaoyuli/VidHalluc/resolve/main/ach_binaryqa.json?download=true)、[ach_mcq.json](https://huggingface.co/datasets/chaoyuli/VidHalluc/resolve/main/ach_mcq.json?download=true)、[sth.json](https://huggingface.co/datasets/chaoyuli/VidHalluc/resolve/main/sth.json?download=true)、[tsh.json](https://huggingface.co/datasets/chaoyuli/VidHalluc/resolve/main/tsh.json?download=true) |
| 动作视频 | [ACH_videos.zip](https://huggingface.co/datasets/chaoyuli/VidHalluc/resolve/main/data/ACH_videos.zip?download=true) |
| 场景转换视频 | [STH_videos.zip](https://huggingface.co/datasets/chaoyuli/VidHalluc/resolve/main/data/STH_videos.zip?download=true) |
| 时序视频 | [TSH_videos.zip](https://huggingface.co/datasets/chaoyuli/VidHalluc/resolve/main/data/TSH_videos.zip?download=true) |

解压视频后，将标注和视频按以下结构放置：

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

使用外部数据目录时设置：

```bash
export VIDHALLUC_DATA_DIR=/path/to/datasets/VidHalluc
```

### 2.3 模型下载

| 模型 | 用途 | 下载链接 |
| :--- | :--- | :--- |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 视频理解与回答生成 | [模型文件](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main) |
| [sup-simcse-roberta-large](https://huggingface.co/princeton-nlp/sup-simcse-roberta-large) | STH 评测中的句子向量计算 | [模型文件](https://huggingface.co/princeton-nlp/sup-simcse-roberta-large/tree/main) |

下载完整权重、配置、分词器和处理器文件，放入：

```text
models/
  Qwen2.5-VL-7B-Instruct/
  sup-simcse-roberta-large/
```

使用外部模型目录时设置：

```bash
export VIDHALLUC_MODEL_DIR=/path/to/models
```

## 3. 实验

### VidHalluc 推理

```bash
# 小规模运行：每个子集选取 8 条输入
SMOKE=1 OUTPUT_DIR=outputs/demo bash scripts/inference.sh

# 单卡全量推理
bash scripts/inference.sh

# 四卡全量推理
NUM_GPUS=4 OUTPUT_DIR=outputs/vidhalluc_4gpu bash scripts/inference.sh
```

### VidHalluc 评测

```bash
# 评测单卡运行的输出
bash scripts/evaluate.sh

# 评测四卡运行的输出
OUTPUT_DIR=outputs/vidhalluc_4gpu bash scripts/evaluate.sh
```

## 4. 仓库结构

```text
configs/
  spod.json                 # 秩 8 方法配置
src/vidhalluc/
  interventions/            # 证据分解与注意力干预
  decoding/                 # 视频解码与回退处理
  evaluators/               # VidHalluc 官方评分函数
  diagnostics/              # 答案格式分析
  resources/                # 评测输入 ID 与提示词
  inference.py              # 模型推理
  evaluation.py             # 评测入口
  cli.py                    # 命令行入口
scripts/
  inference.sh              # 单卡与多卡推理
  evaluate.sh               # 官方评分
  check_environment.py      # 环境与文件检查
tests/                      # CPU 检查
pyproject.toml              # 包配置
requirements.txt            # Python 依赖
```
