<h1 align="center"><code>FlexInfer</code></h1>
<h3 align="center">A decoupled and configurable codebase for LLM acceleration.</h3>

## 🔙 Precursor

This repository is a secondary development effort built on top of [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), with additional features adapted from [vLLM](https://github.com/vllm-project/vllm).

To get started, please refer to the upstream projects [[Nano-vLLM]](https://github.com/GeeeekExplorer/nano-vllm).

## 🚀 Latest News

- [2026/03/18] Added [AWQ](https://arxiv.org/abs/2306.00978) 4-bit support.

## ⚙️ Installation

Create conda environment:

```bash
conda create -n flexinfer python=3.12 -y
conda activate flexinfer
```

Build env and wheels:
```bash
pip install -r requirements.txt
# install flash attn wheels from
# https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

## 🎢 Benchmark

### Setup

- Hardware: NVIDIA RTX A6000
- Model: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- Prompt: (`#bsz`, `#prefill_token`, `#decode_token`)
- Metric: tokens/sec (TPS)

### Quantization

| Engine | Prompt | Prefill TPS | Decode TPS | Total TPS |
|---|---|---|---|---|
| FP | (1, 64, 512) | 1520.39 | 63.04 | 70.68 |
| AWQ 4-bit | (1, 64, 512) | 733.8 | 93.23 | 103.40 |
| FP | (1, 64, 1024) | 1543.63 | 62.91 | 66.72 |
| AWQ 4-bit | (1, 64, 1024) | 717.43 | 93.26 | 98.36 |

## 📧 Contact

For questions, bug reports, or collaboration ideas, contact me by email: yufei.xue[AT]connect[DoT]ust[DoT]hk