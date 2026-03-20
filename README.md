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



## 📧 Contact

For questions, bug reports, or collaboration ideas, contact me by email: yufei.xue[AT]connect[DoT]ust[DoT]hk