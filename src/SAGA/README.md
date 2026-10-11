# SAGA (NeurIPS 2026)

**Stage-wise Attention-Guided Region Sequencing for Adversarial Attacks on Large Vision-Language Models**

[![arXiv](https://img.shields.io/badge/arXiv-2602.04356-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.04356)

## News

- SAGA is accepted to NeurIPS 2026!

## Overview

SAGA uses a fixed attention map from the open-sourced model (e.g., Qwen3-VL) to guide
where and in what order adversarial perturbations are applied. It progressively
updates salient image regions under a limited perturbation budget, without access
to the target model's parameters, gradients, or attention maps. Across fourteen
LVLMs, SAGA improves targeted attack success while producing less perceptible
perturbations.

## Main Results

Results reported in the paper (Tables 1 and 2).

**Attack success rate (ASR, %, higher is better).**

**Closed-sourced LVLMs**

| Target LVLM | X-Transfer | AnyAttack | M-Attack | FOA-Attack | SAGA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemini 2.5 Flash | 1 | 4 | 36 | 35 | **49** |
| Gemini 3 Pro | 0 | 2 | 22 | 25 | **35** |
| Gemini 3.8 Flash | 0 | 4 | 31 | 35 | **43** |
| Grok 4 Fast | 1 | 7 | 44 | 45 | **59** |
| GPT-4.1 | 1 | 7 | 57 | 62 | **67** |
| GPT-5 mini | 1 | 6 | 35 | 36 | **41** |
| GPT-5.6 Luna | 0 | 4 | 9 | 8 | **12** |

**Open-sourced LVLMs**

| Target LVLM | X-Transfer | AnyAttack | M-Attack | FOA-Attack | SAGA |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-1.5-7B | 1 | 7 | 62 | 65 | **67** |
| Gemma 3-4B | 0 | 3 | 17 | 18 | **25** |
| Llama 4 Maverick | 1 | 7 | 41 | 42 | **44** |
| DeepSeek V4.1 Flash | 1 | 6 | 40 | 43 | **56** |
| Qwen3-VL-30B | 1 | 7 | 52 | 53 | **57** |
| Qwen3-VL-235B | 1 | 7 | 57 | 62 | **63** |
| Qwen3.8-27B | 0 | 7 | 42 | 42 | **47** |

**Imperceptibility.** ↓ lower is better; ↑ higher is better.

| Method | ℓ₁ ↓ | ℓ₂ ↓ | SSIM ↑ | PSNR ↑ | LPIPS ↓ | BRISQUE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| X-Transfer | 0.2026 | 0.2351 | 0.3456 | 12.73 | 0.3548 | 52.29 |
| AnyAttack | 0.0530 | 0.0561 | 0.6667 | 25.03 | 0.3354 | 27.50 |
| M-Attack | 0.0317 | 0.0372 | 0.7417 | 28.59 | 0.2279 | 26.84 |
| FOA-Attack | 0.0321 | 0.0376 | 0.7387 | 28.51 | 0.2288 | 26.71 |
| **SAGA** | **0.0269** | **0.0330** | **0.7823** | **29.65** | **0.1965** | **22.51** |

## Preparation

Set up the environment with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/jaehyun-kwak/SAGA.git
cd SAGA
uv sync --frozen --no-dev
```

## Training

Generate adversarial images with the main SAGA configuration:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_saga.sh
```

Place the clean source images in `input/images/aadcd/` before running.
Each sample saves only `original.png`, `adversarial.png`, and `metadata.json`
under `results/saga/<image_id>/`.
The released 1,000 attacked images are in [attacked_images/images](attacked_images/images).

## BibTeX

```bibtex
@inproceedings{kwak2026saga,
  title={Stage-wise Attention-Guided Region Sequencing for Adversarial Attacks on Large Vision-Language Models},
  author={Kwak, Jaehyun and Cao, Nam and Cho, Boryeong and Lee, Segyu and Ahn, Sumyeong and Yun, Se-Young},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
