---
language:
- en
tags:
- pytorch
- causal-lm
- transformer
- text-generation
pipeline_tag: text-generation
---

# Project Agastya (38M Parameter Autoregressive Engine)

Project Agastya is a custom, 38-million parameter autoregressive transformer platform built completely from scratch using PyTorch layers. The architecture utilizes an optimized Byte-Level BPE tokenizer to handle custom vocabulary processing and integrated token ID stop-guards to manage live conversational stream boundaries cleanly.

## 📊 Neural Network Architecture Specs
- **Total Matrix Parameters**: 38,154,240
- **Hidden Channel Dimension (`n_embd`)**: 512 channels
- **Attention Transformer Layers (`n_layer`)**: 12 Blocks
- **Attention Heads Matrix Width (`n_head`)**: 8 Parallel Heads
- **Context Window Horizon (`block_size`)**: 256 sub-word tokens
- **Vocabulary Directory Size**: 2,000 sub-word allocations
- **Weight Calculation Precision**: 32-bit Floating-Point (FP32)

## 📂 Repository File Layout
- `agastya_final_chatbot.pth` — Compiled weight matrices file (~145.5 MB).
- `agastya_tokenizer.json` — Custom trained Byte-Level BPE configuration vocabulary.

## 🏗️ Local Inference Configuration
To run local streaming tests or connect this model straight to your FastAPI backend, initialize your model architecture with these dimensions:
```python
block_size = 256
n_embd = 512
n_head = 8
n_layer = 12
