---
language:
- en
license: mit
tags:
- pytorch
- causal-lm
- transformer
- text-generation
- architecture-from-scratch
- mlops
pipeline_tag: text-generation
library_name: pytorch
metrics:
- perplexity
---

# 🧠 Project Agastya (38M Parameter Causal Engine)

Designed, engineered, and trained from scratch by **Dinesh**, Project Agastya is a custom-architected 38-million parameter autoregressive language model built utilizing pure PyTorch tensor layers. By pairing a custom Byte-Level Byte Pair Encoding (BPE) tokenizer with a multi-layered causal transformer topology, Agastya is optimized for high-speed local stream execution, minimal VRAM overhead, and rapid deployment inside lightweight cloud container spaces.

---

## 📊 Technical Architecture Specifications

Agastya completely bypasses generic pre-packaged wrappers like Hugging Face `transformers` classes for its core training lifecycle. Its neural layers are instantiated explicitly out of raw tensor modules using the following structural dimensions:

| Architectural Component | Specification Parameter | Functional Description |
| :--- | :--- | :--- |
| **Total Parameters** | 38,154,240 (~38M) | Active computational weight matrices |
| **Layer Depth (`n_layer`)** | 12 Blocks | Sequential Pre-LN Transformer layers |
| **Attention Heads (`n_head`)** | 8 Heads | Parallel contextual subspace windows |
| **Embedding Width (`n_embd`)** | 512 Dimensions | Hidden feature vector state width |
| **Context Horizon (`block_size`)**| 256 Sub-word Tokens | Total attention span allocation boundary |
| **Vocabulary Size (`vocab_size`)**| 2,000 Allocations | Specialized Byte-Level BPE tokenizations |
| **Tensor Precision** | 32-bit Floating-Point (FP32) | Core calculation resolution tracking |
| **Active Memory/VRAM Load** | ~184.80 MB | Full network weight footprint in execution |

---

## 🧬 Mathematical Formulation

The core calculation driving Agastya's predictive capability uses a causal Scaled Dot-Product Attention mechanism combined with a strict lower-triangular causal attention mask matrix ($IL$) to enforce autoregressive token serialization.

The attention computation maps matrices according to the following equation:

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + IL\right)V$$

Where:
* $Q$, $K$, and $V$ represent the **Query**, **Key**, and **Value** vector translations extracted out of input hidden tensors via independent linear projection planes.
* $d_k$ is the scaling dimension factor derived directly via:

$$d_k = \frac{n\_embd}{n\_head} = \frac{512}{8} = 64$$

* $IL$ maps token entries outside current causal horizons to $-\infty$ values prior to passing through the softmax classification line, ensuring future tokens remain hidden during calculation loops.

---

## 📂 Repository Structural Layout

```text
├── dataset/
│   ├── generate_chat_data.py   # Script synthesizing custom synthetic text pairs
│   ├── input.txt               # Primary core training corpus dictionary
│   └── large_input.txt         # Expanded corpus handling advanced contextual data
├── frontend/ (Next.js App)
│   ├── app/
│   │   ├── layout.tsx          # System viewport viewport mapping
│   │   └── page.tsx            # Next.js Brutalist chat stream dashboard interface
│   ├── package.json            # Client structural system dependency manifests
│   └── tailwind.config.js      # Styling mapping properties handles
├── model/ (Local Artifact Cache)
│   ├── agastya_final_chatbot.pth # Saved Pytorch tensor layer weights binary
│   └── agastya_tokenizer.json  # Saved custom trained Byte-Level BPE vocab maps
├── train_tokenizer.py          # Dual-track custom BPE engine training pipeline
├── finetune_agastya.py         # Causal cross-entropy gradient tracking train loop
├── talk_to_agastya.py          # Local interactive testing terminal handler
├── main_api.py                 # FastAPI local system loop loop back streaming server
├── register_hf_model.py        # Automated cloud artifact upload synchronization hub
└── benchmark_hf_agastya.py     # Live remote cloud model telemetry benchmarking script
