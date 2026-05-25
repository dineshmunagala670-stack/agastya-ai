# Project Agastya 20M 🚀

An open-source, full-stack, custom character-level autoregressive Transformer language model platform built from scratch in PyTorch. Project Agastya features a real-time token-streaming FastAPI backend, a sleek developer telemetry dashboard built with Next.js, and an automated dynamic MLOps pipeline that pushes updates to GitHub and hot-swaps active GPU VRAM layers on the fly.

---

## 🧠 Model Architecture & Topology

Unlike typical projects that call pre-made APIs, Agastya's neural network structure is fully custom-defined block-by-block using raw PyTorch tensors.

* **Total Parameters:** 20,246,144 (20M)
* **Structural Attention Layers:** 12 stacked Transformer Blocks
* **Attention Mechanism:** Multi-Head Causal Self-Attention (6 Heads, 64 dimensions per head channel width)
* **Internal Hidden Dimension ($d_{model}$):** 384
* **Context Window Horizon:** 256 tokens (Character-level sequencing tokens)

---

## 📁 Repository Structure

```text
├── dataset/
│   ├── generate_chat_data.py  # Script to compile and synthesize training pairs
│   └── input.txt              # Unified text corpus for the model vocabulary
├── frontend/                  # Next.js web interface client application folder
│   ├── src/app/page.tsx       # Live UI dashboard with auto-scroll and MLOps sync
│   └── package.json           # Frontend Node dependency mappings
├── model/                     # Managed weights directory (Tracked via Git LFS)
│   ├── agastya_pretrained.pth # 128-context base attention layer weights
│   ├── agastya_final_chatbot.pth # Fine-tuned production chatbot weights
│   └── vocab_config.pt        # Dynamic character-to-index mapping dictionary
├── finetune_agastya.py        # Optimizing script with weight surgery & automated Git sync
└── main_api.py                # FastAPI server driving non-blocking token streams
