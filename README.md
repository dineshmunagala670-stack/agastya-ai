# Project Agastya 38M 🚀

An open-source, full-stack, custom character-level autoregressive Transformer language model platform built from scratch in PyTorch. Project Agastya features a real-time token-streaming FastAPI backend, a sleek developer telemetry dashboard built with Next.js, and an automated dynamic MLOps pipeline that pushes updates to GitHub and hot-swaps active GPU VRAM layers on the fly.

---

## 🧠 Model Architecture & Topology

Unlike typical projects that call pre-made APIs, Agastya's neural network structure is fully custom-defined block-by-block using raw PyTorch tensors.
📊 Neural Network Matrix Specs

| Architectural Component | Specification Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **38,154,240** | Total abstract capacity for language and logic processing |
| **Hidden Channels (`n_embd`)**| **512 dims** | Internal vector space width for structural layer processing |
| **Attention Layers (`n_layer`)**| **12 Blocks** | Number of sequential transformer layers in the stack |
| **Attention Heads (`n_head`)** | **8 Heads** | Parallel context viewpoints evaluating tokens simultaneously |
| **Context Horizon (`block_size`)**| **256 Tokens** | Maximum historical memory length during active inference |
| **Vocab Dictionary Size** | **2,000 Tokens** | Total size of the trained sub-word BPE token directory |
| **Weight Precision** | **32-bit (FP32)** | Full floating-point precision for highly stable loss convergence |
| **Weights File Size** | **~145.5 MB** | Footprint of the compiled weights binary asset |

---

## 📂 Project File Structure

```text
.
├── dataset/
│   └── input.txt                 # High-density corpus containing 20k+ balanced conversations
├── model/
│   ├── agastya_tokenizer.json    # Trained Byte-Level BPE configuration layout
│   └── agastya_final_chatbot.pth # Compiled 38M parameter weight matrices binary
├── generate_chat_data.py         # Combinatoric synthetic data engine (~13% Math ratio cap)
├── train_tokenizer.py            # Tokenizer engine for building byte-level text mappings
├── finetune_agastya.py           # Core transformer training loop equipped with Git MLOps
├── main_api.py                   # FastAPI production server with async token streaming
└── talk_to_agastya.py            # CLI-based client application for terminal chat testing
