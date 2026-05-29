# benchmark_agastya.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import time
from tokenizers import Tokenizer

# 1. Configuration Setup
block_size = 256           
n_embd = 512               
n_head = 8                 
n_layer = 12               
device = 'cuda' if torch.cuda.is_available() else 'cpu'

TOKENIZER_PATH = 'model/agastya_tokenizer.json'
WEIGHTS_PATH = 'model/agastya_final_chatbot.pth'

if not os.path.exists(TOKENIZER_PATH) or not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError("❌ Missing required compiled model artifacts inside model/ directory.")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()

# 2. Re-initialize 38M Model Structure for Clean Test Boundary
class CausalHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        wei = self.query(x) @ self.key(x).transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return F.softmax(wei, dim=-1) @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CausalHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x): return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x): return x + self.ffwd(self.ln2(x + self.sa(self.ln1(x))))

class AgastyaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        return self.lm_head(self.ln_f(self.blocks(x)))

print("=" * 60)
print("🏋️‍♂️ INITIALIZING AGASTYA 38M HARDWARE BENCHMARK UTILITY")
print("=" * 60)

# Load baseline binary state properties
model = AgastyaGPT().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device), strict=False)
model.eval()

# 🧪 Execution Parameters for the Stress Test
STRESS_PROMPT = "User: Explain the computational difference between a stack data array and a memory heap allocation graph inside a local runtime environment.\nAgastya:"
TARGET_TOKENS_COUNT = 150

print(f"[WARMUP] Spinning up CUDA tensor fields...")
# Warmup pass to prime the GPU pipelines and cache mechanisms cleanly
warmup_ctx = torch.zeros((1, 10), dtype=torch.long, device=device)
with torch.no_grad():
    _ = model(warmup_ctx)

print(f"[RUNNING] Commencing batch validation loop ({TARGET_TOKENS_COUNT} tokens generation test)...")

# Track initial benchmarks
context_ids = tokenizer.encode(STRESS_PROMPT).ids
idx = torch.tensor([context_ids], dtype=torch.long, device=device)

start_time = time.time()
ttft_time = None
tokens_generated = 0

# Sync CUDA operations for ultra-precise time stamping if using an Nvidia GPU
if device == 'cuda':
    torch.cuda.synchronize()

with torch.no_grad():
    for i in range(TARGET_TOKENS_COUNT):
        idx_cond = idx[:, -block_size:]
        
        logits = model(idx_cond)[:, -1, :] / 0.5
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        idx = torch.cat((idx, idx_next), dim=1)
        tokens_generated += 1
        
        # Capture Time to First Token (TTFT) instantly on execution loop 1
        if i == 0:
            if device == 'cuda': torch.cuda.synchronize()
            ttft_time = time.time() - start_time

if device == 'cuda':
    torch.cuda.synchronize()
end_time = time.time()

# 📈 Final Calculation Analysis
total_time = end_time - start_time
tokens_per_sec = tokens_generated / total_time

print("\n" + "=" * 60)
print("📊 FINAL HARDWARE PERFORMANCE REPORT")
print("=" * 60)
print(f"🖥️  Compute Device Layer   :  {device.upper()}")
print(f"⏱️  Time to First Token     :  {ttft_time * 1000:.2f} ms")
print(f"⏳ Total Generation Time   :  {total_time:.4f} seconds")
print(f"🎯 Tokens Generated Count  :  {tokens_generated} tokens")
print(f"⚡ System Throughput Speed :  {tokens_per_sec:.2f} tokens/sec")

if device == 'cuda':
    allocated_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    max_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"💾 Active VRAM Alloc Base  :  {allocated_vram:.2f} MB")
    print(f"🚀 Peak Memory VRAM Cap    :  {max_vram:.2f} MB")
print("=" * 60)