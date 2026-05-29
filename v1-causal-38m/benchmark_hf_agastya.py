# benchmark_hf_agastya.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import os
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

# --- 1. CLOUD REPOSITORY ARTIFACT RETRIEVAL ---
REPO_ID = "Dinesh05976/agastya-ai"

print("=" * 60)
print(f"🤗 FETCHING LIVE ASSETS FROM HUGGING FACE HUB: {REPO_ID}")
print("=" * 60)

try:
    print("[CLOUD] Pulling vocabulary token map configuration...")
    tokenizer_file_path = hf_hub_download(repo_id=REPO_ID, filename="agastya_tokenizer.json")
    
    print("[CLOUD] Pulling 38M weight binary layer matrices...")
    weights_file_path = hf_hub_download(repo_id=REPO_ID, filename="agastya_final_chatbot.pth")
    print("✅ Handshake successful. Model files cached locally.")
except Exception as e:
    print(f"❌ CRITICAL: Failed to download files from Hugging Face Hub: {e}")
    exit()

# --- 2. SETUP MODEL ARCHITECTURE PARAMETERS ---
block_size = 256           
n_embd = 512               
n_head = 8                 
n_layer = 12               
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer.from_file(tokenizer_file_path)
vocab_size = tokenizer.get_vocab_size()

# Core Custom Structure definitions matching your 38M model dimensions exactly
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

# --- 3. MODEL INITIALIZATION & VRAM ALLOCATION ---
print(f"\n🖥️  Initializing memory allocation profiles on: [{device.upper()}]")
model = AgastyaGPT().to(device)

# Load the weights downloaded directly from the Hugging Face Hub cache path
model.load_state_dict(torch.load(weights_file_path, map_location=device), strict=False)
model.eval()
print("🔥 Parameters successfully bound to local tensor execution lines.")

# --- 4. EXECUTE PERFORMANCE EVALUATION PASS ---
BENCHMARK_PROMPT = "User: Benchmark active throughput configuration logic on cloud hosted tensors.\nAgastya:"
TARGET_TOKENS = 100

print(f"\n🚀 Running cloud-source generation stress test ({TARGET_TOKENS} tokens)...")

context_ids = tokenizer.encode(BENCHMARK_PROMPT).ids
idx = torch.tensor([context_ids], dtype=torch.long, device=device)

if device == 'cuda': torch.cuda.synchronize()
start_perf_time = time.time()

ttft_duration = None
generated_count = 0

with torch.no_grad():
    for step in range(TARGET_TOKENS):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)[:, -1, :] / 0.5
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        generated_count += 1
        
        if step == 0:
            if device == 'cuda': torch.cuda.synchronize()
            ttft_duration = time.time() - start_perf_time

if device == 'cuda': torch.cuda.synchronize()
end_perf_time = time.time()

# --- 5. PERFORMANCE REPORT GENERATION ---
total_execution_time = end_perf_time - start_perf_time
tokens_per_second = generated_count / total_execution_time

print("\n" + "=" * 60)
print("📊 HUGGING FACE ASSET PERFORMANCE BENCHMARK REPORT")
print("=" * 60)
print(f"📦 Source Repository   :  https://huggingface.co/{REPO_ID}")
print(f"⏱️  Time to First Token  :  {ttft_duration * 1000:.2f} ms")
print(f"⏳ Processing Duration  :  {total_execution_time:.4f} seconds")
print(f"⚡ Throughput Velocity  :  {tokens_per_second:.2f} tokens/sec")

if device == 'cuda':
    vram_used = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"💾 Active VRAM Burden   :  {vram_used:.2f} MB")
print("=" * 60)