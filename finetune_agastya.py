# finetune_agastya.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import subprocess

# 1. GLOBAL CORE CONFIGURATIONS
batch_size = 12            # OPTIMIZED: Lowered from 32 to 12 to slash VRAM back to ~1.5GB and boost step speeds
block_size = 256           
max_iters = 1500           
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 30            
n_embd = 384
n_head = 6
n_layer = 12
dropout = 0.2

print(f"[SYSTEM] Loading Production-Grade 256 Alignment Engine on: [{device.upper()}]")

# 2. AUTOMATIC VOCABULARY GENERATION
if not os.path.exists('model/vocab_config.pt'):
    print("[SYSTEM] 'model/vocab_config.pt' missing. Scanning dataset/input.txt for vocabulary structures...")
    if not os.path.exists('dataset/input.txt'):
        raise FileNotFoundError("Critical Error: 'dataset/input.txt' missing. Compile your training data first.")
        
    with open('dataset/input.txt', 'r', encoding='utf-8') as f:
        text_data = f.read()
        
    unique_chars = sorted(list(set(text_data)))
    vocab_size = len(unique_chars)
    
    char_to_int = {ch: i for i, ch in enumerate(unique_chars)}
    int_to_char = {i: ch for i, ch in enumerate(unique_chars)}
    
    os.makedirs('model', exist_ok=True)
    torch.save({'mappings': (char_to_int, int_to_char), 'vocab_size': vocab_size}, 'model/vocab_config.pt')
    print(f"[SYSTEM] Created clean vocabulary configuration tracking maps. Total size: {vocab_size} unique tokens.")

# Load synchronized vocabulary indices
vocab_data = torch.load('model/vocab_config.pt', map_location=device)
char_to_int, int_to_char = vocab_data['mappings']
vocab_size = vocab_data['vocab_size']

# 3. TEXT DATA PIPELINE FACTORY
with open('dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Encode complete training corpus
data = torch.tensor([char_to_int[c] for c in text if c in char_to_int], dtype=torch.long)
n_split = int(0.9 * len(data))
train_data = data[:n_split]
val_data = data[n_split:]

def get_batch(split):
    split_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([split_data[i:i+block_size] for i in ix])
    y = torch.stack([split_data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 4. MONOLITHIC TRANSFORMER BLUEPRINT BLOCKS
class CausalHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        wei = self.query(x) @ self.key(x).transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return self.dropout(F.softmax(wei, dim=-1)) @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CausalHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x): return x + self.ffwd(self.ln2(x + self.sa(self.ln1(x))))

class AgastyaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

model = AgastyaGPT().to(device)

# 5. DYNAMIC WEIGHT SURGERY SPLICE LOOP
checkpoint_path = 'model/agastya_pretrained.pth'
if os.path.exists(checkpoint_path):
    print("[INFRASTRUCTURE] Pre-trained matrix block binary spotted. Splicing configurations...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    keys_to_clear = list(state_dict.keys())
    for key in keys_to_clear:
        if any(layer in key for layer in ['token_embedding_table', 'position_embedding_table', 'lm_head', 'tril']):
            state_dict.pop(key, None)
            
    model.load_state_dict(state_dict, strict=False)
    print("[SYSTEM] Loaded 12 core multi-head attention blocks. Context window expanded cleanly to 256.")
else:
    print("[SYSTEM] No base weights detected. Training entirely from scratch.")

# 6. OPTIMIZATION BACKPROPAGATION ENGINE
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"[TRAINING] Booting optimization loops across {max_iters} target epochs...")
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter:4d} | Global Train Loss: {losses['train']:.4f} | Validation Loss: {losses['val']:.4f}")
        
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save final calibrated model weights
torch.save(model.state_dict(), 'model/agastya_final_chatbot.pth')
print("\n[PIPELINE COMPLETE] Weights compiled into 'model/agastya_final_chatbot.pth'.")

