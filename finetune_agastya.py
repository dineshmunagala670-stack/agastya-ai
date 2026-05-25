# finetune_agastya.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import subprocess
from tokenizers import Tokenizer

# 1. GLOBAL CORE CONFIGURATIONS
batch_size = 12            # Optimized to maintain a low VRAM footprint (~1.5GB)
block_size = 256           # Extended sub-word token context window horizon
max_iters = 1500           # Fast fine-tuning iterations checkpoint threshold
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 30            # Fast initialization and validation evaluations
n_embd = 384
n_head = 6
n_layer = 12
dropout = 0.2

print(f"[SYSTEM] Loading Production-Grade Sub-word Alignment Engine on: [{device.upper()}]")

# 2. SUB-WORD TOKENIZER INITIALIZATION
TOKENIZER_PATH = 'model/agastya_tokenizer.json'
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Critical Error: '{TOKENIZER_PATH}' not found. Run 'train_tokenizer.py' first to build the BPE vocabulary.")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()
print(f"[SYSTEM] Loaded BPE Sub-word Tokenizer layout. Vocabulary Density: {vocab_size} tokens.")

# 3. TEXT DATA PIPELINE FACTORY
with open('dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Encode complete training corpus using sub-word compression vectors
print("[DATASET] Encoding raw text corpus into sub-word token arrays...")
encoded_ids = tokenizer.encode(text).ids
data = torch.tensor(encoded_ids, dtype=torch.long)

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

# 4. NEURAL NETWORK ARCHITECTURE BLOCKS
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
    def forward(self, x): return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

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
    print("[INFRASTRUCTURE] Pre-trained checkpoint base file spotted. Splicing configurations...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Prune conflicting layers (Embeddings and context lengths tables shift dimensions)
    keys_to_clear = list(state_dict.keys())
    for key in keys_to_clear:
        if any(layer in key for layer in ['token_embedding_table', 'position_embedding_table', 'lm_head', 'tril']):
            state_dict.pop(key, None)
            
    model.load_state_dict(state_dict, strict=False)
    print("[SYSTEM] Loaded 12 core abstract intelligence attention blocks into the sub-word shell.")
else:
    print("[SYSTEM] No checkpoint directory weights detected. Training entirely from scratch.")

# 6. OPTIMIZATION BACKPROPAGATION ENGINE
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"[TRAINING] Booting optimization loops across {max_iters} target steps...")
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter:4d} | Global Train Loss: {losses['train']:.4f} | Validation Loss: {losses['val']:.4f}")
        
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save finalized calibrated model weights snapshot
torch.save(model.state_dict(), 'model/agastya_final_chatbot.pth')
print("\n[PIPELINE COMPLETE] Newly trained weights compiled into 'model/agastya_final_chatbot.pth'.")

# 7. AUTOMATED UNSTOPPABLE GIT MLOps MACHINE
try:
    print("\n[GIT MLOps] Target iterations accomplished. Running automated synchronization loops...")
    subprocess.run(["git", "add", "."], check=True, shell=True)
    subprocess.run(["git", "commit", "-m", "feat(mlops): automated real-time sub-word token weights adjustment sync"], check=True, shell=True)
    
    print("[GIT MLOps] Resolving historical overlaps via automated remote rebase pipeline...")
    subprocess.run(["git", "pull", "origin", "main", "--rebase"], check=True, shell=True)
    
    print("[GIT MLOps] Pushing workspace architectures and LFS assets safely to cloud remote...")
    subprocess.run(["git", "push", "origin", "main"], check=True, shell=True)
    print("\n[SUCCESS] Entire project repository successfully updated to GitHub remote channels!")
except Exception as e:
    print(f"\n[GIT ERROR] Cloud synchronization paused. Details: {e}")