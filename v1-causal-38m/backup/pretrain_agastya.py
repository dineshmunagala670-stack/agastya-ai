# pretrain_agastya.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# Verify graphics hardware binding
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n[SYSTEM] Project Agastya 20M Engine running on hardware: [{device.upper()}]")

# RTX 4050 6GB HARMONIZED SWEET SPOT CONFIGURATION
batch_size = 32    
block_size = 128   
max_iters = 4000   
learning_rate = 4e-4 
n_embd = 384       
n_head = 6         
n_layer = 12       

# Load clean literary corpus
if not os.path.exists('dataset/large_input.txt'):
    raise FileNotFoundError("Missing training text! Run 'python dataset/fetch_large_data.py' first.")

with open('dataset/large_input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [char_to_int[c] for c in s if c in char_to_int]
decode = lambda l: ''.join([int_to_char[i] for i in l])

# Export vocabulary dictionaries so matching scripts can align perfectly
os.makedirs('model', exist_ok=True)
torch.save({'mappings': (char_to_int, int_to_char), 'vocab_size': vocab_size}, 'model/vocab_config.pt')

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    split_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([split_data[i:i+block_size] for i in ix])
    y = torch.stack([split_data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Core Network Architecture Components
class CausalHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x) 
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CausalHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
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

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

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
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 

        loss = None if targets is None else F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

model = AgastyaGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("--- Commencing 20M Foundation Pre-Training Pipeline ---")
for iter in range(max_iters + 1):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # Fast tracking loop logs every 50 steps instantly
    if iter % 50 == 0:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch('val')
            _, val_loss = model(xv, yv)
        print(f"Step {iter:4d} | Train Loss: {loss.item():.4f} | Validation Loss: {val_loss.item():.4f}")
        model.train()

torch.save(model.state_dict(), 'model/agastya_pretrained.pth')
print("\n[SUCCESS] Foundation pre-training finalized. Base weights saved to 'model/agastya_pretrained.pth'")