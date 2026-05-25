# model/agastya_gpt.py
import torch
import torch.nn as nn
from torch.nn import functional as F

# 1. Hyperparameters (Small scale for quick learning on CPU)
batch_size = 16    # How many text chunks to process in parallel
block_size = 8     # Maximum context length (history window)
max_iters = 600    # Training steps
eval_interval = 100
learning_rate = 1e-3
n_embd = 32        # Character trait dimensions
n_head = 4         # Number of attention heads working in parallel
n_layer = 3        # Number of sequential Transformer blocks stacked up

# 2. Load and Tokenize custom text
with open('dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [char_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_char[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# 3. Transformer Structural Classes
class CausalHead(nn.Module):
    """ One head of masked causal self-attention """
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
    """ Running multiple attention heads in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CausalHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # Project heads back together

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """ Complete Block: Communication + Computation """
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

# 4. Main GPT Model Definition
class AgastyaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Stack layers sequentially using nn.Sequential
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # Maps traits back to characters

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_emb + pos_emb # Combine character identity and layout order
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop current context window down to the size our positional embeddings can read
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Instantiate and configure Optimizer
model = AgastyaGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("--- Phase 1: Training Agastya GPT Engine ---")
for iter in range(max_iters + 1):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter % eval_interval == 0:
        print(f"Step {iter:4d} | Training Loss: {loss.item():.4f}")

print("\n--- Phase 2: Generating Text From Context ---")
# Prompting Agastya with the word "Who " to kick off natural generation
start_context = torch.tensor([encode("Who ")], dtype=torch.long)
generated_ids = model.generate(start_context, max_new_tokens=120)[0].tolist()
print(decode(generated_ids))