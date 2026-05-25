# finetune_agastya.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n[SYSTEM] Loading Production-Grade 256 Alignment Engine on: [{device.upper()}]")

# Load configuration tracking blueprints
vocab_data = torch.load('model/vocab_config.pt')
char_to_int, int_to_char = vocab_data['mappings']
vocab_size = vocab_data['vocab_size']

# TARGET SCALED BLUEPRINT CONFIGURATION
block_size = 256   # Double the memory window to prevent prose-bleeding
batch_size = 16    
learning_rate = 8e-5 
finetune_iters = 1500 
n_embd, n_head, n_layer = 384, 6, 12

if not os.path.exists('dataset/input.txt'):
    raise FileNotFoundError("Missing 'dataset/input.txt'. Run your chat data generation script first!")

with open('dataset/input.txt', 'r', encoding='utf-8') as f:
    raw_content = f.read()

chunks = raw_content.split("User: ")
encoded_pairs = []
space_idx = char_to_int.get(' ', 0)

for chunk in chunks:
    if chunk.strip():
        full_pair = "User: " + chunk.strip() + "\n"
        tokens = [char_to_int[c] for c in full_pair if c in char_to_int]
        if len(tokens) < block_size + 1:
            tokens = tokens + [space_idx] * (block_size + 1 - len(tokens))
        else:
            tokens = tokens[:block_size + 1]
        encoded_pairs.append(torch.tensor(tokens, dtype=torch.long))

def get_batch():
    ix = torch.randint(len(encoded_pairs), (batch_size,))
    xb = torch.stack([encoded_pairs[i][:block_size] for i in ix])
    yb = torch.stack([encoded_pairs[i][1:block_size+1] for i in ix])
    return xb.to(device), yb.to(device)

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
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

# Initialize shell structure at 256
model = AgastyaGPT().to(device)

# SAFE LOADING: Adapt 128 baseline to 256 architecture before training starts
pretrained_path = 'model/agastya_pretrained.pth'
if os.path.exists(pretrained_path):
    state_dict = torch.load(pretrained_path, map_location=device)
    pos_weight = state_dict.get('position_embedding_table.weight')
    
    if pos_weight is not None and pos_weight.shape[0] == 128:
        print("[INFRASTRUCTURE] Expanding pre-trained position weight maps from 128 to 256 structural dimensions...")
        new_pos_weight = torch.zeros(256, 384, device=device)
        new_pos_weight[:128, :] = pos_weight
        new_pos_weight[128:, :] = pos_weight[-1, :].unsqueeze(0)
        state_dict['position_embedding_table.weight'] = new_pos_weight
        
        # Strip old context configuration rules out
        for k in [key for key in state_dict.keys() if 'tril' in key]:
            del state_dict[k]
            
    model.load_state_dict(state_dict, strict=False)
else:
    print("[WARNING] Initializing with raw randomized matrix weights.")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Loaded {len(encoded_pairs)} unique items. Injecting training passes...")
model.train()
for iter in range(finetune_iters + 1):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if iter % 100 == 0:
        print(f"Alignment Step {iter:4d} | Pure Conversational Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'model/agastya_final_chatbot.pth')
print("\n[SUCCESS] Highly trained, native 256 weights saved to 'model/agastya_final_chatbot.pth'")