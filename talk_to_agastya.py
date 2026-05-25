# talk_to_agastya.py
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab_data = torch.load('model/vocab_config.pt')
char_to_int, int_to_char = vocab_data['mappings']
vocab_size = vocab_data['vocab_size']

# MATCHING NATIVE 256 BLUEPRINT CONFIGURATION
block_size, n_embd, n_head, n_layer = 256, 384, 6, 12

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
        
    def generate(self, idx, max_new_tokens, temperature=0.3):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            recent_text = "".join([int_to_char[i] for i in idx[0, -6:].tolist()])
            if "User:" in recent_text:
                idx = idx[:, :-5]
                break
        return idx

model = AgastyaGPT().to(device)
model.load_state_dict(torch.load('model/agastya_final_chatbot.pth', map_location=device))
model.eval()

print("====================================================")
print("   AGASTYA NATIVE 256 MULTI-LINE RUNTIME MODULE     ")
print("   Type 'quit' to terminate system execution loop   ")
print("====================================================")

encode = lambda s: [char_to_int[c] for c in s if c in char_to_int]
decode = lambda l: ''.join([int_to_char[i] for i in l])

while True:
    user_msg = input("\nYou: ")
    if user_msg.lower() == 'quit': break
    
    prompt = f"User: {user_msg}\nAgastya:"
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    
    print("Agastya: ", end="")
    with torch.no_grad():
        out_tokens = model.generate(context, max_new_tokens=250, temperature=0.3)[0].tolist()
    
    response = decode(out_tokens[len(encode(prompt)):])
    print(response.strip())