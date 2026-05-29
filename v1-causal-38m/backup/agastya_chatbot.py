# agastya_chatbot.py
import torch
import torch.nn as nn
from torch.nn import functional as F

# 1. Hyperparameters for our Chatbot Prototype
batch_size = 8
block_size = 16  # Increased slightly to look at longer conversation context
max_iters = 1000 # More iterations so it forces memorization of the answers
learning_rate = 1e-3
n_embd = 64      # Bigger embedding size for better memory
n_head = 4
n_layer = 4

# Load the Chat Dataset
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

# Transformer Components
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
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)

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
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
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
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Smart stopping: If Agastya outputs a newline, stop generating 
            # so it doesn't start fake-typing the next User question!
            if int_to_char[idx_next.item()] == '\n':
                break
        return idx

# 3. Train the Chat Model
model = AgastyaGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Optimizing Agastya for conversation... Please wait...")
for iter in range(max_iters + 1):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Training complete! Final Loss: {loss.item():.4f}")
print("====================================================")
print("             LIVE CHAT WITH AGASTYA                 ")
print("   Type 'quit' to exit the conversation context     ")
print("====================================================")

# 4. The Interactive Chat Loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'quit':
        break
    
    # Format the prompt exactly how the AI learned it
    prompt = f"User: {user_input}\nAgastya:"
    
    # Safely handle characters the model has never seen before
    cleaned_prompt = "".join([c for c in prompt if c in char_to_int])
    
    # Convert prompt to numbers and pass to generator
    context = torch.tensor([encode(cleaned_prompt)], dtype=torch.long)
    
    print("Agastya:", end="")
    generated_tokens = model.generate(context, max_new_tokens=100)[0].tolist()
    
    # Extract only the newly generated response text
    response_text = decode(generated_tokens[len(encode(cleaned_prompt)):])
    print(response_text.strip())