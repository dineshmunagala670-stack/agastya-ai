# model/agastya_bigram.py
import torch
import torch.nn as nn
from torch.nn import functional as F

# 1. Read the training dataset from our dataset folder
with open('dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Setup Tokenizer mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [char_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_char[i] for i in l])

# Convert entire dataset to a long integer tensor
data = torch.tensor(encode(text), dtype=torch.long)

# 3. Define the Bigram Architecture
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # An embedding table is a lookup matrix of size (vocab_size x vocab_size)
        # It contains the raw probability scores (logits) for what character follows another
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (Batch Size, Time Step) matrices of integers
        logits = self.token_embedding_table(idx) # Shapes out to (B, T, Vocab_size)

        if targets is None:
            loss = None
        else:
            # PyTorch expects a specific format for cross entropy loss, so we flatten our matrices
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices forming the current conversation context
        for _ in range(max_new_tokens):
            # Get the predictions for the current text sequence
            logits, loss = self(idx)
            # Focus ONLY on the very last character's prediction array
            logits = logits[:, -1, :] # Shapes down to (B, C)
            # Apply Softmax to convert raw scores into a nice 0% to 100% probability curve
            probs = F.softmax(logits, dim=-1)
            # Sample from the probability distribution to pick the next character ID
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append the newly predicted character ID to our ongoing sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Instantiate Agastya's Bigram brain
model = BigramLanguageModel(vocab_size)

# 4. Setup Training Batch Parameters
batch_size = 4
block_size = 8  # maximum context length to pull from

def get_batch():
    # Randomly pick starting points in our dataset file
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Use AdamW optimizer - a highly popular mechanic for training language models
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

print("--- Phase 1: Agastya Generating Raw Gibberish (Before Training) ---")
context = torch.zeros((1, 1), dtype=torch.long) # start with a blank placeholder index
print(decode(model.generate(context, max_new_tokens=60)[0].tolist()))
print("\n------------------------------------------------------------\n")

print("--- Phase 2: Training Agastya Over 400 Steps ---")
for step in range(401):
    xb, yb = get_batch() # Grab a random slice of training text
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step:3d} | Current Loss: {loss.item():.4f}")

print("\n------------------------------------------------------------\n")
print("--- Phase 3: Agastya Generating Text After Training ---")
print(decode(model.generate(context, max_new_tokens=80)[0].tolist()))