# model/agastya_transformer_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        # Standard expansion and compression to let neurons process data deeply
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class CausalAttentionHead(nn.Module):
    """ A single head of masked self-attention """
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register a buffer ensures the mask grid is saved along with the model
        self.register_buffer('tril', torch.tril(torch.ones(8, 8))) # block size = 8

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Calculate scores, mask future, and apply softmax
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x) # (B, T, head_size)
        return wei @ v

class AgastyaBlock(nn.Module):
    """ A single complete Transformer layer """
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.sa_head = CausalAttentionHead(n_embd, head_size)
        self.ffwd = FeedForward(n_embd)
        # Layer normalization to stabilize training calculations
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Communication step + Residual connection (adding the original input back)
        x = x + self.sa_head(self.ln1(x))
        # Computation step + Residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

# Test run the block structural pipeline
B, T, C = 1, 8, 16
x = torch.randn(B, T, C)
block = AgastyaBlock(n_embd=C, head_size=C)
output = block(x)

print("--- Agastya Complete Transformer Block Output ---")
print(f"Input Shape:  {x.shape}")
print(f"Output Shape: {output.shape}")
print("\nSuccess! The data successfully communicated and computed without shifting shapes.")