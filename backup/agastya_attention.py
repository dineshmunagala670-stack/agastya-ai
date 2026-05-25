# model/agastya_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Let's simulate a batch of 1 sequence, 8 characters long, with 16 mathematical traits per character
B, T, C = 1, 8, 16 
x = torch.randn(B, T, C) # Random data representing embedded characters

# A single Head of Self-Attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

# Compute Queries, Keys, and Values
k = key(x)   # (B, T, head_size)
q = query(x) # (B, T, head_size)
v = value(x) # (B, T, head_size)

# Calculate attention scores (affinities between all positions)
wei = q @ k.transpose(-2, -1) * (head_size ** -0.5) # (B, T, T)

# CRITICAL FOR CHATBOTS: Mask out the future! 
# Agastya shouldn't see future words when predicting the next word.
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

# Apply the weights to our values to get the context-rich output
output = wei @ v

print("--- Agastya's Attention Matrix (The Communication Map) ---")
print(wei[0]) # Show how the 8 characters distribute their attention
print(f"\nOutput shape matching original sequence: {output.shape}")