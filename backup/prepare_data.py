# prepare_data.py
import torch

# 1. Read the text file
with open('dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("--- Dataset Stats ---")
print(f"Total characters in dataset: {len(text)}")

# 2. Recreate our tokenizer mapping based on this file
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [char_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_char[i] for i in l])

# 3. Tokenize the entire text file and convert it into a PyTorch Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Converted dataset into tensor shape: {data.shape}\n")

# 4. Break the data into chunks (Context Length)
# Let's look at a block of 10 characters
block_size = 10
x = data[:block_size]
y = data[1:block_size+1] # Target is shifted by 1 character

print("--- How Agastya Sees Training Chunks ---")
print(f"Input Tokens (X):  {x.tolist()}")
print(f"Target Tokens (Y): {y.tolist()}")

print("\n--- Breaking it down step-by-step ---")
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {decode(context.tolist())} -> predict next character: '{decode([target.item()])}'")