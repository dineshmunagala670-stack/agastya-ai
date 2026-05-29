# talk_to_agastya.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import time
from tokenizers import Tokenizer

block_size = 256           
n_embd = 512               
n_head = 8                 
n_layer = 12               
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 60)
print(f"🤖 INITIALIZING LOCAL INFERENCE PORT (38M ENGINE) ON: [{device.upper()}]")
print("=" * 60)

TOKENIZER_PATH = 'model/agastya_tokenizer.json'
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Critical Error: '{TOKENIZER_PATH}' missing.")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()

stop_id = tokenizer.token_to_id("<|endoftext|>")
user_id = tokenizer.token_to_id("User:")

# Mirroring Persona Dictionary for matching features locally
AGASTYA_PERSONA_ROUTES = {
    "who are you": "I am Agastya, a custom local autoregressive transformer model running on your hardware layout.",
    "what is your name": "My name is Agastya. I am a custom 38M parameter language model optimized for local streaming inference.",
    "who made you": "I was created by Dinesh as an open-source local AI model project, synthesized directly on your workstation hardware.",
    "who is your creator": "My creator is Dinesh. He architected my 12-layer neural network layout and trained me using custom PyTorch modules.",
    "hi": "Hello! Agastya core systems online. How can I assist your development workflow today?",
    "hello": "Greetings! Agastya streaming backend is operational on your local CUDA workstation. What are we building?",
    "hey": "Hey! All 38M parameters are initialized and running at maximum clock speed. What's the plan?"
}

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

model = AgastyaGPT().to(device)

WEIGHTS_PATH = 'model/agastya_final_chatbot.pth'
if os.path.exists(WEIGHTS_PATH):
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    cleaned_weights = {k: v for k, v in state_dict.items() if 'tril' not in k}
    model.load_state_dict(cleaned_weights, strict=False)
    model.eval()
    print(f"[SYSTEM] Upscaled 38M matrix successfully initialized from: {WEIGHTS_PATH}")
else:
    print("[WARNING] Local inference running on uncalibrated variables.")

print("\n🤖 Agastya Interface Operational. Type 'exit' or 'quit' to terminate chat session.")
print("-" * 60)

while True:
    try:
        user_input = input("\nUser: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit']:
            break

        print("Agastya: ", end="", flush=True)

        # Pre-execution checks for identity routing
        clean_query = user_input.lower().replace("?", "").replace("!", "")
        if clean_query in AGASTYA_PERSONA_ROUTES:
            hardcoded_text = AGASTYA_PERSONA_ROUTES[clean_query]
            for word in hardcoded_text.split(" "):
                print(word + " ", end="", flush=True)
                time.sleep(0.04)
            print()
            continue

        # Standard GPU Neural Pipeline Execution
        formatted_prompt = f"User: {user_input}\nAgastya:"
        context_ids = tokenizer.encode(formatted_prompt).ids
        idx = torch.tensor([context_ids], dtype=torch.long, device=device)

        generated_tokens = []
        previous_decoded_string = ""

        for _ in range(200):
            idx_cond = idx[:, -block_size:]
            with torch.no_grad():
                logits = model(idx_cond)[:, -1, :] / 0.5
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            next_token_id = idx_next[0, 0].item()
            
            if next_token_id == stop_id or next_token_id == user_id:
                break
                
            generated_tokens.append(next_token_id)
            
            full_decoded_string = tokenizer.decode(generated_tokens)
            next_text_chunk = full_decoded_string[len(previous_decoded_string):]
            previous_decoded_string = full_decoded_string
                
            if next_text_chunk:
                print(next_text_chunk, end="", flush=True)
        print()

    except KeyboardInterrupt:
        break