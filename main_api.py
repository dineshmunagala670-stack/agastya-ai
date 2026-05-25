# main_api.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn
import asyncio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n[SERVER] Launching Agastya 20M Streaming Inference Stack on: [{device.upper()}]")

# Load compiled vocabulary configurations
vocab_data = torch.load('model/vocab_config.pt', map_location=device)
char_to_int, int_to_char = vocab_data['mappings']
vocab_size = vocab_data['vocab_size']

# System Target Metrics Hyperparameters
block_size = 256
n_embd = 384
n_head = 6
n_layer = 12

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

def load_weights_into_vram():
    checkpoint_path = 'model/agastya_final_chatbot.pth'
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        cleaned_weights = {k: v for k, v in state_dict.items() if 'tril' not in k}
        model.load_state_dict(cleaned_weights, strict=False)
        model.eval()
        return True
    return False

load_weights_into_vram()

app = FastAPI(title="Agastya Streaming Core")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str

encode = lambda s: [char_to_int[c] for c in s if c in char_to_int]
decode = lambda l: ''.join([int_to_char[i] for i in l])

async def character_streamer(prompt: str):
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    idx = context
    
    for _ in range(250):
        idx_cond = idx[:, -block_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :] / 0.3
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        next_char = int_to_char[idx_next[0, 0].item()]
        
        recent_text = "".join([int_to_char[i] for i in idx[0, -7:].tolist()])
        if "User" in recent_text or "user" in recent_text:
            break
            
        yield next_char
        await asyncio.sleep(0.001)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        prompt = f"User: {request.message}\nAgastya:"
        return StreamingResponse(character_streamer(prompt), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
async def reload_weights_endpoint():
    try:
        if device == 'cuda':
            torch.cuda.empty_cache()
        if load_weights_into_vram():
            return {"status": "success", "message": "Neural layers updated instantly in VRAM."}
        raise HTTPException(status_code=404, detail="Weights binary missing.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# DYNAMIC SPECIFICATION ENDPOINT NODE
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": device,
        "context_horizon": block_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "vocab_size": vocab_size,
        "param_count": "20,246,144"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)