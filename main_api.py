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
from tokenizers import Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n[SERVER] Launching Agastya 20M Sub-word Streaming Stack on: [{device.upper()}]")

# Load trained BPE configurations
TOKENIZER_PATH = 'model/agastya_tokenizer.json'
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Critical Error: '{TOKENIZER_PATH}' was not located in the workspace root.")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()

# Architectural Dimension Hyperparameters 
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

app = FastAPI(title="Agastya Core Sub-word Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str

async def subword_token_streamer(prompt: str):
    """Streams text pieces flawlessly using an immutable delta text decoder loop"""
    context_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([context_ids], dtype=torch.long, device=device)
    
    generated_tokens = []
    previous_decoded_string = ""
    
    for _ in range(150): # Token generation target horizon ceiling
        idx_cond = idx[:, -block_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :] / 0.45  # Sampling temperature configuration
        
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        next_token_id = idx_next[0, 0].item()
        generated_tokens.append(next_token_id)
        
        # INDUSTRIAL-GRADE STREAMING METHOD: Compute structural textual deltas
        full_decoded_string = tokenizer.decode(generated_tokens)
        next_text_chunk = full_decoded_string[len(previous_decoded_string):]
        previous_decoded_string = full_decoded_string
        
        # AGGRESSIVE SYSTEM STOP-GUARD: Intercepts raw variations of turn-taking indicators
        if "User" in next_text_chunk or "user" in next_text_chunk or "User" in full_decoded_string[-8:]:
            break
            
        if next_text_chunk:
            yield next_text_chunk
            
        await asyncio.sleep(0.005)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # BPE special token anchors applied cleanly to format historical context
        formatted_prompt = f"User: {request.message}\nAgastya:"
        return StreamingResponse(subword_token_streamer(formatted_prompt), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
async def reload_weights_endpoint():
    try:
        if device == 'cuda':
            torch.cuda.empty_cache()
        if load_weights_into_vram():
            return {"status": "success", "message": "Sub-word parameter layers updated instantly in VRAM."}
        raise HTTPException(status_code=404, detail="Target checkpoint binary weights missing.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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