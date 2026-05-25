# main_api.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn  # CRITICAL: Added to keep the server alive!

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n[SERVER] Launching Agastya 20M Interactive Training Hub on: [{device.upper()}]")

# Load configuration tracking blueprints
if not os.path.exists('model/vocab_config.pt'):
    raise FileNotFoundError("Missing vocab_config.pt. Please run pretrain_agastya.py once first!")

vocab_data = torch.load('model/vocab_config.pt')
char_to_int, int_to_char = vocab_data['mappings']
vocab_size = vocab_data['vocab_size']

# HARMONIZED SWEET SPOT CONFIGURATION
block_size, n_embd, n_head, n_layer = 128, 384, 6, 12
is_training = False  # Global training state lock

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
    def generate(self, idx, max_new_tokens, temperature=0.3):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if int_to_char[idx_next.item()] == '\n': break
        return idx

# Initial model instantiation and weight loading
model = AgastyaGPT().to(device)
if os.path.exists('model/agastya_final_chatbot.pth'):
    model.load_state_dict(torch.load('model/agastya_final_chatbot.pth', map_location=device))
model.eval()

encode = lambda s: [char_to_int[c] for c in s if c in char_to_int]
decode = lambda l: ''.join([int_to_char[i] for i in l])

app = FastAPI(title="Agastya Interactive Training Hub")

# CORS middleware for secure communication with the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class CorrectionRequest(BaseModel):
    prompt: str
    correction: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if is_training:
        return {"response": "[SYSTEM NOTICE]: Agastya is running a neural realignment optimization step right now. Live chat is temporarily frozen."}
    
    prompt = f"User: {request.message}\nAgastya:"
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        out_tokens = model.generate(context, max_new_tokens=100, temperature=0.3)[0].tolist()
    
    response = decode(out_tokens[len(encode(prompt)):])
    return {"response": response.strip()}

@app.post("/submit-correction")
async def submit_correction(request: CorrectionRequest):
    formatted_entry = f"User: {request.prompt.strip()}\nAgastya: {request.correction.strip()}\n\n"
    os.makedirs('dataset', exist_ok=True)
    with open('dataset/input.txt', 'a', encoding='utf-8') as f:
        f.write(formatted_entry)
    return {"status": "success", "message": "Correction appended successfully to training pool array."}

def execute_live_finetune():
    global is_training, model
    is_training = True
    try:
        print("\n--- Initiating Background Human Feedback Realignment ---")
        with open('dataset/input.txt', 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        chunks = raw_content.split("User: ")
        encoded_pairs = []
        space_idx = char_to_int.get(' ', 0)
        
        for chunk in chunks:
            if chunk.strip():
                tokens = encode("User: " + chunk.strip() + "\n")
                if len(tokens) < block_size + 1:
                    tokens = tokens + [space_idx] * (block_size + 1 - len(tokens))
                encoded_pairs.append(torch.tensor(tokens[:block_size+1], dtype=torch.long))
        
        if len(encoded_pairs) == 0:
            print("[ERROR] No valid dataset objects found inside input.txt.")
            return

        # Initialize training environment from pristine pre-trained baseline
        train_model = AgastyaGPT().to(device)
        train_model.load_state_dict(torch.load('model/agastya_pretrained.pth', map_location=device))
        optimizer = torch.optim.AdamW(train_model.parameters(), lr=8e-5)
        
        train_model.train()
        for step in range(301):
            ix = torch.randint(len(encoded_pairs), (min(16, len(encoded_pairs)),))
            xb = torch.stack([encoded_pairs[i][:block_size] for i in ix]).to(device)
            yb = torch.stack([encoded_pairs[i][1:block_size+1] for i in ix]).to(device)
            
            _, loss = train_model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                print(f"   > Background Opt Step {step:3d} | Current Loss Target: {loss.item():.4f}")
        
        torch.save(train_model.state_dict(), 'model/agastya_final_chatbot.pth')
        model.load_state_dict(torch.load('model/agastya_final_chatbot.pth', map_location=device))
        model.eval()
        print("[SYSTEM SUCCESS] Hot-Reload finalized. Optimized weight parameters active.")
    except Exception as e:
        print(f"[SYSTEM FAILURE] Realignment loop crashed: {e}")
    finally:
        is_training = False

@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    global is_training
    if is_training:
        return {"status": "busy", "message": "The background engine is already compiling code parameters."}
    background_tasks.add_task(execute_live_finetune)
    return {"status": "started", "message": "Live alignment loop spawned in background core."}

@app.get("/")
def health():
    return {"status": "online", "training_lock": is_training}

# CRITICAL: This block forces the script to hold open port 8000 indefinitely!
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)