import os
import sys
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tokenizers import Tokenizer
from safetensors.torch import load_file

# Clean path routing to discover your v2 underscore module track
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from v2_bert_50m.model_v2 import AgastyaV2BERT, AgastyaConfig

app = FastAPI(title="Agastya Dual-Core Production Hub", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    message: str

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 UNIFIED HUB STARTUP: Initializing tensors on compute layer [{device.upper()}]")

# =====================================================================
# 🟢 MODULE 1: V1 ENGINE REGISTRY (38M Autoregressive Transformer)
# =====================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=512, n_head=6, block_size=256, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class CausalBlock(nn.Module):
    def __init__(self, d_model=512, n_head=6, block_size=256, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class AgastyaV1CausalModel(nn.Module):
    def __init__(self, vocab_size=2000, d_model=512, n_layer=12, n_head=6, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, d_model),
            wpe = nn.Embedding(block_size, d_model),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([CausalBlock(d_model, n_head, block_size, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(d_model),
        ))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

# 🟢 LOAD V1 PIPELINE ASSETS
V1_TOKENIZER_PATH = "v1-causal-38m/model/agastya_tokenizer.json"
V1_WEIGHTS_PATH = "v1-causal-38m/model/agastya_final_chatbot.pth"

print("⚙️ Initializing Cluster Core 1 [38M Causal Model]...")
v1_tokenizer = Tokenizer.from_file(V1_TOKENIZER_PATH)
v1_model = AgastyaV1CausalModel()

if os.path.exists(V1_WEIGHTS_PATH):
    v1_model.load_state_dict(torch.load(V1_WEIGHTS_PATH, map_location=device))
    print("✔ V1 standard weights verification successful.")
else:
    print("⚠️ WARNING: v1-causal-38m/model/agastya_final_chatbot.pth missing. Running V1 on randomized matrices.")

v1_model.to(device)
v1_model.eval()

# =====================================================================
# 🟡 MODULE 2: V2 ENGINE REGISTRY (55M BERT Safetensors Model)
# =====================================================================
V2_CONFIG_PATH = "v2_bert_50m/config.json"
V2_TOKENIZER_PATH = "v2_bert_50m/model/v2_tokenizer.json"
V2_SAFETENSORS_PATH = "v2_bert_50m/model/model.safetensors"

print("⚙️ Initializing Cluster Core 2 [55M BERT Safetensors Model]...")
v2_config = AgastyaConfig(V2_CONFIG_PATH)
v2_model = AgastyaV2BERT(v2_config)

if os.path.exists(V2_SAFETENSORS_PATH):
    v2_model.load_safetensors(V2_SAFETENSORS_PATH, device=device)
    print("✔ V2 safetensors verification successful.")
else:
    print("⚠️ WARNING: v2_bert_50m/model/model.safetensors missing. Running V2 on randomized matrices.")

v2_model.to(device)
v2_model.eval()

v2_tokenizer = Tokenizer.from_file(V2_TOKENIZER_PATH)
v2_mask_id = v2_tokenizer.token_to_id("[MASK]")

# =====================================================================
# 🛣️ ENDPOINTS & ROUTING LOGIC
# =====================================================================

@app.get("/")
def system_health_telemetry():
    return {
        "status": "ONLINE",
        "engines_loaded": {
            "core_1": "Agastya-v1-Causal-38M (PyTorch Core State)",
            "core_2": "Agastya-v2-BERT-55M (SafeTensors Matrix)"
        },
        "hardware_layer": device
    }

# --- V1 REAL-TIME TOKENS STREAM GENERATOR ---
@app.post("/chat")
async def legacy_v1_chat(request: InferenceRequest):
    user_prompt = request.message
    
    async def token_streamer():
        try:
            encoded = v1_tokenizer.encode(user_prompt)
            input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(device)
            
            # Autoregressive generation window loop
            for _ in range(80):
                # Ensure context doesn't spill past context constraints window
                context_cond = input_ids[:, -256:]
                with torch.no_grad():
                    logits = v1_model(context_cond)
                
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for EOS or empty space tracking bounds
                token_id_scalar = next_token_id.item()
                if token_id_scalar == 0: 
                    break
                    
                decoded_word = v1_tokenizer.id_to_token(token_id_scalar)
                if decoded_word is None:
                    break
                    
                # Yield fragment back to Next.js UI interface pipeline hooks
                yield decoded_word + " "
                await asyncio.sleep(0.02) # Soft synchronization pacing delay
                
                input_ids = torch.cat((input_ids, next_token_id), dim=1)
        except Exception as e:
            yield f"⚠️ [STREAM ERROR: {str(e)}]"

    return StreamingResponse(token_streamer(), media_type="text/plain")

# --- V2 BIDIRECTIONAL MASK PREDICTION CONTROLLER ---
@app.post("/v2/predict")
async def api_v2_predict(request: InferenceRequest):
    user_text = request.message
    
    if "[MASK]" not in user_text:
        raise HTTPException(
            status_code=400, 
            detail="The V2 BERT engine requires a literal '[MASK]' placeholder element inside your string transcript query."
        )
        
    try:
        encoding = v2_tokenizer.encode(user_text)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long).to(device)
        mask_positions = (input_ids == v2_mask_id).nonzero(as_tuple=True)[1]

        with torch.no_grad():
            logits, _ = v2_model(input_ids)

        predictions = []
        for pos in mask_positions:
            mask_logits = logits[0, pos, :]
            pred_id = torch.argmax(mask_logits, dim=-1).item()
            pred_word = v2_tokenizer.id_to_token(pred_id)
            predictions.append(pred_word if pred_word else "[UNK]")

        # Re-compile raw string, updating masked zones step-by-step
        completed_string = user_text
        for word in predictions:
            completed_string = completed_string.replace("[MASK]", word, 1)

        return {
            "engine": "Agastya-v2-BERT-55M",
            "prediction": completed_string,
            "tokens_extracted": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neural calculation error: {str(e)}")