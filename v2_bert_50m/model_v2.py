import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors.torch import save_file, load_file

class AgastyaConfig:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        self.vocab_size = config["vocab_size"]
        self.n_embd = config["n_embd"]
        self.n_head = config["n_head"]
        self.n_layer = config["n_layer"]
        self.block_size = config["block_size"]
        self.dropout = config["dropout"]

class MultiHeadAttention(nn.Module):
    """ 12-Head Bidirectional Attention Layer Matrix """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        # Combined linear projections plane
        self.qkv_projection = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.out_projection = nn.Linear(config.n_embd, config.n_embd, bias=True)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate queries, keys, values and split into 12 parallel heads
        q, k, v = self.qkv_projection(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention calculation (No causal mask for bidirectional BERT layer mapping)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_projection(y))

class FeedForward(nn.Module):
    """ 4x Expansion Multilayer Perceptron Node Layer """
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(), # Transitioned to GELU alignment matching Google BERT spec
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """ Pre-LN Bidirectional Core Encoder Block """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x

class AgastyaV2BERT(nn.Module):
    """ Unified 50M Parameter SafeTensors-Native Transformer Module """
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.language_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        print(f"🧬 AGASTYA V2 CORE INITIALIZED // Total Parameter Nodes: {self.get_parameter_count():,}")

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.block_size, f"Context input length ({T}) exceeds maximum bounds allocation ({self.block_size})."
        
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.language_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    # 💾 HIGH-PERFORMANCE SAFETENSORS MATRICES CONVERSION I/O
    def save_safetensors(self, output_path):
        """ Strips python serialization and dumps pure numeric tensors to disk safely """
        state_dict = self.state_dict()
        # Ensure all tensors are stored in continuous float buffers
        contiguous_state = {k: v.contiguous() for k, v in state_dict.items()}
        save_file(contiguous_state, output_path)
        print(f"🔒 Weights binary successfully hardened and saved to: {output_path}")

    def load_safetensors(self, input_path, device="cpu"):
        """ Zero-copy memory mapped loading sequence bypasses pickle utilities """
        tensors = load_file(input_path, device=device)
        self.load_state_dict(tensors, strict=True)
        print(f"🔓 Connected to active 50M parameter matrix maps out of: {input_path}")

if __name__ == "__main__":
    # Test script initialization bounds validation logic
    config = AgastyaConfig("config.json")
    model = AgastyaV2BERT(config)