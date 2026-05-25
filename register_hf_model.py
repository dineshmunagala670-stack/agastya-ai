# register_hf_model.py
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

# 1. Define a standardized configuration tracking file for the cloud
class AgastyaConfig(PretrainedConfig):
    model_type = "agastya"
    def __init__(self, vocab_size=256, block_size=256, n_embd=384, n_head=6, n_layer=12, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

# 2. Wrap your structural PyTorch layers into the HF standard module
class AgastyaHFModel(PreTrainedModel):
    config_class = AgastyaConfig
    
    def __init__(self, config):
        super().__init__(config)
        # Explicitly map your existing Agastya architecture blocks here
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        
        # Reference your existing TransformerBlock initialization array
        # (Assuming your block and head layers from talk_to_agastya are imported or included here)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
    def forward(self, idx, labels=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=idx.device))
        # Pass through your transformer blocks...
        logits = self.lm_head(self.ln_f(x))
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        # Return standard Hugging Face output format
        return CausalLMOutputWithPast(loss=loss, logits=logits)