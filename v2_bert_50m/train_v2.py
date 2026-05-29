import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from safetensors.torch import save_file

# Import the 50M model architecture we built in the previous step
from model_v2 import AgastyaV2BERT, AgastyaConfig

class BERTMaskedDataset(Dataset):
    """ Bidirectional Masked Language Model (MLM) Dataset Parser """
    def __init__(self, corpus_path, tokenizer_path, block_size=256, mask_prob=0.15):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.block_size = block_size
        self.mask_prob = mask_prob
        
        # Grab special token configurations from your BPE map
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]") or 4
        self.vocab_size = self.tokenizer.get_vocab_size()

        print(f"📖 Loading pre-training text assets from: {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if len(line.strip()) > 10]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        encoding = self.tokenizer.encode(line)
        tokens = encoding.ids[:self.block_size]

        # Padding truncation boundary adjustments
        if len(tokens) < self.block_size:
            tokens += [0] * (self.block_size - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()

        # Create bidirectional BERT masking matrix (15% standard allocation ratio)
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Ensure we don't mask padding elements (0)
        masked_indices &= (input_ids != 0)
        labels[~masked_indices] = -100 # PyTorch CrossEntropy ignores -100 parameters

        # Swap target token vector elements for [MASK] token IDs
        input_ids[masked_indices] = self.mask_token_id

        return input_ids, labels

def train_engine():
    # 1. Instantiate the local path settings
    CONFIG_PATH = "config.json"
    TOKENIZER_PATH = "../v1-causal-38m/model/agastya_tokenizer.json" # Reusing tokenizer configuration mapping
    CORPUS_PATH = "../v1-causal-38m/dataset/input.txt"
    OUTPUT_SAFETENSORS_DIR = "model"
    OUTPUT_SAFETENSORS_FILE = os.path.join(OUTPUT_SAFETENSORS_DIR, "model.safetensors")

    # Ensure output folder directories are established
    os.makedirs(OUTPUT_SAFETENSORS_DIR, exist_ok=True)

    # 2. Hardware Allocation Matrix Execution
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Target Compute Engine Allocated To: [{device.upper()}]")

    # 3. Model & Dataset Loading Initializations
    config = AgastyaConfig(CONFIG_PATH)
    model = AgastyaV2BERT(config).to(device)

    dataset = BERTMaskedDataset(CORPUS_PATH, TOKENIZER_PATH, block_size=config.block_size)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 4. Optimization Matrices Configurations
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    print("\n🚀 Starting Agastya v2 50M Parameters Pre-Training Loop...")
    model.train()
    
    # Simple calibration demo running through 3 steps to establish gradient checks
    epochs = 1
    for epoch in range(epochs):
        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids, labels = input_ids.to(device), labels.to(device)

            optimizer.zero_grad()
            _, loss = model(input_ids, targets=labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 1 == 0:
                print(f"📊 Epoch: {epoch+1} | Step: {step} | Masked Cross-Entropy Loss Value: {loss.item():.4f}")
            
            # Escape early for compilation baseline validations
            if step >= 5:
                break

    # 5. 🔒 EXCLUSIVE SAFETENSORS FILE OUTPUT PACKAGING
    print("\n🔒 Training baseline achieved. Initiating Safetensors weight extraction serialization...")
    
    # Strip python binary serialization structures and grab raw structural arrays
    state_dict = model.state_dict()
    contiguous_state_dict = {k: v.contiguous().cpu() for k, v in state_dict.items()}

    # Hardens the parameters directly into a clean model.safetensors file
    save_file(contiguous_state_dict, OUTPUT_SAFETENSORS_FILE)
    print(f"✨ SUCCESS! 50M parameter binary successfully saved to: {OUTPUT_SAFETENSORS_FILE}")

if __name__ == "__main__":
    train_engine()