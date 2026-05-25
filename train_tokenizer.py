# train_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch
import os

def main():
    print("=" * 60)
    # Target vocabulary size for a lightweight 20M parameter model architecture
    TARGET_VOCAB_SIZE = 2000  
    DATASET_PATH = "dataset/input.txt"
    VOCAB_CONFIG_PATH = "model/vocab_config.pt"

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Critical Error: '{DATASET_PATH}' was not located in the directory tree.")

    print(f"🏋️ Training Byte-Pair Encoding (BPE) sub-word tokenizer layout on {DATASET_PATH}...")

    # 1. Initialize a baseline BPE model architecture shell
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # 2. Configure trainer parameters with standard special stop/padding structures
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[PAD]", "User:", "Agastya:"], 
        vocab_size=TARGET_VOCAB_SIZE,
        min_frequency=2
    )

    # 3. Train the model to cluster character strings together into words
    tokenizer.train(files=[DATASET_PATH], trainer=trainer)
    
    # 4. Extract finalized properties to ensure drop-in compatibility with PyTorch files
    vocab_size = tokenizer.get_vocab_size()
    print(f"[SUCCESS] Tokenizer trained completely! Final vocabulary density: {vocab_size} tokens.")

    # Save the standalone JSON architecture tracking file
    os.makedirs("model", exist_ok=True)
    tokenizer.save("model/agastya_tokenizer.json")

    # 5. Compile token dictionary matrices to protect your PyTorch files from breaking
    # This maps token IDs back to human-readable strings flawlessly
    vocab_dict = tokenizer.get_vocab()
    int_to_char = {idx: token for token, idx in vocab_dict.items()}
    
    # Pack parameters into a drop-in file format matching your current scripts
    torch.save({
        'mappings': (vocab_dict, int_to_char), 
        'vocab_size': vocab_size,
        'is_subword': True
    }, VOCAB_CONFIG_PATH)
    
    print(f"[SUCCESS] Merged sub-word map configuration binary written cleanly to: {VOCAB_CONFIG_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()