# train_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import os

def main():
    print("=" * 60)
    TARGET_VOCAB_SIZE = 2000  
    DATASET_PATH = "dataset/input.txt"

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Critical Error: '{DATASET_PATH}' not found.")

    print(f"🏋️ Training Byte-Level BPE Tokenizer on {DATASET_PATH}...")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[PAD]", "<|endoftext|>", "User:", "Agastya:"], 
        vocab_size=TARGET_VOCAB_SIZE,
        min_frequency=1
    )

    tokenizer.train(files=[DATASET_PATH], trainer=trainer)
    vocab_size = tokenizer.get_vocab_size()
    
    os.makedirs("model", exist_ok=True)
    tokenizer.save("model/agastya_tokenizer.json")
    
    print(f"[SUCCESS] Tokenizer trained completely! Vocabulary density: {vocab_size} tokens.")
    print("=" * 60)

if __name__ == "__main__":
    main()