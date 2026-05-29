# train_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import os

def main():
    print("=" * 60)
    print("🪙 INITIALIZING MULTI-SOURCE TOKENIZER TRAINING ENGINE")
    print("=" * 60)

    TARGET_VOCAB_SIZE = 2000  # Locked to your 38M architecture dimensions
    
    # Updated to match your exact workspace file naming
    FILE_ONE = "dataset/input.txt"
    FILE_TWO = "dataset/large_input.txt"

    # Verify both source files exist before executing the training loop
    missing_files = False
    for file_path in [FILE_ONE, FILE_TWO]:
        if not os.path.exists(file_path):
            print(f"❌ CRITICAL ERROR: Target file '{file_path}' was not found.")
            missing_files = True
            
    if missing_files:
        print("\n💡 Fix: Make sure both text files are placed inside your dataset/ directory.")
        return

    print(f"[TOKENIZER] Compiling dual-track vocabulary data from:")
    print(f"    -> Track 1: {FILE_ONE}")
    print(f"    -> Track 2: {FILE_TWO}")

    # Initialize raw Byte-Level BPE framework to handle spacing anomalies cleanly
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # Define structural special guards
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[PAD]", "<|endoftext|>", "User:", "Agastya:"], 
        vocab_size=TARGET_VOCAB_SIZE,
        min_frequency=1
    )

    # Train directly across your specified files
    target_files = [FILE_ONE, FILE_TWO]
    tokenizer.train(files=target_files, trainer=trainer)
    
    vocab_size = tokenizer.get_vocab_size()
    
    # Save optimized token map layout
    os.makedirs("model", exist_ok=True)
    save_path = "model/agastya_tokenizer.json"
    tokenizer.save(save_path)
    
    print("-" * 60)
    print(f"🎉 SUCCESS: Unified Byte-Level Tokenizer compiled completely!")
    print(f"   * Total Vocabulary Density: {vocab_size} tokens")
    print(f"   * Target Configuration Out: {save_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()