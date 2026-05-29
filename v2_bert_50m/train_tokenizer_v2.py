import os
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

def build_v2_bert_tokenizer():
    # 1. Establish file pathing limits
    CORPUS_PATH = "../v1-causal-38m/dataset/input.txt"
    OUTPUT_DIR = "model"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "v2_tokenizer.json")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Instantiate a clean WordPiece configuration matching Google BERT style
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # 3. Configure the specialized trainer matrix
    # Expands vocab capacity directly to 8,000 token slots
    trainer = WordPieceTrainer(
        vocab_size=8000,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    print(f"🧬 Training V2 WordPiece Tokenizer against corpus: {CORPUS_PATH}")
    print("⏳ Processing frequency distributions (Target Vocab: 8,000)...")
    
    # Run the vocabulary compilation pipeline
    tokenizer.train(files=[CORPUS_PATH], trainer=trainer)

    # 4. Save the hardened configuration map to disk
    tokenizer.save(OUTPUT_FILE)
    print(f"✨ SUCCESS! BERT-compatible V2 tokenizer saved cleanly to: {OUTPUT_FILE}")
    
    # Print telemetry validations to confirm index positions
    print("\n📊 Verified Tokenizer Control IDs:")
    for token in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
        print(f" -> {token}: ID {tokenizer.token_to_id(token)}")

if __name__ == "__main__":
    build_v2_bert_tokenizer()