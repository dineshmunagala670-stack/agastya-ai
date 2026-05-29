import os
import torch
from tokenizers import Tokenizer

# Pulling in your native architecture configurations
from model_v2 import AgastyaV2BERT, AgastyaConfig

def interactive_inference():
    # 1. Establish asset routes
    CONFIG_PATH = "config.json"
    TOKENIZER_PATH = "model/v2_tokenizer.json"
    SAFETENSORS_PATH = "model/model.safetensors"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📡 INITIALIZING V2 LOCAL INFERENCE INTERFACE ON: [{device.upper()}]")

    # 2. Reconstruct model schema shell
    config = AgastyaConfig(CONFIG_PATH)
    model = AgastyaV2BERT(config)
    
    # 3. Bypass legacy pickle serialization using our native safetensors loader block
    model.load_safetensors(SAFETENSORS_PATH, device=device)
    model.to(device)
    model.eval()

    # 4. Load the 8,000 capacity WordPiece codebook
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    mask_token_id = tokenizer.token_to_id("[MASK]")

    print("\n=============================================================")
    print("🤖 Agastya v2 55M Bidirectional Brain Operational.")
    print("👉 Type a sentence containing '[MASK]' to test context completion.")
    print("👉 Example: 'A pointer holds a memory [MASK] inside code.'")
    print("👉 Type 'exit' or 'quit' to terminate session.")
    print("=============================================================\n")

    while True:
        user_input = input("\nUser Input > ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Shutting down evaluation port. Code clean.")
            break

        if "[MASK]" not in user_input:
            print("⚠️ BERT Inference Requires a '[MASK]' token to parse context weights.")
            continue

        try:
            # Tokenize incoming sequence string layers
            encoding = tokenizer.encode(user_input)
            input_ids = torch.tensor([encoding.ids], dtype=torch.long).to(device)

            # Find where the user placed the target mask block inside the tensor array
            mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]

            with torch.no_grad():
                logits, _ = model(input_ids)

            # Extract predictions specifically for the mask coordinates
            for pos in mask_positions:
                mask_logits = logits[0, pos, :]
                predicted_token_id = torch.argmax(mask_logits, dim=-1).item()
                predicted_word = tokenizer.id_to_token(predicted_token_id)

                # Format clean output view
                completed_text = user_input.replace("[MASK]", f"\033[92m{predicted_word}\033[0m")
                print(f"Agastya v2 prediction fills: {completed_text}")

        except Exception as e:
            print(f"❌ Handshake processing failure: {str(e)}")

if __name__ == "__main__":
    interactive_inference()