# register_hf_model.py
import os
from huggingface_hub import HfApi, login, create_repo

print("=" * 60)
print("🤗 SYSTEM: STARTING AGASTYA 38M UPLOAD CORE")
print("=" * 60)

REPO_ID = "Dinesh05976/agastya-ai"
WEIGHTS = "model/agastya_final_chatbot.pth"
TOKENIZER = "model/agastya_tokenizer.json"

# Step 1: Verify files exist locally
for path in [WEIGHTS, TOKENIZER]:
    if not os.path.exists(path):
        print(f"❌ CRITICAL ERROR: Missing local file {path}")
        exit()

# Step 2: Authenticate session
print("\n🔐 Step 1: Logging into Hugging Face Hub...")
login()

api = HfApi()

# Step 3: Ensure remote repository is ready
print(f"\n🏗️ Step 2: Verifying remote repository space: {REPO_ID}")
create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

# Step 4: Write model card configuration strings
readme_content = """---
language:
- en
tags:
- pytorch
- causal-lm
- transformer
- text-generation
pipeline_tag: text-generation
---
# Project Agastya (38M Parameter Engine)
Custom 38-million parameter autoregressive transformer model built from scratch using PyTorch layers.

## 📊 Model Architecture Specs
- **Parameters**: 38,154,240
- **Layers**: 12 Blocks
- **Heads**: 8 Parallel Heads
- **Embed Dimension**: 512 Dims
- **Context Window**: 256 Tokens
- **Vocab Size**: 2,000 Tokens
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

# Step 5: Execute file streams to the cloud matrix
print("\n🚀 Step 3: Uploading weight matrices and vocabulary configurations...")
files_to_upload = [
    (WEIGHTS, "agastya_final_chatbot.pth"),
    (TOKENIZER, "agastya_tokenizer.json"),
    ("README.md", "README.md")
]

for local_path, repo_path in files_to_upload:
    print(f"Streaming: {local_path} ---> Hub: {repo_path}...")
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="model"
        )
        print(f"   ✅ Successfully uploaded {repo_path}")
    except Exception as e:
        print(f"   ❌ Error uploading {repo_path}: {e}")
        exit()

print("\n" + "=" * 60)
print("🎉 ALL ASSETS LIVE ON HUGGING FACE HUB!")
print(f"🔗 View here: https://huggingface.co/{REPO_ID}")
print("=" * 60)