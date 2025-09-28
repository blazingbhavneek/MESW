from huggingface_hub import snapshot_download

local = snapshot_download(
    repo_id="Qwen/Qwen3-Embedding-0.6B",
    local_dir="models/embeddings",
    allow_patterns=["*.json", "*.py", "tokenizer*", "config*"]  # exclude weight files
)
print("Repository cloned to", local)
