from huggingface_hub import snapshot_download

# Basic: latest main branch into the HF cache
snapshot_path = snapshot_download(
    repo_id="Qwen/Qwen3-8B-AWQ",
    local_dir="models/",
)
print(snapshot_path)  # path to the cached snapshot directory
