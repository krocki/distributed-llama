#!/usr/bin/env python3
import sys
from huggingface_hub import snapshot_download

def download_model(repo_id: str, local_dir: str = "./model", revision: str = "main"):
    """
    Download a Hugging Face model to a local directory.

    Args:
        repo_id (str): Hugging Face model repo ID (e.g., "Qwen/Qwen3-30B-A3B").
        local_dir (str): Path where the model will be saved.
        revision (str): Branch, tag, or commit hash. Default is "main".
    """
    print(f"Downloading {repo_id} into {local_dir} (revision={revision}) ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        revision=revision,
        resume_download=True,
        local_dir_use_symlinks=False  # set True if you want to save space
    )
    print(f"✅ Model downloaded to: {local_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: download_model.py <repo_id> [local_dir] [revision]")
        sys.exit(1)

    repo_id = sys.argv[1]
    local_dir = sys.argv[2] if len(sys.argv) > 2 else "./model"
    revision = sys.argv[3] if len(sys.argv) > 3 else "main"

    download_model(repo_id, local_dir, revision)
