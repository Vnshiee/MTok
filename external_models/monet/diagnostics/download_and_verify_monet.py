"""
download_and_verify_monet.py:
Downloads and verifies NOVAglow646/Monet-7B weights from HuggingFace Hub.
"""

import os
import sys
import time
from huggingface_hub import snapshot_download

MODEL_ID = "NOVAglow646/Monet-7B"

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting snapshot download for {MODEL_ID}...")
start_time = time.time()

local_dir = snapshot_download(
    repo_id=MODEL_ID,
    repo_type="model",
    resume_download=True,
    max_workers=4
)

elapsed = time.time() - start_time
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Download completed in {elapsed:.1f}s!")
print(f"Local cache path: {local_dir}")

# Verify safetensors shards
import glob
shards = glob.glob(os.path.join(local_dir, "*.safetensors"))
print(f"Found {len(shards)} safetensors shards:")
for s in sorted(shards):
    size_gb = os.path.getsize(s) / (1024**3)
    print(f"  - {os.path.basename(s)}: {size_gb:.2f} GB")

assert len(shards) >= 4, f"Expected at least 4 safetensors shards, got {len(shards)}"
print("Safetensors shards verification PASSED!")
