#!/usr/bin/env python3
"""
Download CogNet pre-trained checkpoint.
The model weights are hosted separately due to GitHub file size limits.
"""
import os
import sys
import urllib.request
import json

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
CHECKPOINT_URL = "https://huggingface.co/cognet/cognet-40m/resolve/main/cognet_best.pt"
TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, 'tokenizer_v3.json')

def download_checkpoint():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Copy tokenizer from repo if not in checkpoints
    if not os.path.exists(TOKENIZER_PATH):
        repo_tokenizer = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer_v3.json')
        if os.path.exists(repo_tokenizer):
            import shutil
            shutil.copy2(repo_tokenizer, TOKENIZER_PATH)
            print(f"Tokenizer copied to {TOKENIZER_PATH}")
    
    dest = os.path.join(CHECKPOINT_DIR, 'cognet_best.pt')
    if os.path.exists(dest):
        print(f"Checkpoint already exists at {dest}")
        return
    
    print(f"Downloading CogNet checkpoint (~159MB)...")
    print(f"URL: {CHECKPOINT_URL}")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded/1e6:.1f}/{total_size/1e6:.1f} MB)")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(CHECKPOINT_URL, dest, report_progress)
    print(f"\nCheckpoint saved to {dest}")
    print(f"Size: {os.path.getsize(dest)/1e6:.1f} MB")

if __name__ == '__main__':
    download_checkpoint()
