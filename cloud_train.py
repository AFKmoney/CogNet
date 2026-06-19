#!/usr/bin/env python3
"""
CogNetAICL cloud training entry point.

One-shot script for fine-tuning CogNet on the AICL corpus, designed to run on
a GPU cloud instance (RunPod / Lambda Labs / Vast.ai / GCP / AWS). Clones the
repos if missing, installs deps, generates a fresh corpus, and launches the
training pipeline.

QUICK START (on a fresh Ubuntu GPU instance):
    git clone https://github.com/AFKmoney/CogNet.git
    cd CogNet
    pip install -r requirements_aicl.txt
    python cloud_train.py --steps 5000

This will:
  1. Clone AFKmoney/AICL (the compiler) if not present
  2. Generate the AICL corpus (30 algorithms -> spec+code pairs)
  3. Install the AICL compiler in editable mode
  4. Lay out the corpus for the training pipeline
  5. Launch train_pipeline.py with GPU auto-detection

CHECKPOINTS are saved to ./checkpoints/ (cognet_best.pt, cognet_latest.pt).
The best checkpoint is the one with lowest validation loss.

To RESUME training (segment-based, lossless — "repair the car while driving"):
    python cloud_train.py --steps 5000 --resume

To use the MULTI-TARGET corpus (spec -> Python AND Rust):
    python cloud_train.py --steps 5000 --multitarget
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, cwd=None, check=True):
    """Run a command, streaming output, raising on failure."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if check and result.returncode != 0:
        sys.exit(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result.returncode


def ensure_repos():
    """Clone AICL and CogNet repos if not present."""
    aicl_dir = os.path.join(os.path.dirname(HERE), "AICL")
    if not os.path.isdir(aicl_dir):
        print("[cloud_train] cloning AFKmoney/AICL...")
        run(["git", "clone", "--depth", "1",
             "https://github.com/AFKmoney/AICL.git", aicl_dir])
    else:
        print(f"[cloud_train] AICL already present at {aicl_dir}")
    aicl_python = os.path.join(aicl_dir, "python")
    return aicl_python


def install_compiler(aicl_python):
    """Install the AICL compiler in editable mode."""
    print("[cloud_train] installing AICL compiler...")
    run([sys.executable, "-m", "pip", "install", "-e", aicl_python, "-q"])


def generate_corpus(aicl_python, multitarget=False):
    """Run the corpus generator and return the path to the .raw file."""
    targets = "python rust" if multitarget else "python"
    name = "aicl_corpus_multitarget.raw" if multitarget else "aicl_corpus.raw"
    out = os.path.join(HERE, name)
    print(f"[cloud_train] generating corpus (targets: {targets})...")
    run([sys.executable,
         os.path.join(aicl_python, "tools", "corpus_generator.py"),
         "--out", out, "--targets"] + targets.split(),
        cwd=aicl_python)
    # copy into CogNet dir for the training hook
    return out


def prepare_data(corpus_path):
    """Lay out the corpus as wiki.train.raw / wiki.valid.raw for the pipeline."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    cut = int(len(text) * 0.9)
    wt_dir = os.path.join(HERE, "data", "wikitext-2-raw")
    os.makedirs(wt_dir, exist_ok=True)
    with open(os.path.join(wt_dir, "wiki.train.raw"), "w", encoding="utf-8") as f:
        f.write(text[:cut])
    with open(os.path.join(wt_dir, "wiki.valid.raw"), "w", encoding="utf-8") as f:
        f.write(text[cut:])
    # clear cached tokenization so the new corpus re-tokenizes
    for cache in ("train_ids.pt", "valid_ids.pt"):
        c = os.path.join(HERE, "checkpoints", cache)
        if os.path.exists(c):
            os.remove(c)
    print(f"[cloud_train] corpus split: train={cut:,} chars, valid={len(text)-cut:,} chars")


def main():
    parser = argparse.ArgumentParser(
        description="CogNetAICL cloud training (one-shot, GPU-ready).")
    parser.add_argument("--steps", type=int, default=5000,
                        help="training steps (default 5000; ~2-4h on a single GPU)")
    parser.add_argument("--multitarget", action="store_true",
                        help="use the multi-target corpus (spec -> Python + Rust)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from latest checkpoint (segment-based)")
    parser.add_argument("--skip-setup", action="store_true",
                        help="skip repo clone/install (assume already done)")
    args = parser.parse_args()

    if not args.skip_setup:
        aicl_python = ensure_repos()
        install_compiler(aicl_python)
    else:
        # assume AICL already installed
        aicl_python = os.path.join(os.path.dirname(HERE), "AICL", "python")

    corpus = generate_corpus(aicl_python, multitarget=args.multitarget)
    prepare_data(corpus)

    # Launch the training pipeline. It auto-detects CUDA.
    print(f"\n[cloud_train] launching training: {args.steps} steps")
    cmd = [sys.executable, os.path.join(HERE, "train_pipeline.py"),
           "--steps", str(args.steps)]
    run(cmd)


if __name__ == "__main__":
    main()
