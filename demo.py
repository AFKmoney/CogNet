#!/usr/bin/env python3
"""
Quick demo of CogNet text generation.
Usage: python demo.py
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cognet_1b import CogNet1B
from infer import CharTokenizer

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')

def main():
    # Load tokenizer
    tokenizer = CharTokenizer.load(os.path.join(CKPT_DIR, 'tokenizer_v3.json'))
    print(f"Tokenizer loaded: {tokenizer.vocab_size} chars")

    # Create model
    model = CogNet1B(
        vocab_size=tokenizer.vocab_size, hidden_dim=512, num_blocks=6,
        num_channels=6, channel_dim=128, ff_dim=1024, routing_iters=1,
        max_adaptive_steps=2, max_seq_len=192, working_slots=32,
        episodic_slots=64, semantic_slots=128, key_dim=256, dropout=0.1
    )
    
    # Load weights
    ckpt_path = os.path.join(CKPT_DIR, 'cognet_best.pt')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        print("Run: python download_checkpoint.py")
        return
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {params:,} parameters")
    print()

    # Generate text
    prompts = [
        "The ",
        "CogNet is",
        "Once upon a time",
        "In the beginning",
        "Science tells us",
        "Bonjour ",
    ]
    
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=60, temperature=0.7, top_k=40)
        text = tokenizer.decode(gen[0].tolist())
        print(f'Prompt: "{prompt}"')
        print(f'Output: "{text}"')
        print()

if __name__ == '__main__':
    main()
