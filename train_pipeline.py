"""
CogNet Training Pipeline
========================
Character-level language model training on WikiText-2 (or synthetic data).

Features:
- CharTokenizer: character-level tokenizer (printable ASCII + French accents + newline/tab)
- Pre-tokenization with .pt caching for fast loading
- AdamW optimizer with cosine LR schedule + warmup
- Gradient accumulation and clipping
- Checkpoint saving (best + latest) with optimizer state
- Generation test after each evaluation
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import model from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cognet_1b import CogNet1B


# ─── CharTokenizer ───────────────────────────────────────────────────────────

class CharTokenizer:
    """Character-level tokenizer: printable ASCII + French accents + newline/tab."""

    def __init__(self):
        self.chars = sorted(set(
            # Printable ASCII (32-126)
            [chr(i) for i in range(32, 127)]
            # Common French accented characters
            + list('àâäéèêëïîôùûüÿçœæÀÂÄÉÈÊËÏÎÔÙÛÜŸÇŒÆ')
            # Additional European characters
            + list('ëßñ¿«»')
            # Tab and newline
            + ['\t', '\n']
        ))
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(c, self.char_to_id.get(' ', 0)) for c in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.id_to_char.get(i, ' ') for i in ids)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'chars': self.chars,
                'vocab_size': self.vocab_size,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        tok = cls.__new__(cls)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok.chars = data['chars']
        tok.char_to_id = {c: i for i, c in enumerate(tok.chars)}
        tok.id_to_char = {i: c for i, c in enumerate(tok.chars)}
        tok.vocab_size = data['vocab_size']
        return tok


# ─── Dataset ─────────────────────────────────────────────────────────────────

class CharDataset(Dataset):
    """Fixed-length character-level dataset from pre-tokenized IDs."""

    def __init__(self, ids: List[int], seq_len: int):
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.ids[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.ids[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


# ─── Data Loading ────────────────────────────────────────────────────────────

def download_wikitext2(data_dir: str) -> Optional[str]:
    """Download WikiText-2 raw data. Returns data dir path or None on failure."""
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip'
    zip_path = os.path.join(data_dir, 'wikitext-2-raw-v1.zip')
    extract_dir = os.path.join(data_dir, 'wikitext-2-raw')

    if os.path.exists(extract_dir):
        return extract_dir

    try:
        print(f"Downloading WikiText-2 from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)
        os.remove(zip_path)
        print("Download complete.")
        return extract_dir
    except Exception as e:
        print(f"Failed to download WikiText-2: {e}")
        return None


def generate_synthetic_data() -> str:
    """Generate synthetic English + French training data."""
    random.seed(42)
    english_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "In a world of constant change, adaptation is the key to survival.",
        "The architecture of the mind remains one of science's greatest mysteries.",
        "She walked through the garden, noting every flower in bloom.",
        "Knowledge is not merely accumulated; it must be actively constructed.",
        "The river flowed gently through the valley, reflecting the sunset.",
        "Each decision we make shapes the path that lies ahead.",
        "The old library contained volumes that had not been touched in decades.",
        "Innovation often comes from the intersection of different fields.",
        "The mountain stood silent against the horizon, ancient and immovable.",
        "Patterns emerge when we observe the world with careful attention.",
        "The city never sleeps; its lights burn through the darkest nights.",
        "Understanding requires patience and a willingness to question assumptions.",
        "The wind carried the scent of rain across the open field.",
        "Memory is not a recording but a reconstruction of past experience.",
        "The scientist carefully recorded every observation in her notebook.",
        "Language shapes thought, and thought in turn reshapes language.",
        "The forest was alive with the sound of birdsong and rustling leaves.",
        "Progress is rarely linear; it spirals upward through cycles of learning.",
        "The artist found beauty in the most unexpected places.",
        "Every great journey begins with a single step forward.",
        "The ocean stretched endlessly, its surface shimmering in the light.",
        "Complexity arises from the interaction of simple components.",
        "The philosopher pondered the nature of consciousness and being.",
        "Stars dotted the night sky like diamonds scattered on velvet.",
        "The invention of writing transformed human civilization forever.",
        "Curiosity drives discovery and fuels the engine of progress.",
        "The garden was a testament to years of patient cultivation.",
        "Ideas flow like water, finding paths of least resistance.",
        "The clock tower chimed midnight, echoing across the empty square.",
    ]
    french_sentences = [
        "Le renard brun rapide saute par-dessus le chien paresseux.",
        "Dans un monde de changement constant, l'adaptation est la clé de la survie.",
        "L'architecture de l'esprit reste l'un des plus grands mystères de la science.",
        "Elle marchait dans le jardin, remarquant chaque fleur en éclosion.",
        "La connaissance n'est pas simplement accumulée; elle doit être construite activement.",
        "La rivière coulait doucement à travers la vallée, reflétant le coucher du soleil.",
        "Chaque décision que nous prenons façonne le chemin qui nous attend.",
        "L'ancienne bibliothèque contenait des volumes intouchés depuis des décennies.",
        "L'innovation vient souvent de l'intersection de différents domaines.",
        "La montagne se tenait silencieuse contre l'horizon, ancienne et immuable.",
        "Des schémas émergent quand on observe le monde avec attention.",
        "La ville ne dort jamais; ses lumières brûlent les nuits les plus sombres.",
        "Comprendre exige patience et volonté de remettre en question les hypothèses.",
        "Le vent portait l'odeur de la pluie à travers le champ ouvert.",
        "La mémoire n'est pas un enregistrement mais une reconstruction de l'expérience passée.",
        "Le scientifique a soigneusement noté chaque observation dans son carnet.",
        "Le langage façonne la pensée, et la pensée façonne le langage.",
        "La forêt était vivante du son du chant des oiseaux et des feuilles bruissantes.",
        "Le progrès est rarement linéaire; il spirale vers le haut à travers des cycles.",
        "L'artiste trouvait la beauté dans les endroits les plus inattendus.",
    ]

    lines = []
    for _ in range(500):
        if random.random() < 0.6:
            lines.append(random.choice(english_sentences))
        else:
            lines.append(random.choice(french_sentences))
        if random.random() < 0.1:
            lines.append('')

    return '\n'.join(lines)


def load_data(data_dir: str, tokenizer: CharTokenizer, seq_len: int,
              ckpt_dir: str) -> Tuple[CharDataset, CharDataset]:
    """Load and tokenize data, with pre-tokenized caching."""
    train_ids_path = os.path.join(ckpt_dir, 'train_ids.pt')
    valid_ids_path = os.path.join(ckpt_dir, 'valid_ids.pt')

    # Check for cached pre-tokenized data
    if os.path.exists(train_ids_path) and os.path.exists(valid_ids_path):
        print("Loading pre-tokenized data from cache...")
        train_ids = torch.load(train_ids_path, weights_only=True).tolist()
        valid_ids = torch.load(valid_ids_path, weights_only=True).tolist()
        print(f"  Train tokens: {len(train_ids):,}, Valid tokens: {len(valid_ids):,}")
        return CharDataset(train_ids, seq_len), CharDataset(valid_ids, seq_len)

    # Download or generate data
    wikitext_dir = download_wikitext2(data_dir)

    if wikitext_dir:
        train_path = os.path.join(wikitext_dir, 'wiki.train.raw')
        valid_path = os.path.join(wikitext_dir, 'wiki.valid.raw')

        with open(train_path, 'r', encoding='utf-8') as f:
            train_text = f.read()
        with open(valid_path, 'r', encoding='utf-8') as f:
            valid_text = f.read()
    else:
        print("Using synthetic data as fallback...")
        train_text = generate_synthetic_data()
        # Use last 20% as validation
        split_idx = int(len(train_text) * 0.8)
        valid_text = train_text[split_idx:]
        train_text = train_text[:split_idx]

    print(f"Train text length: {len(train_text):,} chars")
    print(f"Valid text length: {len(valid_text):,} chars")

    # Tokenize
    train_ids = tokenizer.encode(train_text)
    valid_ids = tokenizer.encode(valid_text)
    print(f"Train tokens: {len(train_ids):,}, Valid tokens: {len(valid_ids):,}")

    # Save pre-tokenized data
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(torch.tensor(train_ids, dtype=torch.long), train_ids_path)
    torch.save(torch.tensor(valid_ids, dtype=torch.long), valid_ids_path)
    print("Saved pre-tokenized data to cache.")

    return CharDataset(train_ids, seq_len), CharDataset(valid_ids, seq_len)


# ─── Learning Rate Schedule ──────────────────────────────────────────────────

def get_cosine_lr(step: int, warmup_steps: int, max_steps: int,
                  max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─── Training ────────────────────────────────────────────────────────────────

def train(args):
    # ── Config ──
    vocab_size = 136
    hidden_dim = 512
    num_blocks = 6
    num_channels = 6
    channel_dim = 128
    ff_dim = 1024
    routing_iters = 1
    max_adaptive_steps = 2
    max_seq_len = 192
    working_slots = 32
    episodic_slots = 64
    semantic_slots = 128
    key_dim = 256
    dropout = 0.1

    batch_size = 8
    grad_accum_steps = 4
    max_lr = 3e-4
    min_lr = 1e-5
    warmup_steps = 100
    eval_interval = 50
    max_steps = args.steps

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Tokenizer ──
    tok_path = os.path.join(ckpt_dir, 'tokenizer_v3.json')
    if os.path.exists(tok_path):
        print("Loading tokenizer from cache...")
        tokenizer = CharTokenizer.load(tok_path)
    else:
        tokenizer = CharTokenizer()
        tokenizer.save(tok_path)
        print(f"Created tokenizer with vocab_size={tokenizer.vocab_size}")
    # Override vocab_size to match tokenizer
    vocab_size = tokenizer.vocab_size

    # ── Data ──
    train_ds, valid_ds = load_data(data_dir, tokenizer, max_seq_len, ckpt_dir)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False
    )

    # ── Model ──
    model = CogNet1B(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_channels=num_channels,
        channel_dim=channel_dim,
        ff_dim=ff_dim,
        routing_iters=routing_iters,
        max_adaptive_steps=max_adaptive_steps,
        max_seq_len=max_seq_len,
        working_slots=working_slots,
        episodic_slots=episodic_slots,
        semantic_slots=semantic_slots,
        key_dim=key_dim,
        dropout=dropout,
    ).to(device)

    param_count = model.count_parameters()
    print(f"Model parameters: {param_count['total']:,}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, weight_decay=0.1, betas=(0.9, 0.95)
    )

    # ── Resume ──
    start_step = 0
    best_val_loss = float('inf')

    latest_path = os.path.join(ckpt_dir, 'cognet_latest.pt')
    best_path = os.path.join(ckpt_dir, 'cognet_best.pt')
    opt_path = os.path.join(ckpt_dir, 'optimizer.pt')

    # Prefer latest over best for resuming
    resume_path = latest_path if os.path.exists(latest_path) else (
        best_path if os.path.exists(best_path) else None
    )

    if resume_path:
        print(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt.get('step', 0) + 1
        best_val_loss = ckpt.get('metrics', {}).get('val_loss', float('inf'))
        if isinstance(best_val_loss, dict):
            best_val_loss = best_val_loss.get('val_loss', float('inf'))

        if os.path.exists(opt_path):
            try:
                opt_ckpt = torch.load(opt_path, map_location=device, weights_only=False)
                optimizer.load_state_dict(opt_ckpt['optimizer_state_dict'])
                print(f"  Loaded optimizer state from step {opt_ckpt.get('step', '?')}")
            except Exception as e:
                print(f"  Could not load optimizer state: {e}")

        print(f"  Resumed from step {start_step}, best_val_loss={best_val_loss:.4f}")

    # ── Training Loop ──
    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    log_interval = 10

    print(f"\nTraining for {max_steps} steps (starting at step {start_step})...")
    print("-" * 60)

    for step in range(start_step, max_steps):
        # Get batch (recreate iterator when exhausted)
        try:
            batch_x, batch_y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_x, batch_y = next(train_iter)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward
        result = model(batch_x)
        logits = result['logits']

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            batch_y.view(-1),
            ignore_index=-1
        )

        # Scale for gradient accumulation
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        running_loss += loss.item()

        # Optimizer step with accumulation
        if (step + 1) % grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update LR
            lr = get_cosine_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Logging
        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            lr = get_cosine_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            print(f"Step {step+1:5d} | loss={avg_loss:.4f} | lr={lr:.6f}")
            running_loss = 0.0

        # Evaluation + checkpoint
        if (step + 1) % eval_interval == 0 or step == max_steps - 1:
            val_loss = evaluate(model, valid_loader, device, vocab_size)
            print(f"\n  Evaluation at step {step+1}: val_loss={val_loss:.4f}")

            # Save latest
            metrics = {
                'train_loss': running_loss / max(log_interval, 1),
                'val_loss': val_loss,
                'step': step,
            }
            save_checkpoint(model, optimizer, step, metrics, vocab_size,
                            hidden_dim, num_blocks, max_seq_len,
                            tokenizer.vocab_size, latest_path, opt_path)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, metrics, vocab_size,
                                hidden_dim, num_blocks, max_seq_len,
                                tokenizer.vocab_size, best_path, opt_path=None)
                print(f"  New best model! val_loss={val_loss:.4f}")

            # Generation test
            test_generation(model, tokenizer, device, max_seq_len)

            model.train()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             vocab_size: int) -> float:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            result = model(batch_x)
            logits = result['logits']

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                batch_y.view(-1),
                ignore_index=-1,
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += batch_y.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    step: int, metrics: Dict, vocab_size: int,
                    hidden_dim: int, num_blocks: int, max_seq_len: int,
                    tokenizer_vocab_size: int, path: str,
                    opt_path: Optional[str] = None):
    """Save model checkpoint."""
    ckpt = {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'num_blocks': num_blocks,
        'max_seq_len': max_seq_len,
        'tokenizer_vocab_size': tokenizer_vocab_size,
        'metrics': metrics,
        'step': step,
    }
    torch.save(ckpt, path)

    # Save optimizer state separately
    if opt_path:
        opt_ckpt = {
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
        }
        torch.save(opt_ckpt, opt_path)


@torch.no_grad()
def test_generation(model: nn.Module, tokenizer: CharTokenizer,
                    device: torch.device, max_seq_len: int):
    """Test generation after evaluation."""
    model.eval()
    prompts = ["The ", "In ", "Le "]

    print("  ── Generation Test ──")
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        output_ids = model.generate(
            input_ids, max_new_tokens=60, temperature=0.8, top_k=20
        )
        text = tokenizer.decode(output_ids[0].tolist())
        # Truncate for display
        display = text[:120].replace('\n', '\\n')
        print(f"  '{prompt}' → {display}")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CogNet Training Pipeline')
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of training steps (default: 100)')
    args = parser.parse_args()
    train(args)
