#!/usr/bin/env python3
"""Train CogNet for N steps. Resumable with checkpoints."""
import sys, os, time, math, random, torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '/home/z/my-project/download/cognet')
from train_pipeline import CharTokenizer
from cognet_1b import CogNet1B

CKPT = '/home/z/my-project/download/cognet/checkpoints'
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50

def save_ckpt(model, tokenizer, name, metrics):
    path = os.path.join(CKPT, f'cognet_{name}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'hidden_dim': model.hidden_dim,
        'num_blocks': model.num_blocks,
        'max_seq_len': model.max_seq_len,
        'tokenizer_vocab_size': tokenizer.vocab_size,
        'metrics': metrics,
    }, path)
    print(f'  Saved: {path} ({os.path.getsize(path)/1e6:.1f}MB)')

# Load data
tokenizer = CharTokenizer.load(os.path.join(CKPT, 'tokenizer_v3.json'))
train_ids = torch.load(os.path.join(CKPT, 'train_ids.pt'), map_location='cpu')
valid_ids = torch.load(os.path.join(CKPT, 'valid_ids.pt'), map_location='cpu')
print(f'Data: Train {len(train_ids):,} | Valid {len(valid_ids):,}')

# Create model
model = CogNet1B(vocab_size=tokenizer.vocab_size, hidden_dim=512, num_blocks=6,
    num_channels=6, channel_dim=128, ff_dim=1024, routing_iters=1,
    max_adaptive_steps=2, max_seq_len=192, working_slots=32,
    episodic_slots=64, semantic_slots=128, key_dim=256, dropout=0.1)

# Resume - always load model from latest, best_val from best
start_step = 0; best_val = float('inf')
latest_path = os.path.join(CKPT, 'cognet_latest.pt')
best_path = os.path.join(CKPT, 'cognet_best.pt')
load_path = latest_path if os.path.exists(latest_path) else (best_path if os.path.exists(best_path) else None)
if load_path:
    ckpt = torch.load(load_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    start_step = ckpt.get('metrics', {}).get('step', 0)
# Always load best_val from best checkpoint
if os.path.exists(best_path):
    best_ckpt = torch.load(best_path, map_location='cpu')
    best_val = best_ckpt.get('metrics', {}).get('val_loss', float('inf'))
    print(f'Best val_loss: {best_val:.6f} (step {best_ckpt.get("metrics",{}).get("step","?")})')

SEQ_LEN, BS, GA, MAX_TOTAL = 128, 2, 4, 50000
COSINE_END = 10000  # LR decays to 0 by step 10k, then stays low
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.9, 0.95))
opt_path = os.path.join(CKPT, 'optimizer.pt')
if os.path.exists(opt_path) and start_step > 0:
    try: optimizer.load_state_dict(torch.load(opt_path, map_location='cpu'))
    except: pass

def lr_lambda(step):
    s = step + start_step
    if s < 200: return s / 200
    if s >= COSINE_END: return 0.05  # Min LR = 5% of peak after cosine ends
    return 0.5 * (1.0 + math.cos(math.pi * (s-200)/max(COSINE_END-200,1)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
criterion = nn.CrossEntropyLoss(ignore_index=0)

model.train()
running_loss = 0.0; max_start = len(train_ids) - SEQ_LEN - 1
t0 = time.time()

for step in range(1, N_STEPS + 1):
    optimizer.zero_grad(set_to_none=True)
    acc = 0.0
    for _ in range(GA):
        starts = torch.randint(0, max_start, (BS,))
        x = torch.stack([train_ids[s:s+SEQ_LEN] for s in starts])
        y = torch.stack([train_ids[s+1:s+SEQ_LEN+1] for s in starts])
        out = model(x)
        loss = criterion(out['logits'].view(-1, out['logits'].size(-1)), y.view(-1))
        (loss/GA).backward(); acc += loss.item()
        del out, loss, x, y
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); scheduler.step(); running_loss += acc
    gs = step + start_step
    if step % 25 == 0:
        avg = running_loss/25; ppl = math.exp(min(avg,20)); lr = scheduler.get_last_lr()[0]
        print(f'Step {gs:>5d} | Loss: {avg:.4f} | PPL: {ppl:.1f} | LR: {lr:.6f}', flush=True)
        running_loss = 0.0
    # Save checkpoint every 100 steps during training
    if step % 100 == 0 and step < N_STEPS:
        save_ckpt(model, tokenizer, 'latest', {'step': gs})
        torch.save(optimizer.state_dict(), opt_path)
        print(f'  Mid-checkpoint saved at step {gs}', flush=True)

# Eval
model.eval(); val_loss = 0.0; max_v = len(valid_ids) - SEQ_LEN - 1
with torch.no_grad():
    for _ in range(10):
        starts = torch.randint(0, max_v, (BS,))
        x = torch.stack([valid_ids[s:s+SEQ_LEN] for s in starts])
        y = torch.stack([valid_ids[s+1:s+SEQ_LEN+1] for s in starts])
        out = model(x)
        val_loss += criterion(out['logits'].view(-1, out['logits'].size(-1)), y.view(-1)).item()
val_loss /= 10; val_ppl = math.exp(min(val_loss,20))
print(f'Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f}')

gs = start_step + N_STEPS
if val_loss < best_val:
    best_val = val_loss
    save_ckpt(model, tokenizer, 'best', {'step': gs, 'val_loss': val_loss, 'val_ppl': val_ppl,
        'total_params': sum(p.numel() for p in model.parameters())})
    print('NEW BEST!')
save_ckpt(model, tokenizer, 'latest', {'step': gs})
torch.save(optimizer.state_dict(), opt_path)

for p in ['The ', 'CogNet ', 'Bonjour ']:
    ids = torch.tensor([tokenizer.encode(p)], dtype=torch.long)
    with torch.no_grad(): gen = model.generate(ids, max_new_tokens=30, temperature=0.8, top_k=30)
    print(f'  "{p}" -> "{tokenizer.decode(gen[0].tolist())}"')
print(f'Done: step {start_step}->{gs} in {time.time()-t0:.0f}s')
