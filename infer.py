"""
CogNet Inference Engine for Next.js API
=======================================
Loads trained CogNet model and CharTokenizer, supports:
- generate: text generation with temperature/top-k sampling
- analyze: logits analysis, entropy, top predictions
- inspect: model architecture details
- info: model info without loading weights
"""

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

# Import from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cognet_1b import CogNet1B


# ─── Model Config (matches training) ────────────────────────────────────────

MODEL_CONFIG = {
    'vocab_size': 136,
    'hidden_dim': 512,
    'num_blocks': 6,
    'num_channels': 6,
    'channel_dim': 128,
    'ff_dim': 1024,
    'routing_iters': 1,
    'max_adaptive_steps': 2,
    'max_seq_len': 192,
    'working_slots': 32,
    'episodic_slots': 64,
    'semantic_slots': 128,
    'key_dim': 256,
    'dropout': 0.1,
}

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
TOKENIZER_PATH = os.path.join(CKPT_DIR, 'tokenizer_v3.json')
BEST_MODEL_PATH = os.path.join(CKPT_DIR, 'cognet_best.pt')
LATEST_MODEL_PATH = os.path.join(CKPT_DIR, 'cognet_latest.pt')


# ─── CharTokenizer (standalone, no import needed from train_pipeline) ───────

class CharTokenizer:
    """Character-level tokenizer: printable ASCII + French accents + newline/tab."""

    def __init__(self):
        self.chars = sorted(set(
            [chr(i) for i in range(32, 127)]
            + list('àâäéèêëïîôùûüÿçœæÀÂÄÉÈÊËÏÎÔÙÛÜŸÇŒÆ')
            + list('ëßñ¿«»')
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


# ─── JSON Helpers ────────────────────────────────────────────────────────────

def sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


# ─── Model Cache ─────────────────────────────────────────────────────────────

_model_cache: Dict[str, Any] = {
    'model': None,
    'tokenizer': None,
    'device': None,
    'loaded': False,
}


def load_model_and_tokenizer() -> tuple:
    """Load model and tokenizer with caching."""
    if _model_cache['loaded']:
        return _model_cache['model'], _model_cache['tokenizer'], _model_cache['device']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}. "
            "Run train_pipeline.py first to create it."
        )
    tokenizer = CharTokenizer.load(TOKENIZER_PATH)

    # Update vocab_size from tokenizer
    config = dict(MODEL_CONFIG)
    config['vocab_size'] = tokenizer.vocab_size

    # Create model
    model = CogNet1B(**config).to(device)

    # Load weights (prefer best, then latest)
    model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else LATEST_MODEL_PATH
    if model_path and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        step = ckpt.get('metrics', {}).get('step', '?')
        print(f"Loaded model from {model_path} (step={step})")
    else:
        print("WARNING: No trained weights found. Using random initialization.")

    model.eval()

    # Cache
    _model_cache['model'] = model
    _model_cache['tokenizer'] = tokenizer
    _model_cache['device'] = device
    _model_cache['loaded'] = True

    return model, tokenizer, device


# ─── Action Handlers ─────────────────────────────────────────────────────────

def handle_generate(prompt: str, max_tokens: int = 100,
                    temperature: float = 0.8, top_k: int = 20) -> Dict:
    """Generate text from a prompt."""
    model, tokenizer, device = load_model_and_tokenizer()

    # Encode prompt
    ids = tokenizer.encode(prompt)
    if len(ids) == 0:
        ids = [0]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode
    generated_ids = output_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    new_text = tokenizer.decode(generated_ids[len(ids):])

    # Token details
    token_details = []
    for i, tid in enumerate(generated_ids):
        char = tokenizer.decode([tid])
        token_details.append({
            'id': tid,
            'char': char,
            'position': i,
        })

    return sanitize_for_json({
        'action': 'generate',
        'prompt': prompt,
        'generated_text': generated_text,
        'new_text': new_text,
        'token_details': token_details,
        'num_tokens': len(generated_ids),
        'temperature': temperature,
        'top_k': top_k,
    })


def handle_analyze(prompt: str) -> Dict:
    """Analyze logits, entropy, and top predictions."""
    model, tokenizer, device = load_model_and_tokenizer()

    ids = tokenizer.encode(prompt)
    if len(ids) == 0:
        ids = [0]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        result = model(input_ids, return_stats=True)
        logits = result['logits']

    # Analyze last token's predictions
    last_logits = logits[0, -1, :]  # (vocab_size,)
    probs = F.softmax(last_logits, dim=-1)

    # Entropy
    entropy = -(probs * (probs + 1e-10).log()).sum().item()

    # Top 10 predictions
    topk_vals, topk_ids = torch.topk(probs, min(10, probs.size(0)))
    top_predictions = []
    for prob, tid in zip(topk_vals.tolist(), topk_ids.tolist()):
        top_predictions.append({
            'token_id': tid,
            'char': tokenizer.decode([tid]),
            'probability': prob,
        })

    # Per-position entropy
    all_probs = F.softmax(logits[0], dim=-1)
    pos_entropy = (-(all_probs * (all_probs + 1e-10).log()).sum(dim=-1)).tolist()

    # Stats
    stats = result.get('stats', {})
    stats_summary = {}
    for k, v in stats.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            v = None
        stats_summary[k] = v

    return sanitize_for_json({
        'action': 'analyze',
        'prompt': prompt,
        'prompt_length': len(ids),
        'entropy': entropy,
        'top_predictions': top_predictions,
        'per_position_entropy': pos_entropy,
        'model_stats': stats_summary,
    })


def handle_inspect() -> Dict:
    """Return model architecture details."""
    model, tokenizer, device = load_model_and_tokenizer()

    params = model.count_parameters()
    complexity = model.get_complexity_analysis()

    # Layer details
    layers = []
    for i, block in enumerate(model.blocks):
        layer_params = sum(p.numel() for p in block.parameters())
        layers.append({
            'block_index': i,
            'parameters': layer_params,
            'components': ['CognitiveRouter', 'SharedHierarchicalMemory',
                          'AdaptiveComputationBlock', 'CompositionalReasoner'],
        })

    return sanitize_for_json({
        'action': 'inspect',
        'architecture': 'CogNet (Non-Transformer)',
        'total_parameters': params['total'],
        'trainable_parameters': params['trainable'],
        'config': {
            'vocab_size': model.vocab_size,
            'hidden_dim': model.hidden_dim,
            'num_blocks': model.num_blocks,
            'num_channels': model.num_channels,
            'channel_dim': model.channel_dim,
            'ff_dim': model.ff_dim,
            'max_seq_len': model.max_seq_len,
            'tokenizer_vocab_size': tokenizer.vocab_size,
        },
        'complexity_analysis': complexity,
        'layers': layers,
        'device': str(device),
    })


def handle_info() -> Dict:
    """Return model info without loading weights."""
    config = dict(MODEL_CONFIG)

    # Check what's available
    has_tokenizer = os.path.exists(TOKENIZER_PATH)
    has_best = os.path.exists(BEST_MODEL_PATH)
    has_latest = os.path.exists(LATEST_MODEL_PATH)

    # Estimate param count without loading
    model = CogNet1B(**config)
    params = model.count_parameters()

    # Check checkpoint info if available
    checkpoint_info = {}
    if has_best:
        try:
            ckpt = torch.load(BEST_MODEL_PATH, map_location='cpu', weights_only=False)
            checkpoint_info['best'] = {
                'step': ckpt.get('metrics', {}).get('step', None),
                'val_loss': ckpt.get('metrics', {}).get('val_loss', None),
                'val_ppl': ckpt.get('metrics', {}).get('val_ppl', None),
            }
        except Exception:
            checkpoint_info['best'] = {'error': 'Could not read checkpoint'}
    if has_latest:
        try:
            ckpt = torch.load(LATEST_MODEL_PATH, map_location='cpu', weights_only=False)
            checkpoint_info['latest'] = {
                'step': ckpt.get('metrics', {}).get('step', None),
            }
        except Exception:
            checkpoint_info['latest'] = {'error': 'Could not read checkpoint'}

    return sanitize_for_json({
        'action': 'info',
        'model_name': 'CogNet',
        'architecture': 'Non-Transformer (Cognitive Routing)',
        'estimated_parameters': params['total'],
        'config': config,
        'files': {
            'tokenizer': has_tokenizer,
            'best_checkpoint': has_best,
            'latest_checkpoint': has_latest,
        },
        'checkpoint_info': checkpoint_info,
    })


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='CogNet Inference Engine')
    parser.add_argument('action', choices=['generate', 'analyze', 'inspect', 'info'],
                        help='Action to perform')
    parser.add_argument('--prompt', type=str, default='The ',
                        help='Prompt text (for generate/analyze)')
    parser.add_argument('--max-tokens', type=int, default=100,
                        help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Top-k sampling')

    args = parser.parse_args()

    if args.action == 'generate':
        result = handle_generate(args.prompt, args.max_tokens,
                                args.temperature, args.top_k)
    elif args.action == 'analyze':
        result = handle_analyze(args.prompt)
    elif args.action == 'inspect':
        result = handle_inspect()
    elif args.action == 'info':
        result = handle_info()

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
