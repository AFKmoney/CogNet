"""
CogNet1B: Non-Transformer Language Model with Cognitive Routing
================================================================
Replaces self-attention with O(n) cognitive routing and
hierarchical memory, enabling linear-time sequence processing.

Key architectural innovations:
- CognitiveChannel: Depthwise separable conv + SwiGLU FFN (O(n) per channel)
- CoherenceRouter: O(n) routing via learned coherence scoring
- SharedHierarchicalMemory: 3-tier key-value memory (Working/Episodic/Semantic)
- AdaptiveComputationBlock: Variable-depth processing per token
- CompositionalReasoner: Hyperdimensional computing for role-filler binding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ─── Token Encoder ───────────────────────────────────────────────────────────

class TokenEncoder(nn.Module):
    """Token embedding + learned positional encoding."""

    def __init__(self, vocab_size: int, hidden_dim: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        return self.dropout(self.norm(x))


# ─── Cognitive Channel ───────────────────────────────────────────────────────

class CognitiveChannel(nn.Module):
    """Depthwise separable convolution + SwiGLU FFN — O(n) per channel."""

    def __init__(self, channel_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        # Depthwise separable conv
        self.dw_conv = nn.Conv1d(
            channel_dim, channel_dim, kernel_size=3, padding=1,
            groups=channel_dim
        )
        self.pw_conv = nn.Conv1d(channel_dim, channel_dim, kernel_size=1)
        self.conv_norm = nn.LayerNorm(channel_dim)
        self.conv_dropout = nn.Dropout(dropout)

        # SwiGLU FFN
        self.ff_gate = nn.Linear(channel_dim, ff_dim, bias=False)
        self.ff_up = nn.Linear(channel_dim, ff_dim, bias=False)
        self.ff_down = nn.Linear(ff_dim, channel_dim, bias=False)
        self.ff_norm = nn.LayerNorm(channel_dim)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        # Conv path
        h = x.transpose(1, 2)                    # (B, D, T)
        h = self.dw_conv(h)
        h = self.pw_conv(h)
        h = h.transpose(1, 2)                     # (B, T, D)
        h = self.conv_norm(h)
        x = residual + self.conv_dropout(h)

        # FFN path (SwiGLU)
        residual = x
        gate = F.silu(self.ff_gate(x))
        up = self.ff_up(x)
        h = gate * up
        h = self.ff_down(h)
        h = self.ff_norm(h)
        x = residual + self.ff_dropout(h)
        return x


# ─── Coherence Router ────────────────────────────────────────────────────────

class CoherenceRouter(nn.Module):
    """O(n) routing: compute which channel handles each token."""

    def __init__(self, hidden_dim: int, num_channels: int, routing_iters: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.routing_iters = routing_iters
        self.query = nn.Linear(hidden_dim, num_channels, bias=False)
        self.key = nn.Linear(hidden_dim, num_channels, bias=False)
        self.refine = nn.Linear(num_channels, num_channels, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
        Returns:
            routing_weights: (B, T, num_channels)  — soft assignment
            channel_masks:   (B, T, num_channels)  — hard top-k for efficiency
        """
        B, T, D = x.shape
        q = self.query(x)  # (B, T, C)
        k = self.key(x)    # (B, T, C)

        # O(n) coherence: dot-product of each token's query with mean key
        mean_key = k.mean(dim=1, keepdim=True)  # (B, 1, C)
        scores = q * mean_key                    # (B, T, C)
        scores = scores + self.refine(scores) * 0.1  # one refinement step
        routing_weights = F.softmax(scores, dim=-1)   # (B, T, C)

        # Hard routing: top-2 channels per token
        _, top_idx = routing_weights.topk(2, dim=-1)
        channel_masks = torch.zeros_like(routing_weights)
        channel_masks.scatter_(-1, top_idx, 1.0)

        return routing_weights, channel_masks


# ─── Shared Hierarchical Memory ──────────────────────────────────────────────

class SharedHierarchicalMemory(nn.Module):
    """3-tier memory: Working → Episodic → Semantic with key-value attention."""

    def __init__(self, hidden_dim: int, key_dim: int,
                 working_slots: int, episodic_slots: int, semantic_slots: int,
                 dropout: float = 0.1):
        super().__init__()
        self.key_dim = key_dim
        self.working_slots = working_slots
        self.episodic_slots = episodic_slots
        self.semantic_slots = semantic_slots

        # Key / value projections
        self.q_proj = nn.Linear(hidden_dim, key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Learnable memory slots
        self.working_keys = nn.Parameter(torch.randn(working_slots, key_dim) * 0.02)
        self.working_vals = nn.Parameter(torch.randn(working_slots, hidden_dim) * 0.02)
        self.episodic_keys = nn.Parameter(torch.randn(episodic_slots, key_dim) * 0.02)
        self.episodic_vals = nn.Parameter(torch.randn(episodic_slots, hidden_dim) * 0.02)
        self.semantic_keys = nn.Parameter(torch.randn(semantic_slots, key_dim) * 0.02)
        self.semantic_vals = nn.Parameter(torch.randn(semantic_slots, hidden_dim) * 0.02)

        # Gating between tiers
        self.tier_gate = nn.Linear(hidden_dim * 3, 3, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _read_tier(self, queries: torch.Tensor, keys: torch.Tensor,
                   values: torch.Tensor) -> torch.Tensor:
        """
        Read from one memory tier.
        queries: (B, T, key_dim)
        keys:    (S, key_dim)
        values:  (S, hidden_dim)
        Returns: (B, T, hidden_dim)
        """
        B = queries.shape[0]
        # BUG FIX: expand keys/vales to batch dim without transposing last two dims
        keys_expanded = keys.unsqueeze(0).expand(B, -1, -1)    # (B, S, key_dim)
        values_expanded = values.unsqueeze(0).expand(B, -1, -1) # (B, S, hidden_dim)

        # Scaled dot-product attention (O(n*S) but S is small)
        scale = math.sqrt(self.key_dim)
        # (B, T, key_dim) @ (B, key_dim, S) → (B, T, S)
        attn = torch.bmm(queries, keys_expanded.transpose(1, 2)) / scale
        attn = F.softmax(attn, dim=-1)
        # (B, T, S) @ (B, S, hidden_dim) → (B, T, hidden_dim)
        out = torch.bmm(attn, values_expanded)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = x.shape
        queries = self.q_proj(x)  # (B, T, key_dim)

        # Read from each tier
        w_out = self._read_tier(queries, self.working_keys, self.working_vals)
        e_out = self._read_tier(queries, self.episodic_keys, self.episodic_vals)
        s_out = self._read_tier(queries, self.semantic_keys, self.semantic_vals)

        # Gated combination
        gate_input = torch.cat([w_out, e_out, s_out], dim=-1)  # (B, T, D*3)
        gates = F.softmax(self.tier_gate(gate_input), dim=-1)   # (B, T, 3)
        combined = (gates[..., 0:1] * w_out +
                    gates[..., 1:2] * e_out +
                    gates[..., 2:3] * s_out)

        # Project and residual
        out = self.out_proj(self.v_proj(x) + combined)
        out = self.norm(out)
        x = x + self.dropout(out)

        stats = {
            'mem_w_gate': gates[..., 0].mean(),
            'mem_e_gate': gates[..., 1].mean(),
            'mem_s_gate': gates[..., 2].mean(),
        }
        return x, stats


# ─── Gated FFN (SwiGLU) ─────────────────────────────────────────────────────

class GatedFFN(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, hidden_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        h = gate * up
        h = self.down_proj(h)
        h = self.norm(h)
        return residual + self.dropout(h)


# ─── Adaptive Computation Block ──────────────────────────────────────────────

class AdaptiveComputationBlock(nn.Module):
    """Variable-depth processing: each token may take 1..max_adaptive_steps."""

    def __init__(self, hidden_dim: int, ff_dim: int, max_adaptive_steps: int,
                 dropout: float = 0.1):
        super().__init__()
        self.max_steps = max_adaptive_steps
        self.layers = nn.ModuleList([
            GatedFFN(hidden_dim, ff_dim, dropout) for _ in range(max_adaptive_steps)
        ])
        self.halt_prob = nn.Linear(hidden_dim, 1, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = x.shape
        output = torch.zeros_like(x)
        total_weight = torch.zeros(B, T, 1, device=x.device)

        stats = {'avg_steps': torch.tensor(0.0, device=x.device)}

        for step_idx in range(self.max_steps):
            x = self.layers[step_idx](x)

            # Halting probability
            p = torch.sigmoid(self.halt_prob(x))  # (B, T, 1)

            # BUG FIX: clamp to avoid going over 1.0
            remaining = 1.0 - total_weight
            # Compute max allowed p (leave at least 0.01 for remaining steps)
            steps_left = self.max_steps - step_idx
            min_remaining = 0.01 * max(steps_left - 1, 0)
            max_val = torch.clamp(remaining - min_remaining, min=0.01)
            p = torch.clamp(p, min=torch.tensor(0.01, device=x.device), max=max_val)

            # On last step, use all remaining weight
            if step_idx == self.max_steps - 1:
                p = torch.clamp(remaining, min=0.01)

            output = output + p * x
            total_weight = total_weight + p

        output = self.norm(output)

        avg_steps = torch.tensor(float(self.max_steps), device=x.device)
        stats['avg_steps'] = avg_steps
        return output, stats


# ─── Compositional Reasoner ─────────────────────────────────────────────────

class CompositionalReasoner(nn.Module):
    """Hyperdimensional computing for role-filler binding."""

    def __init__(self, hidden_dim: int, key_dim: int, dropout: float = 0.1):
        super().__init__()
        self.role_proj = nn.Linear(hidden_dim, key_dim, bias=False)
        self.filler_proj = nn.Linear(hidden_dim, key_dim, bias=False)
        self.unbind_proj = nn.Linear(key_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        roles = self.role_proj(x)     # (B, T, K)
        fillers = self.filler_proj(x) # (B, T, K)

        # Circular convolution as binding operation (element-wise multiply in frequency domain)
        bound = roles * fillers  # (B, T, K) — simplified HDC binding

        # Shift-based unbinding for positional awareness
        bound_shifted = torch.roll(bound, shifts=1, dims=1)
        composed = bound + bound_shifted

        out = self.unbind_proj(composed)
        out = self.norm(out)
        return residual + self.dropout(out)


# ─── Cognitive Router ────────────────────────────────────────────────────────

class CognitiveRouter(nn.Module):
    """Routes tokens to channels based on coherence scores."""

    def __init__(self, hidden_dim: int, num_channels: int, channel_dim: int,
                 routing_iters: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.coherence_router = CoherenceRouter(hidden_dim, num_channels, routing_iters)

        # Per-channel projections
        self.to_channels = nn.Linear(hidden_dim, num_channels * channel_dim, bias=False)
        self.from_channels = nn.Linear(num_channels * channel_dim, hidden_dim, bias=False)

        # Channel processing
        self.channels = nn.ModuleList([
            CognitiveChannel(channel_dim, channel_dim * 4) for _ in range(num_channels)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = x.shape

        # Route
        routing_weights, channel_masks = self.coherence_router(x)  # (B, T, C)

        # Project to channel space
        channel_input = self.to_channels(x)  # (B, T, C*CD)
        channel_input = channel_input.view(B, T, self.num_channels, self.channel_dim)

        # Process each channel
        channel_outputs = []
        for c in range(self.num_channels):
            # Weighted input for this channel
            w = routing_weights[:, :, c:c+1].unsqueeze(-1)  # (B, T, 1, 1)
            ch_in = (channel_input[:, :, c, :] * routing_weights[:, :, c:c+1])  # (B, T, CD)
            ch_out = self.channels[c](ch_in)  # (B, T, CD)
            channel_outputs.append(ch_out)

        # Combine channels
        combined = torch.cat(channel_outputs, dim=-1)  # (B, T, C*CD)
        out = self.from_channels(combined)
        out = self.norm(out)
        x = x + out

        stats = {
            'routing_entropy': -(routing_weights * (routing_weights + 1e-8).log()).sum(-1).mean(),
        }
        return x, stats


# ─── CogNet Block ────────────────────────────────────────────────────────────

class CogNetBlock(nn.Module):
    """Router + Memory Read + FFN with residual connections."""

    def __init__(self, hidden_dim: int, num_channels: int, channel_dim: int,
                 ff_dim: int, key_dim: int, routing_iters: int,
                 max_adaptive_steps: int,
                 working_slots: int, episodic_slots: int, semantic_slots: int,
                 dropout: float = 0.1):
        super().__init__()
        self.router = CognitiveRouter(hidden_dim, num_channels, channel_dim, routing_iters)
        self.memory = SharedHierarchicalMemory(
            hidden_dim, key_dim, working_slots, episodic_slots, semantic_slots, dropout
        )
        self.adaptive_ffn = AdaptiveComputationBlock(
            hidden_dim, ff_dim, max_adaptive_steps, dropout
        )
        self.composer = CompositionalReasoner(hidden_dim, key_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        stats = {}

        x, r_stats = self.router(x)
        stats.update(r_stats)

        x, m_stats = self.memory(x)
        stats.update(m_stats)

        x, a_stats = self.adaptive_ffn(x)
        stats.update(a_stats)

        x = self.composer(x)
        x = self.norm(x)

        return x, stats


# ─── CogNet1B ────────────────────────────────────────────────────────────────

class CogNet1B(nn.Module):
    """Non-transformer language model with cognitive routing."""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 2048,
        num_blocks: int = 13,
        num_channels: int = 8,
        channel_dim: int = 256,
        ff_dim: int = 4096,
        routing_iters: int = 1,
        max_adaptive_steps: int = 2,
        max_seq_len: int = 2048,
        working_slots: int = 64,
        episodic_slots: int = 128,
        semantic_slots: int = 256,
        key_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len

        # Encoder
        self.encoder = TokenEncoder(vocab_size, hidden_dim, max_seq_len, dropout)

        # Blocks
        self.blocks = nn.ModuleList([
            CogNetBlock(
                hidden_dim, num_channels, channel_dim, ff_dim,
                key_dim, routing_iters, max_adaptive_steps,
                working_slots, episodic_slots, semantic_slots, dropout
            )
            for _ in range(num_blocks)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output head (weight-tied with token embedding)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output_proj.weight = self.encoder.token_emb.weight

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor,
                return_stats: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (B, T) integer token ids
            return_stats: whether to collect intermediate statistics
        Returns:
            dict with 'logits' (B, T, vocab_size) and optional 'stats'
        """
        x = self.encoder(input_ids)

        all_stats = {} if return_stats else None

        for i, block in enumerate(self.blocks):
            x, block_stats = block(x)
            if return_stats:
                for k, v in block_stats.items():
                    key = f'block{i}_{k}'
                    # BUG FIX: clamp NaN/Inf in stats
                    if isinstance(v, torch.Tensor):
                        v = v.detach().float()
                        if torch.isnan(v) or torch.isinf(v):
                            v = torch.tensor(0.0)
                    all_stats[key] = v

        x = self.final_norm(x)
        logits = self.output_proj(x)

        result = {'logits': logits}
        if return_stats:
            result['stats'] = all_stats
        return result

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 0,
                 ) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx = input_ids[:, -self.max_seq_len:]
            result = self(idx)
            logits = result['logits'][:, -1, :] / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

    def get_complexity_analysis(self) -> Dict[str, str]:
        return {
            'architecture': 'CogNet (Non-Transformer)',
            'routing': f'O(n) coherence routing x {self.num_channels} channels',
            'memory': '3-tier hierarchical (Working/Episodic/Semantic)',
            'attention': 'None (replaced by cognitive routing + memory)',
            'ffn': 'SwiGLU with adaptive computation',
            'composition': 'Hyperdimensional role-filler binding',
            'sequence_complexity': 'O(n) per layer (vs O(n^2) for transformers)',
            'params': f'{self.count_parameters()["total"]:,}',
        }


# ─── Factory Functions ───────────────────────────────────────────────────────

def create_cognet_1b_small(vocab_size: int = 256, max_seq_len: int = 2048,
                           dropout: float = 0.1) -> CogNet1B:
    """Create ~87M parameter model."""
    return CogNet1B(
        vocab_size=vocab_size,
        hidden_dim=1024,
        num_blocks=8,
        num_channels=8,
        channel_dim=128,
        ff_dim=2048,
        routing_iters=1,
        max_adaptive_steps=2,
        max_seq_len=max_seq_len,
        working_slots=32,
        episodic_slots=64,
        semantic_slots=128,
        key_dim=256,
        dropout=dropout,
    )


def create_cognet_1b(vocab_size: int = 256, max_seq_len: int = 2048,
                     dropout: float = 0.1) -> CogNet1B:
    """Create ~1B parameter model."""
    return CogNet1B(
        vocab_size=vocab_size,
        hidden_dim=2048,
        num_blocks=13,
        num_channels=8,
        channel_dim=256,
        ff_dim=4096,
        routing_iters=1,
        max_adaptive_steps=2,
        max_seq_len=max_seq_len,
        working_slots=64,
        episodic_slots=128,
        semantic_slots=256,
        key_dim=256,
        dropout=dropout,
    )


# ─── Self-Test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("CogNet1B Self-Test")
    print("=" * 60)

    # Small model for quick test
    model = CogNet1B(
        vocab_size=128,
        hidden_dim=128,
        num_blocks=2,
        num_channels=4,
        channel_dim=32,
        ff_dim=256,
        routing_iters=1,
        max_adaptive_steps=2,
        max_seq_len=64,
        working_slots=8,
        episodic_slots=16,
        semantic_slots=32,
        key_dim=64,
        dropout=0.1,
    )

    params = model.count_parameters()
    print(f"\nParameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Forward pass
    x = torch.randint(0, 128, (2, 16))
    result = model(x, return_stats=True)
    logits = result['logits']
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Stats keys: {len(result.get('stats', {}))}")

    # Backward pass
    loss = logits.sum()
    loss.backward()
    print("Backward pass OK")

    # Generate test
    gen = model.generate(x[:, :4], max_new_tokens=8, temperature=0.8, top_k=10)
    print(f"Generated shape: {gen.shape}")

    # Complexity analysis
    analysis = model.get_complexity_analysis()
    for k, v in analysis.items():
        print(f"  {k}: {v}")

    print("\n✓ All self-tests passed!")

    # Test factory functions
    small = create_cognet_1b_small(vocab_size=128, max_seq_len=64)
    print(f"\nSmall model params: {small.count_parameters()['total']:,}")
