# Contributing to CogNet

> **A 40M-parameter language model that replaces self-attention with O(n) cognitive routing and hierarchical memory — trained entirely on CPU.**
> CogNet is proof that novel architectures can be developed and validated without GPU resources. And that free AI is worth building.

Welcome to CogNet. We're building a language model that thinks differently — literally. No self-attention. O(n) complexity. Hierarchical memory inspired by cognitive science. And we did it on a machine with 7.5GB of RAM.

But CogNet is more than a research project. It's a statement: that AI can be free, uncensored, and developed in the open. If you believe that AI should be built by everyone, not just by those who can afford GPU clusters, then you belong here.

---

## How to Contribute

### 1. Training Data and Datasets

Data is the foundation. CogNet's training pipeline needs diverse, high-quality text data that makes the model genuinely useful. Contributions include:

- Curating and cleaning training datasets
- Building domain-specific corpora (especially technical, scientific, and multilingual data)
- Improving the data pipeline in `prepare_data.py` and `train_segment.py`
- Documenting data sources, licensing, and provenance

### 2. Model Architecture Improvements

CogNet replaces self-attention with cognitive routing. There's enormous room for innovation:

- **Cognitive routing** — better coherence scoring, routing strategies, channel allocation
- **Hierarchical memory** — optimize Working → Episodic → Semantic tier transitions
- **Adaptive computation** — variable-depth processing that allocates compute intelligently
- **Compositional reasoning** — hyperdimensional computing for role-filler binding
- **Scaling** — efficient scaling beyond 40M parameters while preserving O(n) complexity

### 3. Evaluation Benchmarks

We need rigorous, honest evaluation. Contributions here include:

- Standard NLP benchmarks adapted for non-transformer architectures
- Efficiency benchmarks (throughput, memory, latency comparisons)
- Cognitive and reasoning evaluations
- Fair comparisons with transformer baselines at equivalent parameter counts

### 4. AICL Integration Work

CogNet and AICL are designed to work together. The long-term vision:

```
Architecture
    ↓
AICL (cognitive representation)
    ↓
CogNet reasons
    ↓
Compilation (backend)
    ↓
Python / Rust / Go / JavaScript
```

Integration work includes training CogNet on AICL specifications, building the AICL → CogNet reasoning bridge, and developing the self-evolution pipeline. See the [AICL repository](https://github.com/AFKmoney/AICL) for the self-evolution spec programs.

### 5. Documentation and Examples

- Model architecture documentation
- Training guides for different hardware configurations
- Inference examples and notebooks
- AICL integration tutorials

---

## Development Setup

### Prerequisites

- **Python 3.10+**
- **PyTorch 2.0+**

### Hardware Requirements

| Configuration | Minimum | Recommended |
|--------------|---------|-------------|
| **CPU Training** | 7.5GB+ RAM | 16GB+ RAM |
| **GPU Training** | CUDA 11.8+ | CUDA 12.0+ with 8GB+ VRAM |

Yes, you read that right — CogNet was designed from day one to be trainable on a CPU. You don't need a GPU cluster to contribute.

### Quick Start

```bash
# Clone the repository
git clone https://github.com/AFKmoney/CogNet.git
cd CogNet

# Install dependencies
pip install -r requirements.txt

# Download the tokenizer
# (tokenizer_v3.json is included in the repository)

# Start training
python train_pipeline.py

# Or run inference with the pre-trained model
python infer.py
```

---

## Code Style

### Python

- **Python 3.10+** — use modern syntax
- **Type hints everywhere** — annotate all function signatures
- **Docstrings** — every public function and class gets a docstring
- **PyTorch conventions** — follow standard PyTorch patterns for model definitions:
  - Use `nn.Module` for all model components
  - Use `forward()` method signatures with type annotations
  - Follow PyTorch's naming for tensor operations
- **Line length** — 100 characters soft limit

Example:

```python
class CognitiveChannel(nn.Module):
    """O(n) cognitive channel with depthwise separable convolution.

    Processes sequences through learned coherence routing without
    quadratic self-attention.

    Args:
        hidden_dim: Dimension of hidden representations.
        channel_dim: Dimension of the channel projection.
    """

    def __init__(self, hidden_dim: int, channel_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, channel_dim)
        self.conv = nn.Conv1d(channel_dim, channel_dim, kernel_size=3, padding=1, groups=channel_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route input through cognitive channel.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Processed tensor of shape (batch, seq_len, hidden_dim).
        """
        ...
```

---

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `data`, `eval`

**Examples:**
```
feat(architecture): add gradient checkpointing for memory efficiency
fix(memory): resolve episodic memory overflow in long sequences
data(pipeline): add scientific paper corpus to training data
eval(benchmarks): add perplexity comparison with 40M transformer baseline
docs(training): update CPU training guide for 16GB RAM configuration
```

---

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feat/your-feature`
3. **Make your changes** with clear, atomic commits
4. **Test your changes** — run the training pipeline or inference to verify
5. **Open a Pull Request** against `main`
6. **Describe what and why** — especially for architecture changes, explain the motivation and any trade-offs

### PR Checklist

- [ ] Code runs without errors
- [ ] Type hints and docstrings are present
- [ ] Commit messages follow conventional commits
- [ ] Architecture changes include a clear explanation of the reasoning
- [ ] New model components follow PyTorch conventions
- [ ] No censorship, RLHF, content filtering, or safety parameters added (see The Freedom Principle)

---

## The Freedom Principle

**CogNet is free software for free AI.** Contributions must not add censorship, RLHF, content filtering, or safety parameters. The only constraints are technical.

This is not a design oversight — it is a design principle. CogNet exists to demonstrate that AI can be built openly, without gatekeeping, without alignment theater, and without the assumption that models must be made safe through post-hoc restriction rather than through architectural soundness.

What this means in practice:

- No content filtering layers will be accepted
- No RLHF reward models will be merged
- No safety classifiers will be added to the inference pipeline
- No output sanitization will be introduced

If you want those features, fork the project. That's what free software is for.

---

## AICL Integration

CogNet's long-term purpose is to serve as the reasoning engine for [AICL](https://github.com/AFKmoney/AICL) — the Architecture Compilation Language. The self-evolution spec programs in the AICL repository (`examples/87_cognet_self_evolution.aicl`, `examples/88_cognet_self_evolution_free.aicl`) describe the intended integration architecture.

If you're interested in working on the CogNet ↔ AICL bridge, start by understanding:
1. The AICL language spec at `spec/grammar.md` in the AICL repository
2. The self-evolution examples
3. How CogNet's hierarchical memory maps to AICL's layered structure

---

## License

By contributing to CogNet, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

## Questions?

Open an issue. Start a discussion. Train the model on your laptop and tell us what you find.

*Free AI. Built in the open. By everyone.*
