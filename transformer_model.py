"""NanoGPT-style transformer language model with configurable d_ff per layer.

The key design choice: each transformer layer's MLP block has an independently
configurable intermediate dimension (d_ff). This allows targeted expansion of
individual layers' MLP blocks while keeping d_model, attention heads, and
number of layers fixed.

Architecture: Token Embedding + Positional Embedding -> [TransformerBlock] x N -> LayerNorm -> LM Head

Each TransformerBlock:
    LayerNorm -> MultiHeadAttention -> Residual
    LayerNorm -> MLP(d_model -> d_ff -> d_model) -> Residual
"""

import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_dropout(self.proj(y))
        return y


class TransformerMLP(nn.Module):
    """MLP block: d_model -> d_ff -> d_model with GeLU activation.

    This is the block we expand via CSR. The intermediate dimension d_ff
    is independently configurable per layer.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(self.act(self.up(x))))

    @property
    def d_ff(self) -> int:
        return self.up.out_features


class TransformerBlock(nn.Module):
    """Single transformer block: attention + MLP with pre-norm and residuals."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = TransformerMLP(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Transformer language model with configurable d_ff per layer.

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Embedding/residual stream dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        d_ff_list: List of intermediate MLP dimensions, one per layer.
                   If None, defaults to [4 * d_model] * n_layers.
        max_seq_len: Maximum sequence length.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff_list: list[int] | None = None,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_ff_list is None:
            d_ff_list = [4 * d_model] * n_layers
        assert len(d_ff_list) == n_layers, (
            f"d_ff_list length ({len(d_ff_list)}) must match n_layers ({n_layers})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for d_ff in d_ff_list
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding and output weights
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            idx: [B, T] token indices.

        Returns:
            [B, T, vocab_size] logits.
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.emb_dropout(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @property
    def d_ff_list(self) -> list[int]:
        """Return the d_ff for each layer's MLP block."""
        return [block.mlp.d_ff for block in self.blocks]

    def num_params(self, non_embedding: bool = True) -> int:
        """Count parameters, optionally excluding embeddings."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            n -= self.pos_emb.weight.numel()
        return n
