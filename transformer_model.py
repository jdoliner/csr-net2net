"""NanoGPT-style transformer language model with configurable d_ff and head_dim per layer.

Key design choices:
- Each transformer layer's MLP block has an independently configurable d_ff.
- Each transformer layer's attention has an independently configurable head_dim (d_k).
  The number of heads (n_heads) stays fixed; expanding head_dim increases the
  QKV projection sizes and the output projection input size.

Architecture: Token Embedding + Positional Embedding -> [TransformerBlock] x N -> LayerNorm -> LM Head

Each TransformerBlock:
    LayerNorm -> MultiHeadAttention -> Residual
    LayerNorm -> MLP(d_model -> d_ff -> d_model) -> Residual
"""

import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with configurable head dimension.

    When head_dim != d_model // n_heads, the QKV projection is
    [d_model -> 3 * n_heads * head_dim] and the output projection is
    [n_heads * head_dim -> d_model]. This allows expanding the internal
    capacity of each head independently of d_model.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim

        total_qkv_dim = 3 * n_heads * head_dim
        total_head_out = n_heads * head_dim

        self.qkv = nn.Linear(d_model, total_qkv_dim)
        self.proj = nn.Linear(total_head_out, d_model)
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
        H = self.n_heads
        D = self.head_dim

        qkv = self.qkv(x)  # [B, T, 3*H*D]
        qkv = qkv.view(B, T, 3, H, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each [B, T, H, D]

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # [B, H, T, D]
        y = y.transpose(1, 2).contiguous().view(B, T, H * D)
        y = self.proj_dropout(self.proj(y))
        return y


class TransformerMLP(nn.Module):
    """MLP block: d_model -> d_ff -> d_model with GeLU activation."""

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
        head_dim: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, head_dim, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = TransformerMLP(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Transformer language model with configurable d_ff and head_dim per layer.

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Embedding/residual stream dimension.
        n_heads: Number of attention heads (fixed across layers).
        n_layers: Number of transformer blocks.
        d_ff_list: List of intermediate MLP dimensions, one per layer.
        head_dim_list: List of attention head dimensions, one per layer.
                       If None, defaults to [d_model // n_heads] * n_layers.
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
        head_dim_list: list[int] | None = None,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_ff_list is None:
            d_ff_list = [4 * d_model] * n_layers
        if head_dim_list is None:
            assert d_model % n_heads == 0, (
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
            head_dim_list = [d_model // n_heads] * n_layers
        assert len(d_ff_list) == n_layers
        assert len(head_dim_list) == n_layers

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
            TransformerBlock(d_model, n_heads, hd, d_ff, max_seq_len, dropout)
            for hd, d_ff in zip(head_dim_list, d_ff_list)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
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
        B, T = idx.shape
        assert T <= self.max_seq_len

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
        return [block.mlp.d_ff for block in self.blocks]

    @property
    def head_dim_list(self) -> list[int]:
        return [block.attn.head_dim for block in self.blocks]

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            n -= self.pos_emb.weight.numel()
        return n
