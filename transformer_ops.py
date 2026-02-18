"""Expansion operations for transformer MLP blocks.

We expand the intermediate dimension (d_ff) of transformer MLP blocks:
    MLP: d_model -> d_ff -> d_model
    W_up:   [d_ff, d_model]   (incoming weights)
    W_down: [d_model, d_ff]   (outgoing weights)

This maps directly to the CSR/Net2Net framework:
    - W_up rows = neurons' incoming weights (what input features they respond to)
    - W_down columns = neurons' outgoing weights (how they influence the residual stream)
    - Sort neurons via spectral seriation on these combined feature vectors
    - Resample to double d_ff via linear interpolation (CSR) or duplication (Net2Net)
    - Scale W_down by (old_d_ff / new_d_ff) for energy preservation
"""

import torch
import torch.nn as nn

from ops import (
    cosine_similarity_matrix,
    resample_1d,
    resample_2d_cols,
    resample_2d_rows,
    spectral_sort,
)
from transformer_model import TransformerLM


def transformer_layer_similarity_scores(model: TransformerLM) -> list[float]:
    """Compute post-seriation adjacent cosine similarity for each layer's MLP block.

    For each MLP block:
    1. Construct feature vectors by concatenating W_up rows and W_down columns
    2. Sort via spectral seriation
    3. Compute mean adjacent cosine similarity

    Lower scores = more "stretched thin" = would benefit from expansion.

    Args:
        model: The transformer language model.

    Returns:
        List of similarity scores, one per transformer layer.
    """
    scores = []

    for block in model.blocks:
        mlp = block.mlp
        incoming = mlp.up.weight.data     # [d_ff, d_model]
        outgoing = mlp.down.weight.data   # [d_model, d_ff]

        perm = spectral_sort(incoming, outgoing)

        features = torch.cat([incoming, outgoing.T], dim=1)
        sorted_features = features[perm]

        if sorted_features.shape[0] < 2:
            scores.append(1.0)
            continue

        norms = sorted_features.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = sorted_features / norms
        adj_sims = (normalized[:-1] * normalized[1:]).sum(dim=1)
        scores.append(adj_sims.mean().item())

    return scores


def params_after_doubling_mlp(model: TransformerLM, layer_idx: int) -> int:
    """Compute total parameter count if layer_idx's MLP d_ff were doubled.

    Only the MLP's W_up, b_up, W_down weights change:
    - W_up: [d_ff, d_model] -> [2*d_ff, d_model]  (+d_ff * d_model)
    - b_up: [d_ff] -> [2*d_ff]                      (+d_ff)
    - W_down: [d_model, d_ff] -> [d_model, 2*d_ff]  (+d_model * d_ff)

    Args:
        model: The transformer language model.
        layer_idx: Which transformer layer's MLP to consider doubling.

    Returns:
        Total parameter count after doubling.
    """
    current_total = sum(p.numel() for p in model.parameters())
    d_ff = model.blocks[layer_idx].mlp.d_ff
    d_model = model.d_model

    # Additional params from doubling d_ff
    added = d_ff * d_model + d_ff + d_model * d_ff
    return current_total + added


def _permute_mlp_neurons(
    mlp: nn.Module,
    perm: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Permute neurons in an MLP block's intermediate dimension."""
    with torch.no_grad():
        mlp.up.weight.data = mlp.up.weight.data[perm]
        mlp.up.bias.data = mlp.up.bias.data[perm]
        mlp.down.weight.data = mlp.down.weight.data[:, perm]

    if optimizer is not None:
        from ops import _permute_optimizer_state
        _permute_optimizer_state(optimizer, mlp.up.weight, perm, dim=0)
        _permute_optimizer_state(optimizer, mlp.up.bias, perm, dim=0)
        _permute_optimizer_state(optimizer, mlp.down.weight, perm, dim=1)


def expand_transformer_mlp_continuous(
    model: TransformerLM,
    layer_idx: int,
    optimizer: torch.optim.Optimizer,
) -> TransformerLM:
    """Expand a single transformer layer's MLP d_ff via CSR (double it).

    1. Sort MLP neurons via spectral seriation
    2. Resample W_up, b_up, W_down via linear interpolation
    3. Scale W_down by (old_d_ff / new_d_ff) for energy preservation
    4. Resample optimizer state

    Args:
        model: The transformer language model.
        layer_idx: Which layer's MLP to expand.
        optimizer: The optimizer (state will be resampled).

    Returns:
        New TransformerLM with the target layer's MLP doubled.
    """
    device = next(model.parameters()).device
    mlp = model.blocks[layer_idx].mlp

    # Sort
    incoming = mlp.up.weight.data
    outgoing = mlp.down.weight.data
    perm = spectral_sort(incoming, outgoing)
    _permute_mlp_neurons(mlp, perm, optimizer)

    # Build new model with doubled d_ff for this layer
    old_d_ff = mlp.d_ff
    new_d_ff = old_d_ff * 2
    new_d_ff_list = list(model.d_ff_list)
    new_d_ff_list[layer_idx] = new_d_ff

    new_model = TransformerLM(
        vocab_size=model.vocab_size,
        d_model=model.d_model,
        n_heads=model.n_heads,
        n_layers=model.n_layers,
        d_ff_list=new_d_ff_list,
        max_seq_len=model.max_seq_len,
        dropout=model.dropout_rate,
    ).to(device)

    # Copy all weights, resampling only the target MLP
    with torch.no_grad():
        # Copy everything except the target block's MLP
        new_model.tok_emb.weight.data = model.tok_emb.weight.data.clone()
        new_model.pos_emb.weight.data = model.pos_emb.weight.data.clone()
        new_model.ln_f.weight.data = model.ln_f.weight.data.clone()
        new_model.ln_f.bias.data = model.ln_f.bias.data.clone()
        # lm_head shares weight with tok_emb (weight tying), no separate copy needed

        for i, (old_block, new_block) in enumerate(zip(model.blocks, new_model.blocks)):
            # Copy attention (unchanged)
            new_block.ln1.weight.data = old_block.ln1.weight.data.clone()
            new_block.ln1.bias.data = old_block.ln1.bias.data.clone()
            new_block.attn.qkv.weight.data = old_block.attn.qkv.weight.data.clone()
            new_block.attn.qkv.bias.data = old_block.attn.qkv.bias.data.clone()
            new_block.attn.proj.weight.data = old_block.attn.proj.weight.data.clone()
            new_block.attn.proj.bias.data = old_block.attn.proj.bias.data.clone()
            new_block.ln2.weight.data = old_block.ln2.weight.data.clone()
            new_block.ln2.bias.data = old_block.ln2.bias.data.clone()

            if i == layer_idx:
                # Resample MLP weights
                old_mlp = old_block.mlp
                new_mlp = new_block.mlp

                # W_up: [d_ff, d_model] -> [2*d_ff, d_model] (resample rows)
                new_mlp.up.weight.data = resample_2d_rows(
                    old_mlp.up.weight.data, new_d_ff
                )
                # b_up: [d_ff] -> [2*d_ff]
                new_mlp.up.bias.data = resample_1d(old_mlp.up.bias.data, new_d_ff)

                # W_down: [d_model, d_ff] -> [d_model, 2*d_ff] (resample cols)
                new_mlp.down.weight.data = resample_2d_cols(
                    old_mlp.down.weight.data, new_d_ff
                )
                # Energy preservation
                new_mlp.down.weight.data *= (old_d_ff / new_d_ff)

                # W_down bias is [d_model], unchanged
                new_mlp.down.bias.data = old_mlp.down.bias.data.clone()
            else:
                # Copy MLP unchanged
                new_block.mlp.up.weight.data = old_block.mlp.up.weight.data.clone()
                new_block.mlp.up.bias.data = old_block.mlp.up.bias.data.clone()
                new_block.mlp.down.weight.data = old_block.mlp.down.weight.data.clone()
                new_block.mlp.down.bias.data = old_block.mlp.down.bias.data.clone()

    # Resample optimizer state
    _resample_transformer_optimizer_state(optimizer, model, new_model)
    return new_model


def expand_transformer_mlp_net2net(
    model: TransformerLM,
    layer_idx: int,
    optimizer: torch.optim.Optimizer,
    noise_std: float = 1e-3,
) -> TransformerLM:
    """Expand a single transformer layer's MLP d_ff via Net2Net (duplicate neurons).

    Each intermediate neuron is duplicated once. W_down columns scaled by 0.5.
    Small noise added to W_up to break symmetry.

    Args:
        model: The transformer language model.
        layer_idx: Which layer's MLP to expand.
        optimizer: The optimizer (state will be duplicated).
        noise_std: Standard deviation of symmetry-breaking noise.

    Returns:
        New TransformerLM with the target layer's MLP doubled.
    """
    device = next(model.parameters()).device
    mlp = model.blocks[layer_idx].mlp

    old_d_ff = mlp.d_ff
    new_d_ff = old_d_ff * 2
    new_d_ff_list = list(model.d_ff_list)
    new_d_ff_list[layer_idx] = new_d_ff

    new_model = TransformerLM(
        vocab_size=model.vocab_size,
        d_model=model.d_model,
        n_heads=model.n_heads,
        n_layers=model.n_layers,
        d_ff_list=new_d_ff_list,
        max_seq_len=model.max_seq_len,
        dropout=model.dropout_rate,
    ).to(device)

    with torch.no_grad():
        new_model.tok_emb.weight.data = model.tok_emb.weight.data.clone()
        new_model.pos_emb.weight.data = model.pos_emb.weight.data.clone()
        new_model.ln_f.weight.data = model.ln_f.weight.data.clone()
        new_model.ln_f.bias.data = model.ln_f.bias.data.clone()

        for i, (old_block, new_block) in enumerate(zip(model.blocks, new_model.blocks)):
            new_block.ln1.weight.data = old_block.ln1.weight.data.clone()
            new_block.ln1.bias.data = old_block.ln1.bias.data.clone()
            new_block.attn.qkv.weight.data = old_block.attn.qkv.weight.data.clone()
            new_block.attn.qkv.bias.data = old_block.attn.qkv.bias.data.clone()
            new_block.attn.proj.weight.data = old_block.attn.proj.weight.data.clone()
            new_block.attn.proj.bias.data = old_block.attn.proj.bias.data.clone()
            new_block.ln2.weight.data = old_block.ln2.weight.data.clone()
            new_block.ln2.bias.data = old_block.ln2.bias.data.clone()

            if i == layer_idx:
                old_mlp = old_block.mlp
                new_mlp = new_block.mlp

                # W_up: duplicate rows, add noise to copies
                w_up = old_mlp.up.weight.data
                w_up_dup = w_up.clone() + torch.randn_like(w_up) * noise_std
                new_mlp.up.weight.data = torch.cat([w_up, w_up_dup], dim=0)

                b_up = old_mlp.up.bias.data
                b_up_dup = b_up.clone() + torch.randn_like(b_up) * noise_std
                new_mlp.up.bias.data = torch.cat([b_up, b_up_dup], dim=0)

                # W_down: duplicate columns, scale by 0.5
                w_down = old_mlp.down.weight.data * 0.5
                new_mlp.down.weight.data = torch.cat([w_down, w_down], dim=1)

                new_mlp.down.bias.data = old_mlp.down.bias.data.clone()
            else:
                new_block.mlp.up.weight.data = old_block.mlp.up.weight.data.clone()
                new_block.mlp.up.bias.data = old_block.mlp.up.bias.data.clone()
                new_block.mlp.down.weight.data = old_block.mlp.down.weight.data.clone()
                new_block.mlp.down.bias.data = old_block.mlp.down.bias.data.clone()

    _duplicate_transformer_optimizer_state(optimizer, model, new_model)
    return new_model


# ---------------------------------------------------------------------------
# Optimizer state helpers
# ---------------------------------------------------------------------------


def _resample_transformer_optimizer_state(
    optimizer: torch.optim.Optimizer,
    old_model: TransformerLM,
    new_model: TransformerLM,
) -> None:
    """Resample optimizer state for CSR expansion of a transformer."""
    old_params = list(old_model.parameters())
    new_params = list(new_model.parameters())

    new_state = {}
    for old_p, new_p in zip(old_params, new_params):
        if old_p not in optimizer.state:
            continue
        old_s = optimizer.state[old_p]
        new_s = {}

        for key, val in old_s.items():
            if not isinstance(val, torch.Tensor):
                new_s[key] = val
            elif val.dim() == 0:
                new_s[key] = val.clone()
            elif key in ("exp_avg", "exp_avg_sq"):
                buf = val
                if buf.shape == new_p.shape:
                    new_s[key] = buf.clone()
                elif buf.dim() == 1:
                    new_s[key] = resample_1d(buf, new_p.shape[0])
                elif buf.dim() == 2:
                    result = buf
                    if result.shape[0] != new_p.shape[0]:
                        result = resample_2d_rows(result, new_p.shape[0])
                    if result.shape[1] != new_p.shape[1]:
                        result = resample_2d_cols(result, new_p.shape[1])
                    new_s[key] = result
            else:
                new_s[key] = val.clone()

        new_state[new_p] = new_s

    optimizer.param_groups[0]["params"] = list(new_model.parameters())
    optimizer.state.clear()
    optimizer.state.update(new_state)


def _duplicate_transformer_optimizer_state(
    optimizer: torch.optim.Optimizer,
    old_model: TransformerLM,
    new_model: TransformerLM,
) -> None:
    """Duplicate optimizer state for Net2Net expansion of a transformer."""
    old_params = list(old_model.parameters())
    new_params = list(new_model.parameters())

    new_state = {}
    for old_p, new_p in zip(old_params, new_params):
        if old_p not in optimizer.state:
            continue
        old_s = optimizer.state[old_p]
        new_s = {}

        for key, val in old_s.items():
            if not isinstance(val, torch.Tensor):
                new_s[key] = val
            elif val.dim() == 0:
                new_s[key] = val.clone()
            elif key in ("exp_avg", "exp_avg_sq"):
                buf = val
                if buf.shape == new_p.shape:
                    new_s[key] = buf.clone()
                elif buf.dim() == 1:
                    new_s[key] = torch.cat([buf, buf], dim=0)
                elif buf.dim() == 2:
                    result = buf
                    if result.shape[0] != new_p.shape[0]:
                        result = torch.cat([result, result], dim=0)
                    if result.shape[1] != new_p.shape[1]:
                        result = torch.cat([result, result], dim=1)
                    new_s[key] = result
            else:
                new_s[key] = val.clone()

        new_state[new_p] = new_s

    optimizer.param_groups[0]["params"] = list(new_model.parameters())
    optimizer.state.clear()
    optimizer.state.update(new_state)
