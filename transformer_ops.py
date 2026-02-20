"""Expansion operations for transformer MLP blocks and attention heads.

MLP expansion (d_ff):
    W_up:   [d_ff, d_model]   (incoming)
    W_down: [d_model, d_ff]   (outgoing)
    Sort neurons, resample along d_ff axis, scale W_down.

Attention head expansion (head_dim / d_k):
    Per head h, the "neurons" are the d_k internal dimensions. Each is a
    quadruplet: column i of W_Q^h, column i of W_K^h, column i of W_V^h,
    and row i of W_O^h.

    The fused QKV weight [3*H*d_k, d_model] is reshaped to [3, H, d_k, d_model]
    for per-head processing. The proj weight [d_model, H*d_k] is reshaped to
    [d_model, H, d_k].

    Scaling corrections after doubling d_k:
    - W_O (proj): scale by 0.5 (energy preservation, same as MLP)
    - W_Q, W_K: scale by 2^{-0.25} (QK dot product correction for softmax)
    - W_V: no extra scaling (handled by W_O scaling)
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


# ---------------------------------------------------------------------------
# Helper: build a new model and copy all weights
# ---------------------------------------------------------------------------


def _new_model_from(
    model: TransformerLM,
    d_ff_list: list[int] | None = None,
    head_dim_list: list[int] | None = None,
) -> TransformerLM:
    """Create a new TransformerLM with optionally modified d_ff_list/head_dim_list."""
    device = next(model.parameters()).device
    return TransformerLM(
        vocab_size=model.vocab_size,
        d_model=model.d_model,
        n_heads=model.n_heads,
        n_layers=model.n_layers,
        d_ff_list=d_ff_list or list(model.d_ff_list),
        head_dim_list=head_dim_list or list(model.head_dim_list),
        max_seq_len=model.max_seq_len,
        dropout=model.dropout_rate,
    ).to(device)


def _copy_shared_weights(old_model: TransformerLM, new_model: TransformerLM) -> None:
    """Copy embeddings and final layer norm (always unchanged during expansion)."""
    with torch.no_grad():
        new_model.tok_emb.weight.data = old_model.tok_emb.weight.data.clone()
        new_model.pos_emb.weight.data = old_model.pos_emb.weight.data.clone()
        new_model.ln_f.weight.data = old_model.ln_f.weight.data.clone()
        new_model.ln_f.bias.data = old_model.ln_f.bias.data.clone()


def _copy_block_attn(old_block, new_block) -> None:
    """Copy attention weights unchanged between blocks."""
    with torch.no_grad():
        new_block.attn.qkv.weight.data = old_block.attn.qkv.weight.data.clone()
        new_block.attn.qkv.bias.data = old_block.attn.qkv.bias.data.clone()
        new_block.attn.proj.weight.data = old_block.attn.proj.weight.data.clone()
        new_block.attn.proj.bias.data = old_block.attn.proj.bias.data.clone()


def _copy_block_mlp(old_block, new_block) -> None:
    """Copy MLP weights unchanged between blocks."""
    with torch.no_grad():
        new_block.mlp.up.weight.data = old_block.mlp.up.weight.data.clone()
        new_block.mlp.up.bias.data = old_block.mlp.up.bias.data.clone()
        new_block.mlp.down.weight.data = old_block.mlp.down.weight.data.clone()
        new_block.mlp.down.bias.data = old_block.mlp.down.bias.data.clone()


def _copy_block_norms(old_block, new_block) -> None:
    """Copy layer norm weights between blocks."""
    with torch.no_grad():
        new_block.ln1.weight.data = old_block.ln1.weight.data.clone()
        new_block.ln1.bias.data = old_block.ln1.bias.data.clone()
        new_block.ln2.weight.data = old_block.ln2.weight.data.clone()
        new_block.ln2.bias.data = old_block.ln2.bias.data.clone()


# ---------------------------------------------------------------------------
# MLP expansion (unchanged logic, refactored to use helpers)
# ---------------------------------------------------------------------------


def transformer_layer_similarity_scores(model: TransformerLM) -> list[float]:
    """Compute post-seriation adjacent cosine similarity for each layer's MLP block."""
    scores = []
    for block in model.blocks:
        mlp = block.mlp
        incoming = mlp.up.weight.data
        outgoing = mlp.down.weight.data
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
    """Compute total parameter count if layer_idx's MLP d_ff were doubled."""
    current_total = sum(p.numel() for p in model.parameters())
    d_ff = model.blocks[layer_idx].mlp.d_ff
    d_model = model.d_model
    added = d_ff * d_model + d_ff + d_model * d_ff
    return current_total + added


def params_after_doubling_attn(model: TransformerLM, layer_idx: int) -> int:
    """Compute total parameter count if layer_idx's attention head_dim were doubled."""
    current_total = sum(p.numel() for p in model.parameters())
    head_dim = model.blocks[layer_idx].attn.head_dim
    n_heads = model.n_heads
    d_model = model.d_model

    # QKV: [3*H*d_k, d_model] -> [3*H*2*d_k, d_model], bias [3*H*d_k] -> [3*H*2*d_k]
    added_qkv = 3 * n_heads * head_dim * d_model + 3 * n_heads * head_dim
    # Proj: [d_model, H*d_k] -> [d_model, H*2*d_k], bias unchanged [d_model]
    added_proj = d_model * n_heads * head_dim
    return current_total + added_qkv + added_proj


def _permute_mlp_neurons(mlp, perm, optimizer=None):
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
    model: TransformerLM, layer_idx: int, optimizer: torch.optim.Optimizer,
) -> TransformerLM:
    """Expand a single transformer layer's MLP d_ff via CSR (double it)."""
    mlp = model.blocks[layer_idx].mlp

    # Sort
    perm = spectral_sort(mlp.up.weight.data, mlp.down.weight.data)
    _permute_mlp_neurons(mlp, perm, optimizer)

    old_d_ff = mlp.d_ff
    new_d_ff = old_d_ff * 2
    new_d_ff_list = list(model.d_ff_list)
    new_d_ff_list[layer_idx] = new_d_ff

    new_model = _new_model_from(model, d_ff_list=new_d_ff_list)
    _copy_shared_weights(model, new_model)

    with torch.no_grad():
        for i, (ob, nb) in enumerate(zip(model.blocks, new_model.blocks)):
            _copy_block_norms(ob, nb)
            _copy_block_attn(ob, nb)

            if i == layer_idx:
                nb.mlp.up.weight.data = resample_2d_rows(ob.mlp.up.weight.data, new_d_ff)
                nb.mlp.up.bias.data = resample_1d(ob.mlp.up.bias.data, new_d_ff)
                nb.mlp.down.weight.data = resample_2d_cols(ob.mlp.down.weight.data, new_d_ff)
                nb.mlp.down.weight.data *= (old_d_ff / new_d_ff)
                nb.mlp.down.bias.data = ob.mlp.down.bias.data.clone()
            else:
                _copy_block_mlp(ob, nb)

    _resample_transformer_optimizer_state(optimizer, model, new_model)
    return new_model


def expand_transformer_mlp_net2net(
    model: TransformerLM, layer_idx: int, optimizer: torch.optim.Optimizer,
    noise_std: float = 1e-3,
) -> TransformerLM:
    """Expand a single transformer layer's MLP d_ff via Net2Net (duplicate)."""
    mlp = model.blocks[layer_idx].mlp
    old_d_ff = mlp.d_ff
    new_d_ff = old_d_ff * 2
    new_d_ff_list = list(model.d_ff_list)
    new_d_ff_list[layer_idx] = new_d_ff

    new_model = _new_model_from(model, d_ff_list=new_d_ff_list)
    _copy_shared_weights(model, new_model)

    with torch.no_grad():
        for i, (ob, nb) in enumerate(zip(model.blocks, new_model.blocks)):
            _copy_block_norms(ob, nb)
            _copy_block_attn(ob, nb)

            if i == layer_idx:
                w_up = ob.mlp.up.weight.data
                nb.mlp.up.weight.data = torch.cat(
                    [w_up, w_up.clone() + torch.randn_like(w_up) * noise_std], dim=0
                )
                b_up = ob.mlp.up.bias.data
                nb.mlp.up.bias.data = torch.cat(
                    [b_up, b_up.clone() + torch.randn_like(b_up) * noise_std], dim=0
                )
                w_down = ob.mlp.down.weight.data * 0.5
                nb.mlp.down.weight.data = torch.cat([w_down, w_down], dim=1)
                nb.mlp.down.bias.data = ob.mlp.down.bias.data.clone()
            else:
                _copy_block_mlp(ob, nb)

    _duplicate_transformer_optimizer_state(optimizer, model, new_model)
    return new_model


# ---------------------------------------------------------------------------
# Attention head expansion
# ---------------------------------------------------------------------------


def _extract_per_head_qkv(
    qkv_weight: torch.Tensor, qkv_bias: torch.Tensor, n_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape fused QKV into per-head format.

    Args:
        qkv_weight: [3*H*D, d_model]
        qkv_bias: [3*H*D]
        n_heads: H
        head_dim: D

    Returns:
        weight: [3, H, D, d_model]
        bias: [3, H, D]
    """
    d_model = qkv_weight.shape[1]
    w = qkv_weight.view(3, n_heads, head_dim, d_model)
    b = qkv_bias.view(3, n_heads, head_dim)
    return w, b


def _fuse_per_head_qkv(
    weight: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape per-head QKV back to fused format.

    Args:
        weight: [3, H, D, d_model]
        bias: [3, H, D]

    Returns:
        fused_weight: [3*H*D, d_model]
        fused_bias: [3*H*D]
    """
    fused_w = weight.reshape(-1, weight.shape[-1])  # [3*H*D, d_model]
    fused_b = bias.reshape(-1)  # [3*H*D]
    return fused_w, fused_b


def _extract_per_head_proj(
    proj_weight: torch.Tensor, n_heads: int, head_dim: int
) -> torch.Tensor:
    """Reshape output projection to per-head format.

    Args:
        proj_weight: [d_model, H*D]

    Returns:
        [d_model, H, D]
    """
    d_model = proj_weight.shape[0]
    return proj_weight.view(d_model, n_heads, head_dim)


def _fuse_per_head_proj(weight: torch.Tensor) -> torch.Tensor:
    """Reshape per-head output projection back to fused format.

    Args:
        weight: [d_model, H, D]

    Returns:
        [d_model, H*D]
    """
    return weight.reshape(weight.shape[0], -1)


def expand_transformer_attn_continuous(
    model: TransformerLM, layer_idx: int, optimizer: torch.optim.Optimizer,
) -> TransformerLM:
    """Expand attention head_dim via CSR (double it) for all heads in a layer.

    Per head:
    1. Construct feature vectors from Q, K, V columns and O rows
    2. Sort via spectral seriation
    3. Resample Q, K, V (columns) and O (rows) via linear interpolation
    4. Apply scaling corrections:
       - W_Q, W_K: scale by (old_d_k / new_d_k)^0.25 = 2^{-0.25}
       - W_O (proj): scale by old_d_k / new_d_k = 0.5
    """
    attn = model.blocks[layer_idx].attn
    n_heads = attn.n_heads
    old_hd = attn.head_dim
    new_hd = old_hd * 2
    d_model = model.d_model

    # Extract per-head weights
    qkv_w, qkv_b = _extract_per_head_qkv(
        attn.qkv.weight.data, attn.qkv.bias.data, n_heads, old_hd
    )  # [3, H, D, d_model], [3, H, D]
    proj_w = _extract_per_head_proj(
        attn.proj.weight.data, n_heads, old_hd
    )  # [d_model, H, D]

    # Sort and resample each head independently
    new_qkv_w = torch.zeros(3, n_heads, new_hd, d_model, device=qkv_w.device)
    new_qkv_b = torch.zeros(3, n_heads, new_hd, device=qkv_b.device)
    new_proj_w = torch.zeros(d_model, n_heads, new_hd, device=proj_w.device)

    qk_scale = (old_hd / new_hd) ** 0.25  # 2^{-0.25} for doubling

    for h in range(n_heads):
        # Feature vector for each dimension i: concat Q[:,i], K[:,i], V[:,i], O[i,:]
        # Q: qkv_w[0, h] -> [D, d_model]
        # K: qkv_w[1, h] -> [D, d_model]
        # V: qkv_w[2, h] -> [D, d_model]
        # O: proj_w[:, h] -> [d_model, D], transpose to [D, d_model]
        w_q = qkv_w[0, h]  # [D, d_model]
        w_k = qkv_w[1, h]  # [D, d_model]
        w_v = qkv_w[2, h]  # [D, d_model]
        w_o = proj_w[:, h].T  # [D, d_model]

        incoming = torch.cat([w_q, w_k, w_v, w_o], dim=1)  # [D, 4*d_model]
        # For spectral_sort we need "incoming" and "outgoing" — we use the combined
        # feature vector as "incoming" and a dummy outgoing (the function just needs
        # something to concatenate). Actually, let's use spectral_sort's actual
        # interface: incoming=[D, features], outgoing=[features2, D].
        # We can split the features: use Q+K+V as "incoming" and O^T as "outgoing"
        qkv_features = torch.cat([w_q, w_k, w_v], dim=1)  # [D, 3*d_model]
        o_features = proj_w[:, h]  # [d_model, D] — this IS the outgoing matrix

        perm = spectral_sort(qkv_features, o_features)

        # Permute all four matrices for this head
        w_q_sorted = w_q[perm]
        w_k_sorted = w_k[perm]
        w_v_sorted = w_v[perm]
        w_o_sorted = proj_w[:, h][:, perm]  # [d_model, D] -> permute cols

        b_q_sorted = qkv_b[0, h][perm]
        b_k_sorted = qkv_b[1, h][perm]
        b_v_sorted = qkv_b[2, h][perm]

        # Resample: rows of Q/K/V (dimensions), cols of O
        new_qkv_w[0, h] = resample_2d_rows(w_q_sorted, new_hd) * qk_scale
        new_qkv_w[1, h] = resample_2d_rows(w_k_sorted, new_hd) * qk_scale
        new_qkv_w[2, h] = resample_2d_rows(w_v_sorted, new_hd)  # V: no QK scaling
        new_proj_w[:, h] = resample_2d_cols(w_o_sorted, new_hd) * (old_hd / new_hd)  # energy

        new_qkv_b[0, h] = resample_1d(b_q_sorted, new_hd) * qk_scale
        new_qkv_b[1, h] = resample_1d(b_k_sorted, new_hd) * qk_scale
        new_qkv_b[2, h] = resample_1d(b_v_sorted, new_hd)

    # Build new model
    new_hd_list = list(model.head_dim_list)
    new_hd_list[layer_idx] = new_hd
    new_model = _new_model_from(model, head_dim_list=new_hd_list)
    _copy_shared_weights(model, new_model)

    with torch.no_grad():
        for i, (ob, nb) in enumerate(zip(model.blocks, new_model.blocks)):
            _copy_block_norms(ob, nb)
            _copy_block_mlp(ob, nb)

            if i == layer_idx:
                nb.attn.qkv.weight.data, nb.attn.qkv.bias.data = _fuse_per_head_qkv(
                    new_qkv_w, new_qkv_b
                )
                nb.attn.proj.weight.data = _fuse_per_head_proj(new_proj_w)
                nb.attn.proj.bias.data = ob.attn.proj.bias.data.clone()
            else:
                _copy_block_attn(ob, nb)

    _resample_transformer_optimizer_state(optimizer, model, new_model)
    return new_model


def expand_transformer_attn_net2net(
    model: TransformerLM, layer_idx: int, optimizer: torch.optim.Optimizer,
    noise_std: float = 1e-3,
) -> TransformerLM:
    """Expand attention head_dim via Net2Net (duplicate) for all heads in a layer.

    Each internal dimension is duplicated. Q/K scaled by 2^{-0.25},
    O scaled by 0.5, noise added to Q/K/V for symmetry breaking.
    """
    attn = model.blocks[layer_idx].attn
    n_heads = attn.n_heads
    old_hd = attn.head_dim
    new_hd = old_hd * 2
    d_model = model.d_model

    qkv_w, qkv_b = _extract_per_head_qkv(
        attn.qkv.weight.data, attn.qkv.bias.data, n_heads, old_hd
    )
    proj_w = _extract_per_head_proj(attn.proj.weight.data, n_heads, old_hd)

    new_qkv_w = torch.zeros(3, n_heads, new_hd, d_model, device=qkv_w.device)
    new_qkv_b = torch.zeros(3, n_heads, new_hd, device=qkv_b.device)
    new_proj_w = torch.zeros(d_model, n_heads, new_hd, device=proj_w.device)

    qk_scale = (old_hd / new_hd) ** 0.25

    for h in range(n_heads):
        for qkv_idx in range(3):  # Q=0, K=1, V=2
            w = qkv_w[qkv_idx, h]  # [D, d_model]
            b = qkv_b[qkv_idx, h]  # [D]
            w_dup = w.clone() + torch.randn_like(w) * noise_std
            b_dup = b.clone() + torch.randn_like(b) * noise_std

            scale = qk_scale if qkv_idx < 2 else 1.0  # Scale Q, K; not V
            new_qkv_w[qkv_idx, h] = torch.cat([w * scale, w_dup * scale], dim=0)
            new_qkv_b[qkv_idx, h] = torch.cat([b * scale, b_dup * scale], dim=0)

        # O: duplicate cols, scale by 0.5
        w_o = proj_w[:, h] * 0.5  # [d_model, D]
        new_proj_w[:, h] = torch.cat([w_o, w_o], dim=1)

    new_hd_list = list(model.head_dim_list)
    new_hd_list[layer_idx] = new_hd
    new_model = _new_model_from(model, head_dim_list=new_hd_list)
    _copy_shared_weights(model, new_model)

    with torch.no_grad():
        for i, (ob, nb) in enumerate(zip(model.blocks, new_model.blocks)):
            _copy_block_norms(ob, nb)
            _copy_block_mlp(ob, nb)

            if i == layer_idx:
                nb.attn.qkv.weight.data, nb.attn.qkv.bias.data = _fuse_per_head_qkv(
                    new_qkv_w, new_qkv_b
                )
                nb.attn.proj.weight.data = _fuse_per_head_proj(new_proj_w)
                nb.attn.proj.bias.data = ob.attn.proj.bias.data.clone()
            else:
                _copy_block_attn(ob, nb)

    _duplicate_transformer_optimizer_state(optimizer, model, new_model)
    return new_model


# ---------------------------------------------------------------------------
# Optimizer state helpers
# ---------------------------------------------------------------------------


def _resample_transformer_optimizer_state(optimizer, old_model, new_model):
    """Resample optimizer state for CSR expansion."""
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


def _duplicate_transformer_optimizer_state(optimizer, old_model, new_model):
    """Duplicate optimizer state for Net2Net expansion."""
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
