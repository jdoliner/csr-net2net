"""Core operations for Continuous Signal Resampling and Net2Net expansion.

This module implements:
- Spectral Seriation (canonical sorting of neurons via Fiedler vector)
- Linear resampling of weight matrices
- Full model expansion for both CSR and Net2Net methods
- Optimizer state handling (permutation + interpolation)

All expansion functions work with MLP models of arbitrary depth.
Hidden layer i has:
  - incoming weights: model.layers[i].weight  (shape [hidden_i, prev_dim])
  - outgoing weights: model.layers[i+1].weight  (shape [next_dim, hidden_i])
  - bias: model.layers[i].bias  (shape [hidden_i])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MLP


def cosine_similarity_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix.

    Args:
        vectors: [N, D] tensor of feature vectors.

    Returns:
        [N, N] cosine similarity matrix with values in [-1, 1].
    """
    norms = vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = vectors / norms
    return normalized @ normalized.T


def spectral_sort(incoming: torch.Tensor, outgoing: torch.Tensor) -> torch.Tensor:
    """Compute canonical neuron ordering via Spectral Seriation.

    Constructs feature vectors by concatenating incoming and outgoing weights,
    builds a cosine similarity graph, and returns the permutation that sorts
    neurons by the Fiedler vector (second-smallest eigenvector of the Laplacian).

    Args:
        incoming: [N, D_in] weight matrix (rows = neurons' incoming weights).
        outgoing: [D_out, N] weight matrix (columns = neurons' outgoing weights).

    Returns:
        Permutation indices [N] that sort neurons into canonical order.
    """
    N = incoming.shape[0]
    assert outgoing.shape[1] == N, (
        f"Dimension mismatch: incoming has {N} neurons, outgoing has {outgoing.shape[1]}"
    )

    # Construct feature vectors: concat incoming weights + outgoing weights
    features = torch.cat([incoming, outgoing.T], dim=1)

    # Cosine similarity matrix, shifted to non-negative
    S = cosine_similarity_matrix(features)
    S_shifted = (S + 1.0) / 2.0
    S_shifted.fill_diagonal_(0.0)

    # Graph Laplacian: L = D - S
    D = S_shifted.sum(dim=1)
    L = torch.diag(D) - S_shifted

    # Fiedler vector (second-smallest eigenvector)
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1]

    perm = torch.argsort(fiedler_vector)
    return perm


def permute_layer_neurons(
    model: MLP,
    layer_idx: int,
    perm: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Permute neurons in a hidden layer, updating weights, biases, and optimizer state.

    For hidden layer `layer_idx` (0-indexed into hidden layers):
    - Permute rows of layers[layer_idx].weight and bias
    - Permute columns of layers[layer_idx + 1].weight
    - Permute corresponding optimizer state buffers

    Args:
        model: The MLP model.
        layer_idx: Which hidden layer to permute (0 to num_hidden_layers-1).
        perm: Permutation indices [N].
        optimizer: Optional optimizer whose state should also be permuted.
    """
    current_layer = model.layers[layer_idx]
    next_layer = model.layers[layer_idx + 1]

    with torch.no_grad():
        current_layer.weight.data = current_layer.weight.data[perm]
        current_layer.bias.data = current_layer.bias.data[perm]
        next_layer.weight.data = next_layer.weight.data[:, perm]

    if optimizer is not None:
        _permute_optimizer_state(optimizer, current_layer.weight, perm, dim=0)
        _permute_optimizer_state(optimizer, current_layer.bias, perm, dim=0)
        _permute_optimizer_state(optimizer, next_layer.weight, perm, dim=1)


def _permute_optimizer_state(
    optimizer: torch.optim.Optimizer,
    param: torch.Tensor,
    perm: torch.Tensor,
    dim: int,
) -> None:
    """Permute Adam optimizer state (exp_avg, exp_avg_sq) for a parameter."""
    state = optimizer.state.get(param, {})
    for key in ("exp_avg", "exp_avg_sq"):
        if key in state:
            state[key] = torch.index_select(state[key], dim, perm)


def resample_1d(signal: torch.Tensor, new_size: int) -> torch.Tensor:
    """Resample a 1D signal using linear interpolation with align_corners=True."""
    x = signal.unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=new_size, mode="linear", align_corners=True)
    return x.squeeze(0).squeeze(0)


def resample_2d_rows(weight: torch.Tensor, new_rows: int) -> torch.Tensor:
    """Resample a 2D weight matrix along axis 0 (rows/neurons)."""
    x = weight.T.unsqueeze(0)
    x = F.interpolate(x, size=new_rows, mode="linear", align_corners=True)
    return x.squeeze(0).T


def resample_2d_cols(weight: torch.Tensor, new_cols: int) -> torch.Tensor:
    """Resample a 2D weight matrix along axis 1 (columns/inputs)."""
    x = weight.unsqueeze(0)
    x = F.interpolate(x, size=new_cols, mode="linear", align_corners=True)
    return x.squeeze(0)


def expand_model_continuous(
    model: MLP,
    new_width: int,
    optimizer: torch.optim.Optimizer,
) -> MLP:
    """Expand all hidden layers to new_width via Continuous Signal Resampling.

    Algorithm:
    1. Sort all hidden layers via spectral seriation
    2. Resample all weights/biases to new_width via linear interpolation
    3. Scale outgoing weights by (old_width / new_width) for energy preservation
    4. Apply same operations to optimizer state

    Args:
        model: The MLP to expand.
        new_width: Target width for all hidden layers.
        optimizer: The optimizer (state will be resampled).

    Returns:
        New MLP with expanded widths (same device as input).
    """
    device = next(model.parameters()).device
    num_hidden = model.num_hidden_layers

    # Phase 1: Sort all hidden layers
    for i in range(num_hidden):
        incoming = model.layers[i].weight.data
        outgoing = model.layers[i + 1].weight.data
        perm = spectral_sort(incoming, outgoing)
        permute_layer_neurons(model, i, perm, optimizer)

    # Phase 2 & 3: Build new model and resample
    old_widths = list(model.hidden_widths)
    new_widths = [new_width] * num_hidden

    new_model = MLP(
        input_dim=model.input_dim,
        hidden_widths=new_widths,
        dropout=model.dropout_rate,
        num_classes=model.layers[-1].out_features,
    ).to(device)

    with torch.no_grad():
        for i in range(len(model.layers)):
            old_w = model.layers[i].weight.data
            old_b = model.layers[i].bias.data
            new_layer = new_model.layers[i]

            # Determine what resampling is needed
            old_rows, old_cols = old_w.shape
            new_rows, new_cols = new_layer.weight.shape

            w = old_w

            # Resample columns first (input dimension expansion from previous layer)
            if old_cols != new_cols:
                w = resample_2d_cols(w, new_cols)
                # Energy preservation: scale by old_cols/new_cols
                w = w * (old_cols / new_cols)

            # Resample rows (this layer's neuron expansion)
            if old_rows != new_rows:
                w = resample_2d_rows(w, new_rows)

            new_layer.weight.data = w

            # Bias: resample if size changed, otherwise clone
            if old_b.shape[0] != new_layer.bias.shape[0]:
                new_layer.bias.data = resample_1d(old_b, new_layer.bias.shape[0])
            else:
                new_layer.bias.data = old_b.clone()

    # Resample optimizer state
    _resample_optimizer_state(optimizer, model, new_model)

    return new_model


def expand_model_net2net(
    model: MLP,
    new_width: int,
    optimizer: torch.optim.Optimizer,
    noise_std: float = 1e-3,
) -> MLP:
    """Expand all hidden layers to new_width via Net2Net (duplication with noise).

    Each hidden neuron is duplicated exactly once (2x expansion).
    Incoming weights get small noise added; outgoing weights are scaled by 0.5.

    Args:
        model: The MLP to expand.
        new_width: Target width for all hidden layers (must be 2x current).
        optimizer: The optimizer (state will be duplicated).
        noise_std: Standard deviation of symmetry-breaking noise.

    Returns:
        New MLP with expanded widths.
    """
    device = next(model.parameters()).device
    num_hidden = model.num_hidden_layers

    for i in range(num_hidden):
        old_w = model.layers[i].out_features
        assert new_width == 2 * old_w, (
            f"Net2Net expects exact 2x expansion, but layer {i} has width "
            f"{old_w} and target is {new_width}"
        )

    new_widths = [new_width] * num_hidden

    new_model = MLP(
        input_dim=model.input_dim,
        hidden_widths=new_widths,
        dropout=model.dropout_rate,
        num_classes=model.layers[-1].out_features,
    ).to(device)

    with torch.no_grad():
        for i in range(len(model.layers)):
            old_w = model.layers[i].weight.data
            old_b = model.layers[i].bias.data
            old_rows, old_cols = old_w.shape
            new_layer = new_model.layers[i]
            new_rows, new_cols = new_layer.weight.shape

            is_output_layer = (i == len(model.layers) - 1)

            # Expand columns (input dim) if previous layer was expanded
            if old_cols != new_cols:
                # Duplicate columns and scale by 0.5
                w = torch.cat([old_w * 0.5, old_w * 0.5], dim=1)
            else:
                w = old_w.clone()

            # Expand rows (this layer's neurons) if this is a hidden layer
            if old_rows != new_rows:
                assert not is_output_layer, "Output layer should not be expanded"
                w_dup = w.clone() + torch.randn_like(w) * noise_std
                w = torch.cat([w, w_dup], dim=0)

                b_dup = old_b.clone() + torch.randn_like(old_b) * noise_std
                new_layer.bias.data = torch.cat([old_b, b_dup], dim=0)
            else:
                new_layer.bias.data = old_b.clone()

            new_layer.weight.data = w

    # Duplicate optimizer state
    _duplicate_optimizer_state(optimizer, model, new_model)

    return new_model


def _resample_optimizer_state(
    optimizer: torch.optim.Optimizer,
    old_model: MLP,
    new_model: MLP,
) -> None:
    """Resample optimizer state to match new model dimensions (CSR)."""
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


def _duplicate_optimizer_state(
    optimizer: torch.optim.Optimizer,
    old_model: MLP,
    new_model: MLP,
) -> None:
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
