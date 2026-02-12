"""Core operations for Continuous Signal Resampling and Net2Net expansion.

This module implements:
- Spectral Seriation (canonical sorting of neurons via Fiedler vector)
- Linear resampling of weight matrices
- Full model expansion for both CSR and Net2Net methods
- Optimizer state handling (permutation + interpolation)
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
    # incoming: [N, D_in], outgoing.T: [N, D_out] -> features: [N, D_in + D_out]
    features = torch.cat([incoming, outgoing.T], dim=1)

    # Cosine similarity matrix S in [-1, 1]
    S = cosine_similarity_matrix(features)

    # Shift to non-negative: S' = (S + 1) / 2, range [0, 1]
    S_shifted = (S + 1.0) / 2.0

    # Zero out self-loops for clean Laplacian
    S_shifted.fill_diagonal_(0.0)

    # Graph Laplacian: L = D - S
    D = S_shifted.sum(dim=1)
    L = torch.diag(D) - S_shifted

    # Compute Fiedler vector (eigenvector for second-smallest eigenvalue)
    # Use eigh for symmetric matrices — eigenvalues returned in ascending order
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1]  # Second eigenvector (index 1)

    # Sort by Fiedler vector values
    perm = torch.argsort(fiedler_vector)
    return perm


def permute_layer_neurons(
    model: MLP,
    layer_idx: int,
    perm: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Permute neurons in a hidden layer, updating weights, biases, and optimizer state.

    For hidden layer `layer_idx` (0-indexed: 0=fc1, 1=fc2):
    - Permute rows of W_l and bias_l
    - Permute columns of W_{l+1}
    - Permute corresponding optimizer state buffers

    Args:
        model: The MLP model.
        layer_idx: Which hidden layer to permute (0 or 1).
        perm: Permutation indices [N].
        optimizer: Optional optimizer whose state should also be permuted.
    """
    layers = [model.fc1, model.fc2, model.fc3]
    current_layer = layers[layer_idx]
    next_layer = layers[layer_idx + 1]

    with torch.no_grad():
        # Permute rows of current layer's weight and bias
        current_layer.weight.data = current_layer.weight.data[perm]
        current_layer.bias.data = current_layer.bias.data[perm]

        # Permute columns of next layer's weight
        next_layer.weight.data = next_layer.weight.data[:, perm]

    # Permute optimizer state
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
    """Permute Adam optimizer state (exp_avg, exp_avg_sq) for a parameter.

    Args:
        optimizer: The optimizer.
        param: The parameter tensor whose state to permute.
        perm: Permutation indices.
        dim: Dimension along which to permute.
    """
    state = optimizer.state.get(param, {})
    for key in ("exp_avg", "exp_avg_sq"):
        if key in state:
            state[key] = torch.index_select(state[key], dim, perm)


def resample_1d(signal: torch.Tensor, new_size: int) -> torch.Tensor:
    """Resample a 1D signal (e.g., bias) using linear interpolation.

    Args:
        signal: [N] tensor.
        new_size: Target size.

    Returns:
        [new_size] tensor.
    """
    # interpolate expects [B, C, W] format
    x = signal.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
    x = F.interpolate(x, size=new_size, mode="linear", align_corners=True)
    return x.squeeze(0).squeeze(0)  # [new_size]


def resample_2d_rows(weight: torch.Tensor, new_rows: int) -> torch.Tensor:
    """Resample a 2D weight matrix along axis 0 (rows/neurons).

    Treats each column as an independent 1D signal and resamples.

    Args:
        weight: [N, D] tensor.
        new_rows: Target number of rows.

    Returns:
        [new_rows, D] tensor.
    """
    # interpolate expects [B, C, W]: treat as batch=1, channels=D, width=N
    # We need to interpolate along the row dimension
    x = weight.T.unsqueeze(0)  # [1, D, N]
    x = F.interpolate(x, size=new_rows, mode="linear", align_corners=True)
    return x.squeeze(0).T  # [new_rows, D]


def resample_2d_cols(weight: torch.Tensor, new_cols: int) -> torch.Tensor:
    """Resample a 2D weight matrix along axis 1 (columns/inputs).

    Treats each row as an independent 1D signal and resamples.

    Args:
        weight: [M, N] tensor.
        new_cols: Target number of columns.

    Returns:
        [M, new_cols] tensor.
    """
    # interpolate expects [B, C, W]: treat as batch=1, channels=M, width=N
    x = weight.unsqueeze(0)  # [1, M, N]
    x = F.interpolate(x, size=new_cols, mode="linear", align_corners=True)
    return x.squeeze(0)  # [M, new_cols]


def expand_model_continuous(
    model: MLP,
    new_hidden1: int,
    new_hidden2: int,
    optimizer: torch.optim.Optimizer,
) -> MLP:
    """Expand model via Continuous Signal Resampling.

    Algorithm:
    1. Sort both hidden layers via spectral seriation
    2. Resample all weights/biases to new widths via linear interpolation
    3. Scale outgoing weights by (old_width / new_width) for energy preservation
    4. Apply same operations to optimizer state

    Args:
        model: The small MLP to expand.
        new_hidden1: Target width for hidden layer 1.
        new_hidden2: Target width for hidden layer 2.
        optimizer: The optimizer (state will be resampled).

    Returns:
        New MLP with expanded widths (same device as input).
    """
    device = model.fc1.weight.device

    # Phase 1: Sort both layers
    perm1 = spectral_sort(model.fc1.weight.data, model.fc2.weight.data)
    permute_layer_neurons(model, 0, perm1, optimizer)

    perm2 = spectral_sort(model.fc2.weight.data, model.fc3.weight.data)
    permute_layer_neurons(model, 1, perm2, optimizer)

    # Phase 2 & 3: Resample and build new model
    old_h1 = model.fc1.out_features
    old_h2 = model.fc2.out_features

    new_model = MLP(hidden1=new_hidden1, hidden2=new_hidden2).to(device)

    with torch.no_grad():
        # --- Layer 1 (fc1): resample rows from old_h1 -> new_hidden1 ---
        new_model.fc1.weight.data = resample_2d_rows(model.fc1.weight.data, new_hidden1)
        new_model.fc1.bias.data = resample_1d(model.fc1.bias.data, new_hidden1)

        # --- Layer 2 (fc2): resample cols (from layer 1 expansion), then rows ---
        w2 = resample_2d_cols(model.fc2.weight.data, new_hidden1)  # [old_h2, new_h1]
        w2 = w2 * (old_h1 / new_hidden1)  # Energy preservation for layer 1 expansion
        w2 = resample_2d_rows(w2, new_hidden2)  # [new_h2, new_h1]
        new_model.fc2.weight.data = w2
        new_model.fc2.bias.data = resample_1d(model.fc2.bias.data, new_hidden2)

        # --- Layer 3 (fc3): resample cols from old_h2 -> new_hidden2 ---
        new_model.fc3.weight.data = resample_2d_cols(model.fc3.weight.data, new_hidden2)
        new_model.fc3.weight.data = new_model.fc3.weight.data * (old_h2 / new_hidden2)
        new_model.fc3.bias.data = model.fc3.bias.data.clone()

    # Rebuild optimizer with new parameters and resampled state
    _resample_optimizer_state_continuous(
        optimizer, model, new_model, old_h1, old_h2, new_hidden1, new_hidden2
    )

    return new_model


def _resample_optimizer_state_continuous(
    optimizer: torch.optim.Optimizer,
    old_model: MLP,
    new_model: MLP,
    old_h1: int,
    old_h2: int,
    new_h1: int,
    new_h2: int,
) -> None:
    """Resample optimizer state to match new model dimensions.

    Applies the same interpolation used for weights to the optimizer's
    exp_avg (momentum) and exp_avg_sq (variance) buffers.

    Args:
        optimizer: The optimizer to update.
        old_model: The pre-expansion model.
        new_model: The post-expansion model.
        old_h1: Old hidden layer 1 width.
        old_h2: Old hidden layer 2 width.
        new_h1: New hidden layer 1 width.
        new_h2: New hidden layer 2 width.
    """
    old_params = list(old_model.parameters())
    new_params = list(new_model.parameters())

    # Build the new optimizer state
    new_state = {}
    for old_p, new_p in zip(old_params, new_params):
        if old_p not in optimizer.state:
            continue
        old_s = optimizer.state[old_p]
        new_s = {}

        for key, val in old_s.items():
            if not isinstance(val, torch.Tensor):
                # Non-tensor state (rare, but copy as-is)
                new_s[key] = val
            elif val.dim() == 0:
                # Scalar tensor (e.g., step counter) — clone as-is
                new_s[key] = val.clone()
            elif key in ("exp_avg", "exp_avg_sq"):
                # These are the buffers that need resampling
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
                # Unknown tensor state — clone as-is
                new_s[key] = val.clone()

        new_state[new_p] = new_s

    # Update param_groups to point to new parameters
    optimizer.param_groups[0]["params"] = list(new_model.parameters())

    # Replace optimizer state
    optimizer.state.clear()
    optimizer.state.update(new_state)


def expand_model_net2net(
    model: MLP,
    new_hidden1: int,
    new_hidden2: int,
    optimizer: torch.optim.Optimizer,
    noise_std: float = 1e-3,
) -> MLP:
    """Expand model via Net2Net (duplication with noise).

    Algorithm:
    1. Duplicate each neuron exactly once (128 -> 256)
    2. Add small noise to incoming weights to break symmetry
    3. Scale outgoing weights by 0.5

    Args:
        model: The small MLP to expand.
        new_hidden1: Target width for hidden layer 1.
        new_hidden2: Target width for hidden layer 2.
        optimizer: The optimizer (state will be duplicated).
        noise_std: Standard deviation of symmetry-breaking noise.

    Returns:
        New MLP with expanded widths.
    """
    device = model.fc1.weight.device
    old_h1 = model.fc1.out_features
    old_h2 = model.fc2.out_features

    assert new_hidden1 == 2 * old_h1, "Net2Net expects exact 2x expansion"
    assert new_hidden2 == 2 * old_h2, "Net2Net expects exact 2x expansion"

    new_model = MLP(hidden1=new_hidden1, hidden2=new_hidden2).to(device)

    with torch.no_grad():
        # --- Expand hidden layer 1 ---
        # Duplicate fc1 weights/bias: [old_h1, 784] -> [2*old_h1, 784]
        w1 = model.fc1.weight.data
        b1 = model.fc1.bias.data
        w1_dup = w1.clone()
        b1_dup = b1.clone()
        # Add noise to duplicates
        w1_dup = w1_dup + torch.randn_like(w1_dup) * noise_std
        b1_dup = b1_dup + torch.randn_like(b1_dup) * noise_std
        new_model.fc1.weight.data = torch.cat([w1, w1_dup], dim=0)
        new_model.fc1.bias.data = torch.cat([b1, b1_dup], dim=0)

        # Scale fc2 incoming weights (columns) and duplicate
        w2 = model.fc2.weight.data  # [old_h2, old_h1]
        # Each original neuron now has 2 copies, so scale by 0.5
        w2_scaled = w2 * 0.5
        new_model.fc2.weight.data[:old_h2, :] = torch.cat(
            [w2_scaled, w2_scaled], dim=1
        )  # placeholder, will be overwritten below

        # --- Expand hidden layer 2 ---
        # Duplicate fc2 rows: need to handle both expansions
        # fc2: [old_h2, old_h1] -> [2*old_h2, 2*old_h1]
        # First expand columns (layer 1 expansion): [old_h2, 2*old_h1]
        w2_col_expanded = torch.cat([w2_scaled, w2_scaled], dim=1)
        # Then duplicate rows (layer 2 expansion): [2*old_h2, 2*old_h1]
        w2_row_dup = w2_col_expanded.clone()
        w2_row_dup = w2_row_dup + torch.randn_like(w2_row_dup) * noise_std
        new_model.fc2.weight.data = torch.cat([w2_col_expanded, w2_row_dup], dim=0)

        b2 = model.fc2.bias.data
        b2_dup = b2.clone() + torch.randn_like(b2) * noise_std
        new_model.fc2.bias.data = torch.cat([b2, b2_dup], dim=0)

        # Scale fc3 incoming weights (columns) and duplicate
        w3 = model.fc3.weight.data  # [10, old_h2]
        w3_scaled = w3 * 0.5
        new_model.fc3.weight.data = torch.cat([w3_scaled, w3_scaled], dim=1)
        new_model.fc3.bias.data = model.fc3.bias.data.clone()

    # Duplicate optimizer state
    _duplicate_optimizer_state_net2net(optimizer, old_model=model, new_model=new_model)

    return new_model


def _duplicate_optimizer_state_net2net(
    optimizer: torch.optim.Optimizer,
    old_model: MLP,
    new_model: MLP,
) -> None:
    """Duplicate optimizer state for Net2Net expansion.

    Each buffer is duplicated (concatenated) along the expanded dimension.

    Args:
        optimizer: The optimizer to update.
        old_model: Pre-expansion model.
        new_model: Post-expansion model.
    """
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
                # Scalar tensor (e.g., step counter) — clone as-is
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
