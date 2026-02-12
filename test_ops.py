"""Tests for ops.py — spectral sorting, resampling, and model expansion.

Key properties to verify:
1. Spectral sort returns a valid permutation
2. Resampling preserves boundary values (align_corners)
3. Net2Net expansion is function-preserving (up to noise)
4. CSR expansion produces correct shapes
5. Optimizer state is correctly permuted and resampled
6. Energy preservation: pre-activations are approximately preserved
"""

import pytest
import torch
import torch.nn as nn

from models import MLP
from ops import (
    cosine_similarity_matrix,
    expand_model_continuous,
    expand_model_net2net,
    permute_layer_neurons,
    resample_1d,
    resample_2d_cols,
    resample_2d_rows,
    spectral_sort,
)


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def small_model(device):
    torch.manual_seed(42)
    return MLP(hidden1=128, hidden2=128).to(device)


@pytest.fixture
def tiny_model(device):
    """Smaller model for faster tests."""
    torch.manual_seed(42)
    return MLP(hidden1=16, hidden2=16).to(device)


# --- Test cosine similarity matrix ---


class TestCosineSimilarity:
    def test_self_similarity_is_one(self):
        v = torch.randn(8, 32)
        S = cosine_similarity_matrix(v)
        diag = S.diag()
        assert torch.allclose(diag, torch.ones(8), atol=1e-6)

    def test_symmetric(self):
        v = torch.randn(8, 32)
        S = cosine_similarity_matrix(v)
        assert torch.allclose(S, S.T, atol=1e-6)

    def test_range(self):
        v = torch.randn(8, 32)
        S = cosine_similarity_matrix(v)
        assert S.min() >= -1.0 - 1e-6
        assert S.max() <= 1.0 + 1e-6


# --- Test spectral sort ---


class TestSpectralSort:
    def test_returns_valid_permutation(self):
        incoming = torch.randn(16, 32)
        outgoing = torch.randn(8, 16)
        perm = spectral_sort(incoming, outgoing)
        assert perm.shape == (16,)
        # Check it's a valid permutation (all indices 0..15 present)
        assert set(perm.tolist()) == set(range(16))

    def test_sorted_similarity_is_smoother(self):
        """After sorting, adjacent neurons should be more similar on average."""
        torch.manual_seed(123)
        # Create weights with some structure (not purely random)
        # Use a smooth function so spectral sort can discover the order
        t = torch.linspace(0, 2 * 3.14159, 16)
        incoming = torch.stack([torch.sin(t + i * 0.1) for i in range(32)], dim=1)
        # Shuffle to destroy the order
        shuffle = torch.randperm(16)
        incoming_shuffled = incoming[shuffle]
        outgoing = torch.randn(8, 16)[:, shuffle]

        perm = spectral_sort(incoming_shuffled, outgoing)
        sorted_incoming = incoming_shuffled[perm]

        # Compute adjacent similarities before and after sort
        def adj_similarity(w):
            sims = []
            for i in range(len(w) - 1):
                cos = nn.functional.cosine_similarity(
                    w[i].unsqueeze(0), w[i + 1].unsqueeze(0)
                )
                sims.append(cos.item())
            return sum(sims) / len(sims)

        sim_shuffled = adj_similarity(incoming_shuffled)
        sim_sorted = adj_similarity(sorted_incoming)
        # Sorted should have higher average adjacent similarity
        assert sim_sorted > sim_shuffled, (
            f"Sorted similarity ({sim_sorted:.4f}) should be > "
            f"shuffled ({sim_shuffled:.4f})"
        )

    def test_dimension_mismatch_raises(self):
        incoming = torch.randn(16, 32)
        outgoing = torch.randn(8, 10)  # Wrong: 10 != 16
        with pytest.raises(AssertionError):
            spectral_sort(incoming, outgoing)


# --- Test permute_layer_neurons ---


class TestPermuteLayerNeurons:
    def test_permutation_preserves_function(self, tiny_model, device):
        """Permuting neurons should not change the model's output."""
        x = torch.randn(4, 784, device=device)
        out_before = tiny_model(x).clone()

        # Create a random permutation and apply to layer 0
        perm = torch.randperm(16)
        permute_layer_neurons(tiny_model, 0, perm)

        out_after = tiny_model(x)
        assert torch.allclose(out_before, out_after, atol=1e-5), (
            f"Max diff: {(out_before - out_after).abs().max():.6f}"
        )

    def test_permutation_preserves_function_layer2(self, tiny_model, device):
        """Permuting layer 2 neurons should not change model output."""
        x = torch.randn(4, 784, device=device)
        out_before = tiny_model(x).clone()

        perm = torch.randperm(16)
        permute_layer_neurons(tiny_model, 1, perm)

        out_after = tiny_model(x)
        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_optimizer_state_permuted(self, tiny_model, device):
        """Optimizer state should be reordered along with weights."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)

        # Do a forward/backward to populate optimizer state
        x = torch.randn(4, 784, device=device)
        loss = tiny_model(x).sum()
        loss.backward()
        optimizer.step()

        # Record optimizer state before permutation
        w1_exp_avg_before = optimizer.state[tiny_model.fc1.weight]["exp_avg"].clone()

        perm = torch.randperm(16)
        permute_layer_neurons(tiny_model, 0, perm, optimizer)

        w1_exp_avg_after = optimizer.state[tiny_model.fc1.weight]["exp_avg"]
        # Verify rows were permuted
        expected = w1_exp_avg_before[perm]
        assert torch.allclose(w1_exp_avg_after, expected, atol=1e-6)


# --- Test resampling ---


class TestResampling:
    def test_1d_identity(self):
        """Resampling to same size should be identity."""
        signal = torch.randn(16)
        result = resample_1d(signal, 16)
        assert torch.allclose(result, signal, atol=1e-5)

    def test_1d_preserves_endpoints(self):
        """With align_corners=True, first and last values are preserved."""
        signal = torch.randn(16)
        result = resample_1d(signal, 32)
        assert torch.allclose(result[0], signal[0], atol=1e-6)
        assert torch.allclose(result[-1], signal[-1], atol=1e-6)

    def test_1d_linear_signal(self):
        """A linear signal should be perfectly resampled."""
        signal = torch.linspace(0, 1, 8)
        result = resample_1d(signal, 16)
        expected = torch.linspace(0, 1, 16)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_2d_rows_shape(self):
        w = torch.randn(16, 32)
        result = resample_2d_rows(w, 32)
        assert result.shape == (32, 32)

    def test_2d_cols_shape(self):
        w = torch.randn(16, 32)
        result = resample_2d_cols(w, 64)
        assert result.shape == (16, 64)

    def test_2d_rows_preserves_endpoints(self):
        w = torch.randn(16, 32)
        result = resample_2d_rows(w, 32)
        assert torch.allclose(result[0], w[0], atol=1e-5)
        assert torch.allclose(result[-1], w[-1], atol=1e-5)

    def test_2d_cols_preserves_endpoints(self):
        w = torch.randn(16, 32)
        result = resample_2d_cols(w, 64)
        assert torch.allclose(result[:, 0], w[:, 0], atol=1e-5)
        assert torch.allclose(result[:, -1], w[:, -1], atol=1e-5)

    def test_2d_rows_identity(self):
        w = torch.randn(16, 32)
        result = resample_2d_rows(w, 16)
        assert torch.allclose(result, w, atol=1e-5)

    def test_2d_cols_identity(self):
        w = torch.randn(16, 32)
        result = resample_2d_cols(w, 32)
        assert torch.allclose(result, w, atol=1e-5)


# --- Test Net2Net expansion ---


class TestNet2NetExpansion:
    def test_output_shape(self, tiny_model, device):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        # Populate optimizer state
        x = torch.randn(4, 784, device=device)
        loss = tiny_model(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_net2net(tiny_model, 32, 32, optimizer, noise_std=0.0)
        assert new_model.hidden_widths == (32, 32)

    def test_function_preserving_no_noise(self, tiny_model, device):
        """Without noise, Net2Net should be exactly function-preserving."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        x = torch.randn(8, 784, device=device)
        out_before = tiny_model(x).clone()

        new_model = expand_model_net2net(tiny_model, 32, 32, optimizer, noise_std=0.0)
        out_after = new_model(x)

        assert torch.allclose(out_before, out_after, atol=1e-4), (
            f"Max diff: {(out_before - out_after).abs().max():.6f}"
        )

    def test_function_approximately_preserving_with_noise(self, tiny_model, device):
        """With small noise, outputs should be close."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        x = torch.randn(8, 784, device=device)
        out_before = tiny_model(x).clone()

        new_model = expand_model_net2net(tiny_model, 32, 32, optimizer, noise_std=1e-3)
        out_after = new_model(x)

        # Should be close but not exact due to noise
        max_diff = (out_before - out_after).abs().max().item()
        assert max_diff < 1.0, f"Output diverged too much: max_diff={max_diff}"

    def test_optimizer_state_has_correct_shapes(self, tiny_model, device):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        x = torch.randn(4, 784, device=device)
        loss = tiny_model(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_net2net(tiny_model, 32, 32, optimizer, noise_std=0.0)

        for param in new_model.parameters():
            if param in optimizer.state:
                state = optimizer.state[param]
                if "exp_avg" in state:
                    assert state["exp_avg"].shape == param.shape, (
                        f"exp_avg shape {state['exp_avg'].shape} != param shape {param.shape}"
                    )
                if "exp_avg_sq" in state:
                    assert state["exp_avg_sq"].shape == param.shape


# --- Test CSR expansion ---


class TestCSRExpansion:
    def test_output_shape(self, tiny_model, device):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        x = torch.randn(4, 784, device=device)
        loss = tiny_model(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_continuous(tiny_model, 32, 32, optimizer)
        assert new_model.hidden_widths == (32, 32)

    def test_output_dimensions(self, tiny_model, device):
        """Verify all weight matrices have correct dimensions after expansion."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        x = torch.randn(4, 784, device=device)
        loss = tiny_model(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_continuous(tiny_model, 32, 32, optimizer)

        assert new_model.fc1.weight.shape == (32, 784)
        assert new_model.fc1.bias.shape == (32,)
        assert new_model.fc2.weight.shape == (32, 32)
        assert new_model.fc2.bias.shape == (32,)
        assert new_model.fc3.weight.shape == (10, 32)
        assert new_model.fc3.bias.shape == (10,)

    def test_optimizer_state_has_correct_shapes(self, tiny_model, device):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        x = torch.randn(4, 784, device=device)
        loss = tiny_model(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_continuous(tiny_model, 32, 32, optimizer)

        for param in new_model.parameters():
            if param in optimizer.state:
                state = optimizer.state[param]
                if "exp_avg" in state:
                    assert state["exp_avg"].shape == param.shape, (
                        f"exp_avg shape {state['exp_avg'].shape} != param shape {param.shape}"
                    )
                if "exp_avg_sq" in state:
                    assert state["exp_avg_sq"].shape == param.shape

    def test_energy_preservation(self, tiny_model, device):
        """After expansion, the expected pre-activation magnitudes should be similar.

        We test this by checking that on random input, the pre-activation values
        of the expanded model are in a similar range to the original.
        """
        torch.manual_seed(42)
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        x = torch.randn(32, 784, device=device)

        # Get pre-activation values from original model
        with torch.no_grad():
            h1_orig = tiny_model.fc1(x.view(32, -1))  # [32, 16]
            h1_act = torch.relu(h1_orig)
            h2_orig = tiny_model.fc2(h1_act)  # [32, 16]

        # Populate optimizer state
        loss = tiny_model(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_continuous(tiny_model, 32, 32, optimizer)

        with torch.no_grad():
            h1_new = new_model.fc1(x.view(32, -1))  # [32, 32]
            h1_new_act = torch.relu(h1_new)
            h2_new = new_model.fc2(h1_new_act)  # [32, 32]

        # The mean magnitude of pre-activations should be in a similar range
        # (not exactly equal due to interpolation introducing new neurons)
        ratio_h1 = h1_new.abs().mean() / h1_orig.abs().mean()
        ratio_h2 = h2_new.abs().mean() / h2_orig.abs().mean()

        # Allow generous tolerance (0.3x to 3x) - the key is we're not way off
        assert 0.3 < ratio_h1.item() < 3.0, f"Layer 1 energy ratio: {ratio_h1:.3f}"
        assert 0.3 < ratio_h2.item() < 3.0, f"Layer 2 energy ratio: {ratio_h2:.3f}"

    def test_model_can_train_after_expansion(self, tiny_model, device):
        """Verify the expanded model can do forward/backward/step without errors."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        x = torch.randn(4, 784, device=device)

        # Populate optimizer state
        loss = tiny_model(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_continuous(tiny_model, 32, 32, optimizer)

        # Should be able to train
        for _ in range(3):
            optimizer.zero_grad()
            out = new_model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()

    def test_fc3_bias_preserved(self, tiny_model, device):
        """The output layer bias should be exactly preserved (no resampling)."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        x = torch.randn(4, 784, device=device)
        loss = tiny_model(x).sum()
        loss.backward()
        optimizer.step()

        original_bias = tiny_model.fc3.bias.data.clone()
        new_model = expand_model_continuous(tiny_model, 32, 32, optimizer)
        assert torch.allclose(new_model.fc3.bias.data, original_bias)


# --- Test Net2Net can also train after expansion ---


class TestNet2NetTrainability:
    def test_model_can_train_after_expansion(self, tiny_model, device):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        x = torch.randn(4, 784, device=device)
        loss = tiny_model(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_net2net(tiny_model, 32, 32, optimizer, noise_std=1e-3)

        for _ in range(3):
            optimizer.zero_grad()
            out = new_model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
