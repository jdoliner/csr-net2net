"""Tests for ops.py — spectral sorting, resampling, and model expansion.

Key properties to verify:
1. Spectral sort returns a valid permutation
2. Resampling preserves boundary values (align_corners)
3. Net2Net expansion is function-preserving (up to noise)
4. CSR expansion produces correct shapes
5. Optimizer state is correctly permuted and resampled
6. Energy preservation: pre-activations are approximately preserved
7. Works with arbitrary depth (2 and 4 hidden layers)
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

# Use a small input dim for fast tests
TEST_INPUT_DIM = 64
TINY_WIDTH = 16
EXPANDED_WIDTH = 32


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def tiny_model_2layer(device):
    """2 hidden layer model for testing."""
    torch.manual_seed(42)
    return MLP(input_dim=TEST_INPUT_DIM, hidden_widths=[TINY_WIDTH, TINY_WIDTH]).to(device)


@pytest.fixture
def tiny_model_4layer(device):
    """4 hidden layer model for testing depth generalization."""
    torch.manual_seed(42)
    return MLP(input_dim=TEST_INPUT_DIM, hidden_widths=[TINY_WIDTH] * 4).to(device)


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
        assert set(perm.tolist()) == set(range(16))

    def test_sorted_similarity_is_smoother(self):
        """After sorting, adjacent neurons should be more similar on average."""
        torch.manual_seed(123)
        t = torch.linspace(0, 2 * 3.14159, 16)
        incoming = torch.stack([torch.sin(t + i * 0.1) for i in range(32)], dim=1)
        shuffle = torch.randperm(16)
        incoming_shuffled = incoming[shuffle]
        outgoing = torch.randn(8, 16)[:, shuffle]

        perm = spectral_sort(incoming_shuffled, outgoing)
        sorted_incoming = incoming_shuffled[perm]

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
        assert sim_sorted > sim_shuffled

    def test_dimension_mismatch_raises(self):
        incoming = torch.randn(16, 32)
        outgoing = torch.randn(8, 10)
        with pytest.raises(AssertionError):
            spectral_sort(incoming, outgoing)


# --- Test permute_layer_neurons ---


class TestPermuteLayerNeurons:
    def test_permutation_preserves_function_layer0(self, tiny_model_2layer, device):
        """Permuting neurons should not change the model's output."""
        tiny_model_2layer.eval()
        x = torch.randn(4, TEST_INPUT_DIM, device=device)
        out_before = tiny_model_2layer(x).clone()

        perm = torch.randperm(TINY_WIDTH)
        permute_layer_neurons(tiny_model_2layer, 0, perm)

        out_after = tiny_model_2layer(x)
        assert torch.allclose(out_before, out_after, atol=1e-5), (
            f"Max diff: {(out_before - out_after).abs().max():.6f}"
        )

    def test_permutation_preserves_function_layer1(self, tiny_model_2layer, device):
        """Permuting layer 1 neurons should not change model output."""
        tiny_model_2layer.eval()
        x = torch.randn(4, TEST_INPUT_DIM, device=device)
        out_before = tiny_model_2layer(x).clone()

        perm = torch.randperm(TINY_WIDTH)
        permute_layer_neurons(tiny_model_2layer, 1, perm)

        out_after = tiny_model_2layer(x)
        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_permutation_preserves_function_4layer(self, tiny_model_4layer, device):
        """Permuting any hidden layer in a 4-layer model preserves function."""
        tiny_model_4layer.eval()
        x = torch.randn(4, TEST_INPUT_DIM, device=device)

        for layer_idx in range(4):
            out_before = tiny_model_4layer(x).clone()
            perm = torch.randperm(TINY_WIDTH)
            permute_layer_neurons(tiny_model_4layer, layer_idx, perm)
            out_after = tiny_model_4layer(x)
            assert torch.allclose(out_before, out_after, atol=1e-5), (
                f"Layer {layer_idx} permutation changed output: "
                f"max diff {(out_before - out_after).abs().max():.6f}"
            )

    def test_optimizer_state_permuted(self, tiny_model_2layer, device):
        """Optimizer state should be reordered along with weights."""
        optimizer = torch.optim.AdamW(tiny_model_2layer.parameters(), lr=1e-3)

        x = torch.randn(4, TEST_INPUT_DIM, device=device)
        loss = tiny_model_2layer(x).sum()
        loss.backward()
        optimizer.step()

        layer0_weight = tiny_model_2layer.layers[0].weight
        exp_avg_before = optimizer.state[layer0_weight]["exp_avg"].clone()

        perm = torch.randperm(TINY_WIDTH)
        permute_layer_neurons(tiny_model_2layer, 0, perm, optimizer)

        exp_avg_after = optimizer.state[layer0_weight]["exp_avg"]
        expected = exp_avg_before[perm]
        assert torch.allclose(exp_avg_after, expected, atol=1e-6)


# --- Test resampling ---


class TestResampling:
    def test_1d_identity(self):
        signal = torch.randn(16)
        result = resample_1d(signal, 16)
        assert torch.allclose(result, signal, atol=1e-5)

    def test_1d_preserves_endpoints(self):
        signal = torch.randn(16)
        result = resample_1d(signal, 32)
        assert torch.allclose(result[0], signal[0], atol=1e-6)
        assert torch.allclose(result[-1], signal[-1], atol=1e-6)

    def test_1d_linear_signal(self):
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
    def test_output_shape(self, tiny_model_2layer, device):
        optimizer = torch.optim.AdamW(tiny_model_2layer.parameters(), lr=1e-3)
        x = torch.randn(4, TEST_INPUT_DIM, device=device)
        loss = tiny_model_2layer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_net2net(tiny_model_2layer, EXPANDED_WIDTH, optimizer, noise_std=0.0)
        assert new_model.hidden_widths == (EXPANDED_WIDTH, EXPANDED_WIDTH)

    def test_function_preserving_no_noise(self, tiny_model_2layer, device):
        """Without noise, Net2Net should be exactly function-preserving."""
        tiny_model_2layer.eval()
        optimizer = torch.optim.AdamW(tiny_model_2layer.parameters(), lr=1e-3)
        x = torch.randn(8, TEST_INPUT_DIM, device=device)
        out_before = tiny_model_2layer(x).clone()

        new_model = expand_model_net2net(tiny_model_2layer, EXPANDED_WIDTH, optimizer, noise_std=0.0)
        new_model.eval()
        out_after = new_model(x)

        assert torch.allclose(out_before, out_after, atol=1e-4), (
            f"Max diff: {(out_before - out_after).abs().max():.6f}"
        )

    def test_function_preserving_4layer(self, tiny_model_4layer, device):
        """Net2Net should be function-preserving on a 4-layer model."""
        tiny_model_4layer.eval()
        optimizer = torch.optim.AdamW(tiny_model_4layer.parameters(), lr=1e-3)
        x = torch.randn(8, TEST_INPUT_DIM, device=device)
        out_before = tiny_model_4layer(x).clone()

        new_model = expand_model_net2net(tiny_model_4layer, EXPANDED_WIDTH, optimizer, noise_std=0.0)
        new_model.eval()
        out_after = new_model(x)

        assert torch.allclose(out_before, out_after, atol=1e-4), (
            f"Max diff: {(out_before - out_after).abs().max():.6f}"
        )

    def test_function_approximately_preserving_with_noise(self, tiny_model_2layer, device):
        """With small noise, outputs should be close."""
        optimizer = torch.optim.AdamW(tiny_model_2layer.parameters(), lr=1e-3)
        x = torch.randn(8, TEST_INPUT_DIM, device=device)
        out_before = tiny_model_2layer(x).clone()

        new_model = expand_model_net2net(tiny_model_2layer, EXPANDED_WIDTH, optimizer, noise_std=1e-3)
        out_after = new_model(x)

        max_diff = (out_before - out_after).abs().max().item()
        assert max_diff < 1.0, f"Output diverged too much: max_diff={max_diff}"

    def test_optimizer_state_has_correct_shapes(self, tiny_model_2layer, device):
        optimizer = torch.optim.AdamW(tiny_model_2layer.parameters(), lr=1e-3)
        x = torch.randn(4, TEST_INPUT_DIM, device=device)
        loss = tiny_model_2layer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_net2net(tiny_model_2layer, EXPANDED_WIDTH, optimizer, noise_std=0.0)

        for param in new_model.parameters():
            if param in optimizer.state:
                state = optimizer.state[param]
                if "exp_avg" in state:
                    assert state["exp_avg"].shape == param.shape
                if "exp_avg_sq" in state:
                    assert state["exp_avg_sq"].shape == param.shape


# --- Test CSR expansion ---


class TestCSRExpansion:
    def test_output_shape(self, tiny_model_2layer, device):
        optimizer = torch.optim.AdamW(tiny_model_2layer.parameters(), lr=1e-3)
        x = torch.randn(4, TEST_INPUT_DIM, device=device)
        loss = tiny_model_2layer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_continuous(tiny_model_2layer, EXPANDED_WIDTH, optimizer)
        assert new_model.hidden_widths == (EXPANDED_WIDTH, EXPANDED_WIDTH)

    def test_output_shape_4layer(self, tiny_model_4layer, device):
        optimizer = torch.optim.AdamW(tiny_model_4layer.parameters(), lr=1e-3)
        x = torch.randn(4, TEST_INPUT_DIM, device=device)
        loss = tiny_model_4layer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_continuous(tiny_model_4layer, EXPANDED_WIDTH, optimizer)
        assert new_model.hidden_widths == (EXPANDED_WIDTH,) * 4

    def test_output_dimensions(self, tiny_model_2layer, device):
        """Verify all weight matrices have correct dimensions after expansion."""
        optimizer = torch.optim.AdamW(tiny_model_2layer.parameters(), lr=1e-3)
        x = torch.randn(4, TEST_INPUT_DIM, device=device)
        loss = tiny_model_2layer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_continuous(tiny_model_2layer, EXPANDED_WIDTH, optimizer)

        # layers[0]: input -> hidden1
        assert new_model.layers[0].weight.shape == (EXPANDED_WIDTH, TEST_INPUT_DIM)
        assert new_model.layers[0].bias.shape == (EXPANDED_WIDTH,)
        # layers[1]: hidden1 -> hidden2
        assert new_model.layers[1].weight.shape == (EXPANDED_WIDTH, EXPANDED_WIDTH)
        assert new_model.layers[1].bias.shape == (EXPANDED_WIDTH,)
        # layers[2]: hidden2 -> output
        assert new_model.layers[2].weight.shape == (10, EXPANDED_WIDTH)
        assert new_model.layers[2].bias.shape == (10,)

    def test_optimizer_state_has_correct_shapes(self, tiny_model_2layer, device):
        optimizer = torch.optim.AdamW(tiny_model_2layer.parameters(), lr=1e-3)
        x = torch.randn(4, TEST_INPUT_DIM, device=device)
        loss = tiny_model_2layer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_continuous(tiny_model_2layer, EXPANDED_WIDTH, optimizer)

        for param in new_model.parameters():
            if param in optimizer.state:
                state = optimizer.state[param]
                if "exp_avg" in state:
                    assert state["exp_avg"].shape == param.shape
                if "exp_avg_sq" in state:
                    assert state["exp_avg_sq"].shape == param.shape

    def test_energy_preservation(self, tiny_model_2layer, device):
        """Pre-activation magnitudes should be in a similar range after expansion."""
        torch.manual_seed(42)
        optimizer = torch.optim.AdamW(tiny_model_2layer.parameters(), lr=1e-3)
        x = torch.randn(32, TEST_INPUT_DIM, device=device)

        with torch.no_grad():
            h1_orig = tiny_model_2layer.layers[0](x.view(32, -1))
            h1_act = torch.relu(h1_orig)
            h2_orig = tiny_model_2layer.layers[1](h1_act)

        loss = tiny_model_2layer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_continuous(tiny_model_2layer, EXPANDED_WIDTH, optimizer)

        with torch.no_grad():
            h1_new = new_model.layers[0](x.view(32, -1))
            h1_new_act = torch.relu(h1_new)
            h2_new = new_model.layers[1](h1_new_act)

        ratio_h1 = h1_new.abs().mean() / h1_orig.abs().mean()
        ratio_h2 = h2_new.abs().mean() / h2_orig.abs().mean()

        assert 0.3 < ratio_h1.item() < 3.0, f"Layer 1 energy ratio: {ratio_h1:.3f}"
        assert 0.3 < ratio_h2.item() < 3.0, f"Layer 2 energy ratio: {ratio_h2:.3f}"

    def test_model_can_train_after_expansion(self, tiny_model_2layer, device):
        """Verify the expanded model can do forward/backward/step without errors."""
        optimizer = torch.optim.AdamW(tiny_model_2layer.parameters(), lr=1e-3)
        x = torch.randn(4, TEST_INPUT_DIM, device=device)
        loss = tiny_model_2layer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_continuous(tiny_model_2layer, EXPANDED_WIDTH, optimizer)

        for _ in range(3):
            optimizer.zero_grad()
            out = new_model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()

    def test_model_can_train_after_expansion_4layer(self, tiny_model_4layer, device):
        """4-layer model can train after CSR expansion."""
        optimizer = torch.optim.AdamW(tiny_model_4layer.parameters(), lr=1e-3)
        x = torch.randn(4, TEST_INPUT_DIM, device=device)
        loss = tiny_model_4layer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_continuous(tiny_model_4layer, EXPANDED_WIDTH, optimizer)

        for _ in range(3):
            optimizer.zero_grad()
            out = new_model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()

    def test_output_bias_preserved(self, tiny_model_2layer, device):
        """The output layer bias should be exactly preserved (no resampling)."""
        optimizer = torch.optim.AdamW(tiny_model_2layer.parameters(), lr=1e-3)
        x = torch.randn(4, TEST_INPUT_DIM, device=device)
        loss = tiny_model_2layer(x).sum()
        loss.backward()
        optimizer.step()

        original_bias = tiny_model_2layer.layers[-1].bias.data.clone()
        new_model = expand_model_continuous(tiny_model_2layer, EXPANDED_WIDTH, optimizer)
        assert torch.allclose(new_model.layers[-1].bias.data, original_bias)


# --- Test Net2Net trainability ---


class TestNet2NetTrainability:
    def test_model_can_train_after_expansion(self, tiny_model_2layer, device):
        optimizer = torch.optim.AdamW(tiny_model_2layer.parameters(), lr=1e-3)
        x = torch.randn(4, TEST_INPUT_DIM, device=device)
        loss = tiny_model_2layer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_net2net(tiny_model_2layer, EXPANDED_WIDTH, optimizer, noise_std=1e-3)

        for _ in range(3):
            optimizer.zero_grad()
            out = new_model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()

    def test_model_can_train_after_expansion_4layer(self, tiny_model_4layer, device):
        optimizer = torch.optim.AdamW(tiny_model_4layer.parameters(), lr=1e-3)
        x = torch.randn(4, TEST_INPUT_DIM, device=device)
        loss = tiny_model_4layer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_model_net2net(tiny_model_4layer, EXPANDED_WIDTH, optimizer, noise_std=1e-3)

        for _ in range(3):
            optimizer.zero_grad()
            out = new_model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
