"""Tests for transformer expansion operations.

Verifies:
1. TransformerLM forward pass works
2. Similarity scoring returns correct count and doesn't modify model
3. Net2Net expansion is function-preserving (no noise)
4. CSR expansion produces correct shapes
5. Optimizer state has correct shapes after expansion
6. Models can train after expansion
7. Only target layer's MLP is modified
"""

import pytest
import torch

from transformer_model import TransformerLM
from transformer_ops import (
    expand_transformer_mlp_continuous,
    expand_transformer_mlp_net2net,
    params_after_doubling_mlp,
    transformer_layer_similarity_scores,
)

VOCAB_SIZE = 256  # Small vocab for tests
D_MODEL = 32
N_HEADS = 2
N_LAYERS = 3
D_FF = 64
MAX_SEQ_LEN = 16


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def tiny_transformer(device):
    torch.manual_seed(42)
    return TransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff_list=[D_FF] * N_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,  # No dropout for deterministic tests
    ).to(device)


class TestTransformerModel:
    def test_forward_shape(self, tiny_transformer, device):
        x = torch.randint(0, VOCAB_SIZE, (2, 8), device=device)
        logits = tiny_transformer(x)
        assert logits.shape == (2, 8, VOCAB_SIZE)

    def test_d_ff_list(self, tiny_transformer):
        assert tiny_transformer.d_ff_list == [D_FF] * N_LAYERS

    def test_asymmetric_d_ff(self, device):
        torch.manual_seed(42)
        model = TransformerLM(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=3,
            d_ff_list=[64, 128, 256],
            max_seq_len=MAX_SEQ_LEN,
        ).to(device)
        assert model.d_ff_list == [64, 128, 256]
        x = torch.randint(0, VOCAB_SIZE, (2, 8), device=device)
        logits = model(x)
        assert logits.shape == (2, 8, VOCAB_SIZE)


class TestSimilarityScores:
    def test_returns_correct_count(self, tiny_transformer):
        scores = transformer_layer_similarity_scores(tiny_transformer)
        assert len(scores) == N_LAYERS

    def test_scores_in_range(self, tiny_transformer):
        scores = transformer_layer_similarity_scores(tiny_transformer)
        for s in scores:
            assert -1.0 <= s <= 1.0

    def test_does_not_modify_model(self, tiny_transformer, device):
        x = torch.randint(0, VOCAB_SIZE, (2, 8), device=device)
        out_before = tiny_transformer(x).clone()
        _ = transformer_layer_similarity_scores(tiny_transformer)
        out_after = tiny_transformer(x)
        assert torch.allclose(out_before, out_after, atol=1e-6)


class TestParamsAfterDoubling:
    def test_increases_params(self, tiny_transformer):
        current = sum(p.numel() for p in tiny_transformer.parameters())
        for i in range(N_LAYERS):
            new_p = params_after_doubling_mlp(tiny_transformer, i)
            assert new_p > current

    def test_correct_increase(self, tiny_transformer):
        """Doubling d_ff should add d_ff * d_model (up) + d_ff (bias) + d_model * d_ff (down)."""
        current = sum(p.numel() for p in tiny_transformer.parameters())
        new_p = params_after_doubling_mlp(tiny_transformer, 0)
        expected_increase = D_FF * D_MODEL + D_FF + D_MODEL * D_FF
        assert new_p == current + expected_increase


class TestNet2NetExpansion:
    def test_function_preserving_no_noise(self, tiny_transformer, device):
        optimizer = torch.optim.AdamW(tiny_transformer.parameters(), lr=1e-3)
        x = torch.randint(0, VOCAB_SIZE, (2, 8), device=device)
        out_before = tiny_transformer(x).clone()

        new_model = expand_transformer_mlp_net2net(
            tiny_transformer, 1, optimizer, noise_std=0.0
        )
        out_after = new_model(x)

        assert torch.allclose(out_before, out_after, atol=1e-4), (
            f"Max diff: {(out_before - out_after).abs().max():.6f}"
        )

    def test_d_ff_doubled(self, tiny_transformer, device):
        optimizer = torch.optim.AdamW(tiny_transformer.parameters(), lr=1e-3)
        new_model = expand_transformer_mlp_net2net(
            tiny_transformer, 1, optimizer, noise_std=0.0
        )
        expected = [D_FF, D_FF * 2, D_FF]
        assert new_model.d_ff_list == expected

    def test_other_layers_unchanged(self, tiny_transformer, device):
        """Non-target layers should have identical weights."""
        optimizer = torch.optim.AdamW(tiny_transformer.parameters(), lr=1e-3)

        old_w0_up = tiny_transformer.blocks[0].mlp.up.weight.data.clone()
        old_w2_up = tiny_transformer.blocks[2].mlp.up.weight.data.clone()

        new_model = expand_transformer_mlp_net2net(
            tiny_transformer, 1, optimizer, noise_std=0.0
        )

        assert torch.allclose(new_model.blocks[0].mlp.up.weight.data, old_w0_up)
        assert torch.allclose(new_model.blocks[2].mlp.up.weight.data, old_w2_up)


class TestCSRExpansion:
    def test_d_ff_doubled(self, tiny_transformer, device):
        optimizer = torch.optim.AdamW(tiny_transformer.parameters(), lr=1e-3)
        x = torch.randint(0, VOCAB_SIZE, (2, 8), device=device)
        loss = tiny_transformer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_transformer_mlp_continuous(
            tiny_transformer, 0, optimizer
        )
        expected = [D_FF * 2, D_FF, D_FF]
        assert new_model.d_ff_list == expected

    def test_weight_shapes(self, tiny_transformer, device):
        optimizer = torch.optim.AdamW(tiny_transformer.parameters(), lr=1e-3)
        x = torch.randint(0, VOCAB_SIZE, (2, 8), device=device)
        loss = tiny_transformer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_transformer_mlp_continuous(
            tiny_transformer, 0, optimizer
        )

        # Layer 0 MLP should be doubled
        assert new_model.blocks[0].mlp.up.weight.shape == (D_FF * 2, D_MODEL)
        assert new_model.blocks[0].mlp.up.bias.shape == (D_FF * 2,)
        assert new_model.blocks[0].mlp.down.weight.shape == (D_MODEL, D_FF * 2)
        assert new_model.blocks[0].mlp.down.bias.shape == (D_MODEL,)

        # Layer 1 should be unchanged
        assert new_model.blocks[1].mlp.up.weight.shape == (D_FF, D_MODEL)

    def test_optimizer_state_shapes(self, tiny_transformer, device):
        optimizer = torch.optim.AdamW(tiny_transformer.parameters(), lr=1e-3)
        x = torch.randint(0, VOCAB_SIZE, (2, 8), device=device)
        loss = tiny_transformer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_transformer_mlp_continuous(
            tiny_transformer, 1, optimizer
        )

        for param in new_model.parameters():
            if param in optimizer.state:
                state = optimizer.state[param]
                if "exp_avg" in state:
                    assert state["exp_avg"].shape == param.shape
                if "exp_avg_sq" in state:
                    assert state["exp_avg_sq"].shape == param.shape

    def test_can_train_after_expansion(self, tiny_transformer, device):
        optimizer = torch.optim.AdamW(tiny_transformer.parameters(), lr=1e-3)
        x = torch.randint(0, VOCAB_SIZE, (2, 8), device=device)
        loss = tiny_transformer(x).sum()
        loss.backward()
        optimizer.step()

        new_model = expand_transformer_mlp_continuous(
            tiny_transformer, 0, optimizer
        )

        for _ in range(3):
            optimizer.zero_grad()
            out = new_model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()

    def test_sequential_expansions(self, tiny_transformer, device):
        """Expand layers 0 then 2, verify shapes and trainability."""
        optimizer = torch.optim.AdamW(tiny_transformer.parameters(), lr=1e-3)
        x = torch.randint(0, VOCAB_SIZE, (2, 8), device=device)
        loss = tiny_transformer(x).sum()
        loss.backward()
        optimizer.step()

        model = expand_transformer_mlp_continuous(tiny_transformer, 0, optimizer)
        assert model.d_ff_list == [D_FF * 2, D_FF, D_FF]

        model = expand_transformer_mlp_continuous(model, 2, optimizer)
        assert model.d_ff_list == [D_FF * 2, D_FF, D_FF * 2]

        for _ in range(3):
            optimizer.zero_grad()
            out = model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()

    def test_embedding_preserved(self, tiny_transformer, device):
        """Token and position embeddings should be exactly preserved."""
        optimizer = torch.optim.AdamW(tiny_transformer.parameters(), lr=1e-3)
        x = torch.randint(0, VOCAB_SIZE, (2, 8), device=device)
        loss = tiny_transformer(x).sum()
        loss.backward()
        optimizer.step()

        old_tok = tiny_transformer.tok_emb.weight.data.clone()
        old_pos = tiny_transformer.pos_emb.weight.data.clone()

        new_model = expand_transformer_mlp_continuous(
            tiny_transformer, 1, optimizer
        )

        assert torch.allclose(new_model.tok_emb.weight.data, old_tok)
        assert torch.allclose(new_model.pos_emb.weight.data, old_pos)
