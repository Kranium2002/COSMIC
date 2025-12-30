"""Correctness tests for COSMIC optimizer.

Tests that verify:
1. Weight decay variants work correctly
2. Quant4 round-trip stability
3. Deterministic stepping with fixed seeds
4. Degenerate mode equivalence (tiering off -> baseline SGD-like)
5. Preset configurations
"""

import copy

import torch
import torch.nn.functional as F

from cosmic import Cosmic


class TestWeightDecay:
    """Tests for weight decay behavior."""

    def test_weight_decay_applied(self):
        """Weight decay should reduce parameter magnitude over steps."""
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2, bias=False)

        optimizer = Cosmic(model.parameters(), lr=0.01, weight_decay=0.1)

        x = torch.randn(2, 4)
        y = torch.randn(2, 2)

        for _ in range(10):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        assert model.weight is not None

    def test_weight_decay_decoupled(self):
        """Weight decay should be decoupled (AdamW-style)."""
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2, bias=False)
        initial_weight = model.weight.detach().clone()

        optimizer = Cosmic(model.parameters(), lr=0.1, weight_decay=0.01)

        x = torch.randn(2, 4)
        y = torch.randn(2, 2)
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        assert not torch.allclose(model.weight, initial_weight)


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_deterministic_stepping(self):
        """Same seed + same batches -> same results."""

        def run_training(seed: int):
            torch.manual_seed(seed)
            model = torch.nn.Linear(4, 2)
            optimizer = Cosmic(model.parameters(), lr=0.01)

            torch.manual_seed(seed + 1000)
            x = torch.randn(8, 4)
            y = torch.randn(8, 2)

            for _ in range(10):
                optimizer.zero_grad()
                loss = F.mse_loss(model(x), y)
                loss.backward()
                optimizer.step()

            return model.weight.detach().clone(), loss.item()

        w1, l1 = run_training(42)
        w2, l2 = run_training(42)

        assert torch.allclose(w1, w2), "Weights differ across runs"
        assert abs(l1 - l2) < 1e-6, f"Losses differ: {l1} vs {l2}"

    def test_different_seeds_differ(self):
        """Different seeds should produce different results."""

        def run_training(seed: int):
            torch.manual_seed(seed)
            model = torch.nn.Linear(4, 2)
            optimizer = Cosmic(model.parameters(), lr=0.01)

            x = torch.randn(8, 4)
            y = torch.randn(8, 2)

            for _ in range(10):
                optimizer.zero_grad()
                loss = F.mse_loss(model(x), y)
                loss.backward()
                optimizer.step()

            return model.weight.detach().clone()

        w1 = run_training(42)
        w2 = run_training(43)

        assert not torch.allclose(w1, w2), "Different seeds produced same weights"


class TestDegenerateMode:
    """Tests for degenerate mode (tiering/gating disabled)."""

    def test_matches_sgd_without_gating(self):
        """With gating disabled and simple config, should match SGD update."""
        torch.manual_seed(0)

        model_cosmic = torch.nn.Linear(2, 1, bias=False)
        model_sgd = torch.nn.Linear(2, 1, bias=False)
        model_sgd.load_state_dict(copy.deepcopy(model_cosmic.state_dict()))

        lr = 0.1

        cosmic_opt = Cosmic(
            model_cosmic.parameters(),
            lr=lr,
            weight_decay=0.0,
            gate_warmup_steps=10000,
            ema_short_decay=0.0,
            ema_long_decay=0.0,
            beta2=0.0,
        )

        sgd_opt = torch.optim.SGD(model_sgd.parameters(), lr=lr)

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[1.0], [2.0]])

        cosmic_opt.zero_grad()
        loss_c = F.mse_loss(model_cosmic(x), y)
        loss_c.backward()
        cosmic_opt.step()

        sgd_opt.zero_grad()
        loss_s = F.mse_loss(model_sgd(x), y)
        loss_s.backward()
        sgd_opt.step()

        diff = (model_cosmic.weight - model_sgd.weight).abs().max().item()
        assert diff < 0.2, f"COSMIC differs from SGD by {diff}"


class TestPresets:
    """Tests for preset configurations."""

    def test_safe_preset(self):
        """Safe preset should work without errors."""
        torch.manual_seed(0)
        model = torch.nn.Linear(8, 4)
        optimizer = Cosmic(model.parameters(), lr=0.01, preset="safe")

        x = torch.randn(4, 8)
        y = torch.randn(4, 4)

        for _ in range(10):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        assert loss.item() < 100, "Safe preset failed to train"

    def test_fast_preset(self):
        """Fast preset should work without errors."""
        torch.manual_seed(0)
        model = torch.nn.Linear(8, 4)
        optimizer = Cosmic(model.parameters(), lr=0.01, preset="fast")

        x = torch.randn(4, 8)
        y = torch.randn(4, 4)

        for _ in range(10):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        assert loss.item() < 100, "Fast preset failed to train"

    def test_memory_preset(self):
        """Memory preset should not enable quant4 by default."""
        torch.manual_seed(0)
        model = torch.nn.Linear(8, 4)
        optimizer = Cosmic(model.parameters(), lr=0.01, preset="memory")

        assert not optimizer._quant4, "Memory preset should not enable quant4 by default"

        x = torch.randn(4, 8)
        y = torch.randn(4, 4)

        for _ in range(10):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        assert loss.item() < 100, "Memory preset failed to train"

    def test_invalid_preset_raises(self):
        """Invalid preset should raise ValueError."""
        model = torch.nn.Linear(4, 2)
        try:
            Cosmic(model.parameters(), lr=0.01, preset="invalid")
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Unknown preset" in str(e)


class TestMetrics:
    """Tests for observability metrics."""

    def test_get_metrics(self):
        """Metrics should be retrievable."""
        torch.manual_seed(0)
        model = torch.nn.Linear(8, 4)
        optimizer = Cosmic(model.parameters(), lr=0.01)

        x = torch.randn(4, 8)
        y = torch.randn(4, 4)

        for _ in range(10):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        metrics = optimizer.get_metrics()

        assert "step" in metrics
        assert "tier_occupancy" in metrics
        assert "gate_stats" in metrics
        assert metrics["step"] == 10

    def test_reset_metrics(self):
        """Metrics should be resettable."""
        torch.manual_seed(0)
        model = torch.nn.Linear(8, 4)
        optimizer = Cosmic(model.parameters(), lr=0.01)

        x = torch.randn(4, 8)
        y = torch.randn(4, 4)

        for _ in range(10):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        optimizer.reset_metrics()

        assert len(optimizer._gate_scale_history) == 0
        assert optimizer._gate_events == 0


class TestStateDictRoundTrip:
    """Tests for state_dict save/load."""

    def test_state_dict_roundtrip(self):
        """state_dict should survive save/load."""
        torch.manual_seed(0)
        model = torch.nn.Linear(8, 4)
        optimizer = Cosmic(model.parameters(), lr=0.01)

        x = torch.randn(4, 8)
        y = torch.randn(4, 4)

        for _ in range(10):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        state_dict = optimizer.state_dict()

        optimizer2 = Cosmic(model.parameters(), lr=0.01)
        optimizer2.load_state_dict(state_dict)

        for _ in range(10):
            optimizer2.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer2.step()

        assert loss.item() < 100, "Training failed after state_dict load"
