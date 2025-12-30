import torch

from cosmic.extension import get_extension


def test_quant4_pack_unpack():
    """Test that quant4 packing and unpacking preserves values approximately."""
    ext = get_extension()
    original = torch.randn(100, dtype=torch.float32)
    packed, scale, numel = ext.quant4_pack(original)
    unpacked = ext.quant4_unpack(packed, scale, numel)
    assert unpacked.shape == original.shape
    assert torch.allclose(unpacked, original, atol=0.5)


def test_tier_signal():
    """Test that tier_signal returns a reasonable value."""
    ext = get_extension()
    grad = torch.randn(100, dtype=torch.float32)
    signal = ext.tier_signal(grad)
    assert isinstance(signal, float)
    assert signal >= 0


def test_cosmic_fused_step():
    """Test the fused COSMIC dual-EMA kernel."""
    ext = get_extension()

    param = torch.randn(100, dtype=torch.float32)
    grad = torch.randn(100, dtype=torch.float32)
    ema_short = torch.zeros(100, dtype=torch.float32)
    ema_long = torch.zeros(100, dtype=torch.float32)
    exp_avg_sq = torch.zeros(100, dtype=torch.float32)

    param_before = param.clone()
    ema_short_before = ema_short.clone()

    ext.cosmic_fused_step(
        [param],
        [grad],
        [ema_short],
        [ema_long],
        [exp_avg_sq],
        [1.0],
        [0.01],
        1,
        0.9,
        0.99,
        0.999,
        1e-8,
        0.0,
    )

    assert not torch.allclose(param, param_before)
    assert not torch.allclose(ema_short, ema_short_before)
    assert exp_avg_sq.abs().sum() > 0
