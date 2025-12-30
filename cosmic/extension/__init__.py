"""Thin wrapper for the CPU-only C++ extension."""

from __future__ import annotations

from types import ModuleType

import torch

from cosmic.extension.build import build_extension, build_extension_from_env

_extension: ModuleType | None = None


def _try_import_prebuilt() -> ModuleType | None:
    try:
        import importlib

        return importlib.import_module("cosmic_cpu")
    except Exception:
        return None


def get_extension(verbose: bool | None = None) -> ModuleType:
    global _extension
    if _extension is None:
        prebuilt = _try_import_prebuilt()
        if prebuilt is not None:
            _extension = prebuilt
        elif verbose is None:
            _extension = build_extension_from_env()
        else:
            _extension = build_extension(verbose=verbose)
    return _extension


def quant4_pack(tensor: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("quant4_pack expects a torch.Tensor")
    ext = get_extension()
    packed, scale, numel = ext.quant4_pack(tensor)
    return packed, float(scale), int(numel)


def quant4_unpack(packed: torch.Tensor, scale: float, numel: int) -> torch.Tensor:
    if not isinstance(packed, torch.Tensor):
        raise TypeError("quant4_unpack expects a torch.Tensor")
    ext = get_extension()
    return ext.quant4_unpack(packed, float(scale), int(numel))


def quant4_pack_many(
    tensors: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[float], list[int]]:
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("quant4_pack_many expects a list of torch.Tensor")
    ext = get_extension()
    packed_list, scales, numels = ext.quant4_pack_many(tensors)
    return (
        list(packed_list),
        [float(s) for s in scales],
        [int(n) for n in numels],
    )


def quant4_unpack_many(
    packed_list: list[torch.Tensor],
    scales: list[float],
    numels: list[int],
) -> list[torch.Tensor]:
    if not all(isinstance(t, torch.Tensor) for t in packed_list):
        raise TypeError("quant4_unpack_many expects a list of torch.Tensor")
    ext = get_extension()
    return list(ext.quant4_unpack_many(packed_list, scales, numels))


def quant4_pack_blocks(
    tensor: torch.Tensor,
    block_size: int,
    use_zero_point: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int, int, bool]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("quant4_pack_blocks expects a torch.Tensor")
    ext = get_extension()
    packed, scales, numel, blk, use_zp = ext.quant4_pack_blocks(
        tensor, int(block_size), bool(use_zero_point)
    )
    return packed, scales, int(numel), int(blk), bool(use_zp)


def quant4_unpack_blocks(
    packed: torch.Tensor,
    scales: torch.Tensor,
    numel: int,
    block_size: int,
    use_zero_point: bool = False,
) -> torch.Tensor:
    if not isinstance(packed, torch.Tensor):
        raise TypeError("quant4_unpack_blocks expects a torch.Tensor for packed")
    if not isinstance(scales, torch.Tensor):
        raise TypeError("quant4_unpack_blocks expects a torch.Tensor for scales")
    ext = get_extension()
    return ext.quant4_unpack_blocks(
        packed,
        scales,
        int(numel),
        int(block_size),
        bool(use_zero_point),
    )


def tier_signal(grad: torch.Tensor) -> float:
    if not isinstance(grad, torch.Tensor):
        raise TypeError("tier_signal expects a torch.Tensor")
    ext = get_extension()
    return float(ext.tier_signal(grad))


def cosmic_fused_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    ema_shorts: list[torch.Tensor],
    ema_longs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    blends: list[float],
    effective_lrs: list[float],
    step: int,
    decay_short: float,
    decay_long: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> None:
    """Fused COSMIC dual-EMA step.

    This is the main optimized path that fuses all operations into a single
    cache-efficient pass per parameter:

    1. Update short-horizon EMA (fast momentum)
    2. Update long-horizon EMA (slow momentum)
    3. Blend EMAs based on tier
    4. Update second moment (adaptive LR)
    5. Apply parameter update with weight decay

    Args:
        params: Parameter tensors
        grads: Gradient tensors
        ema_shorts: Short-horizon EMA state tensors
        ema_longs: Long-horizon EMA state tensors
        exp_avg_sqs: Second moment state tensors
        blends: Per-parameter blend values (0=long EMA, 1=short EMA)
        effective_lrs: Per-parameter effective learning rates (after tier/gate scaling)
        step: Current step (1-indexed)
        decay_short: Short EMA decay (typically 0.9)
        decay_long: Long EMA decay (typically 0.99)
        beta2: Second moment decay (typically 0.999)
        eps: Numerical stability term
        weight_decay: Decoupled weight decay
    """
    ext = get_extension()
    ext.cosmic_fused_step(
        params,
        grads,
        ema_shorts,
        ema_longs,
        exp_avg_sqs,
        blends,
        effective_lrs,
        step,
        decay_short,
        decay_long,
        beta2,
        eps,
        weight_decay,
    )


def cosmic_fused_step_quant4(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    ema_short_qs: list[torch.Tensor],
    ema_short_scales: list[torch.Tensor],
    ema_long_qs: list[torch.Tensor],
    ema_long_scales: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    blends: list[float],
    effective_lrs: list[float],
    step: int,
    decay_short: float,
    decay_long: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    block_size: int,
) -> None:
    ext = get_extension()
    ext.cosmic_fused_step_quant4(
        params,
        grads,
        ema_short_qs,
        ema_short_scales,
        ema_long_qs,
        ema_long_scales,
        exp_avg_sqs,
        blends,
        effective_lrs,
        step,
        decay_short,
        decay_long,
        beta2,
        eps,
        weight_decay,
        block_size,
    )


def cosmic_fused_step_sign(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    effective_lrs: list[float],
    weight_decay: float,
) -> None:
    ext = get_extension()
    ext.cosmic_fused_step_sign(
        params,
        grads,
        effective_lrs,
        weight_decay,
    )


__all__ = [
    "get_extension",
    "quant4_pack",
    "quant4_unpack",
    "quant4_pack_many",
    "quant4_unpack_many",
    "quant4_pack_blocks",
    "quant4_unpack_blocks",
    "tier_signal",
    "cosmic_fused_step",
    "cosmic_fused_step_quant4",
    "cosmic_fused_step_sign",
]
