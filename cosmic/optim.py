"""Cosmic optimizer wrapper around the CPU-only extension.

COSMIC is a CPU-only PyTorch optimizer that combines:
- SGD-style parameter updates
- Dual EMA momentum (short + long horizon blending)
- Tiered parameter management (Tier2 full state, Tier1 scalar EMA + sign, Tier0 sign-only
  when tiered_state=True)
- Adaptive gating based on gradient signal stability
- Optional 4-bit quantized EMA state for memory efficiency
- Optional tiered state allocation for memory efficiency in lower tiers

Algorithm summary:
1. Compute dual EMA: ema_short = decay_s * ema_short + (1-decay_s) * grad
                     ema_long  = decay_l * ema_long  + (1-decay_l) * grad
2. Blend by tier:    momentum = blend[tier] * ema_short + (1-blend[tier]) * ema_long
3. Gate update:      effective_lr = lr * gate_scale (based on gradient variance/spikes)
4. Update params:    param -= effective_lr * momentum + weight_decay * param

When tiered_state=True, Tier2 uses the dual-EMA/adaptive denom path, while Tier1 uses
sign updates scaled by a scalar EMA magnitude and Tier0 uses sign-only updates.

Presets:
- "safe": Conservative defaults, works across many tasks without tuning
- "fast": Aggressive gating/tiering for speed on stable tasks
- "memory": Tuned for quant4 workflows; pass quant4=True to enable
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch.optim import Optimizer

from cosmic._logging import get_logger, log_event
from cosmic.extension import (
    cosmic_fused_step,
    cosmic_fused_step_quant4,
    cosmic_fused_step_sign,
    quant4_pack,
    quant4_pack_blocks,
    quant4_pack_many,
    quant4_unpack,
    quant4_unpack_blocks,
    tier_signal,
)

PRESETS: dict[str, dict[str, object]] = {
    "safe": {
        "ema_short_decay": 0.9,
        "ema_long_decay": 0.99,
        "gate_tier_max": 1,
        "gate_min_scale": 0.1,
        "gate_warmup_steps": 300,
        "gate_update_interval": 10,
        "quant4": False,
        "tier_config": {
            "tier2_fraction": 0.1,
            "tier1_fraction": 0.2,
            "reassignment_interval": 100,
        },
    },
    "fast": {
        "ema_short_decay": 0.8,
        "ema_long_decay": 0.95,
        "gate_tier_max": 2,
        "gate_min_scale": 0.05,
        "gate_warmup_steps": 100,
        "gate_update_interval": 5,
        "quant4": False,
        "tier_config": {
            "tier2_fraction": 0.2,
            "tier1_fraction": 0.3,
            "reassignment_interval": 50,
            "tier2_fraction_warmup": 0.5,
            "tier1_fraction_warmup": 0.3,
            "warmup_steps": 50,
        },
    },
    "memory": {
        "ema_short_decay": 0.9,
        "ema_long_decay": 0.99,
        "gate_tier_max": 1,
        "gate_min_scale": 0.1,
        "gate_warmup_steps": 200,
        "gate_update_interval": 10,
        "quant4_interval": 5,
        "tier_config": {
            "tier2_fraction": 0.1,
            "tier1_fraction": 0.2,
            "reassignment_interval": 100,
        },
    },
}


@dataclass(frozen=True)
class TierConfig:
    tier2_fraction: float = 0.3
    tier1_fraction: float = 0.3
    reassignment_interval: int = 200
    budget_bytes: int | None = None
    tier2_budget_ratio: float | None = None
    tier2_fraction_warmup: float | None = None
    tier1_fraction_warmup: float | None = None
    warmup_steps: int | None = None
    ema_mismatch_weight: float = 0.0
    min_tier_steps: int | None = 200
    signal_ema_decay: float | None = 0.9
    signal_margin: float = 0.1
    force_tier2_warmup_steps: int | None = 200
    full_sync_interval: int | None = 200
    full_sync_tier: int = 1


class Cosmic(Optimizer):
    """CPU-only optimizer with a C++ multi-tensor update path.

    COSMIC combines SGD updates with dual EMA momentum, tiered parameter
    management, and adaptive gating for stable, efficient CPU training.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        weight_decay: Weight decay coefficient (default: 0.0)
        preset: One of "safe", "fast", or "memory" for predefined configs
        tier_config: TierConfig for tiered parameter management
        tier_lr_scales: Per-tier LR multipliers (tier0, tier1, tier2)
        force_tier2_layers: Layer name substrings to force into tier2
        param_name_map: Mapping from param id to name for layer forcing
        gate_tier_max: Max tier to apply gating (0-2)
        gate_min_scale: Minimum gate scale when triggered
        gate_warmup_steps: Steps before gating activates
        gate_update_interval: Steps between gate scale recomputation
        quant4: Enable 4-bit quantized EMA storage
        quant4_interval: Steps between quantization
        quant4_block_size: Block size for quant4 EMA scales (fused quant4 path)
        ema_short_decay: Short-horizon EMA decay (higher = more smoothing)
        ema_long_decay: Long-horizon EMA decay
        log_every: Log interval (0 = disabled)
        tiered_state: If True, keep full optimizer state only for Tier2; Tier1 uses
            a scalar EMA of |grad| to scale sign updates; Tier0 uses sign updates.
        min_lr_scale: Minimum LR scale applied after tiering/gating (fraction of base LR)

    Example:
        >>> optimizer = Cosmic(model.parameters(), lr=1e-3, preset="safe")
        >>> optimizer = Cosmic(model.parameters(), lr=1e-2, preset="fast")
        >>> optimizer = Cosmic(model.parameters(), lr=1e-3, preset="memory", quant4=True)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        preset: str | None = None,
        tier_config: TierConfig | None = None,
        tier_lr_scales: tuple[float, float, float] | None = None,
        force_tier2_layers: Iterable[str] | None = None,
        param_name_map: dict[object, str] | None = None,
        gate_tier_max: int = 0,
        gate_min_scale: float = 0.2,
        gate_warmup_steps: int = 200,
        gate_update_interval: int = 10,
        quant4: bool = False,
        quant4_interval: int = 1,
        quant4_block_size: int = 256,
        log_every: int = 0,
        ema_short_decay: float = 0.9,
        ema_long_decay: float = 0.99,
        beta2: float = 0.999,
        eps: float = 1e-8,
        use_fused: bool = True,
        tiered_state: bool = True,
        min_lr_scale: float = 0.1,
    ) -> None:
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PRESETS.keys())}")
            preset_config = PRESETS[preset]

            if ema_short_decay == 0.9:
                ema_short_decay = float(preset_config.get("ema_short_decay", ema_short_decay))
            if ema_long_decay == 0.99:
                ema_long_decay = float(preset_config.get("ema_long_decay", ema_long_decay))
            if gate_tier_max == 0:
                gate_tier_max = int(preset_config.get("gate_tier_max", gate_tier_max))
            if gate_min_scale == 0.2:
                gate_min_scale = float(preset_config.get("gate_min_scale", gate_min_scale))
            if gate_warmup_steps == 200:
                gate_warmup_steps = int(preset_config.get("gate_warmup_steps", gate_warmup_steps))
            if gate_update_interval == 10:
                gate_update_interval = int(
                    preset_config.get("gate_update_interval", gate_update_interval)
                )
            if quant4_interval == 1:
                quant4_interval = int(preset_config.get("quant4_interval", quant4_interval))
            if tier_config is None:
                tier_dict = preset_config.get("tier_config", {})
                if isinstance(tier_dict, dict):
                    tier_config = TierConfig(**tier_dict)

        if lr <= 0:
            raise ValueError("lr must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= ema_short_decay < 1.0:
            raise ValueError("ema_short_decay must be in [0, 1)")
        if not 0.0 <= ema_long_decay < 1.0:
            raise ValueError("ema_long_decay must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if quant4_interval < 1:
            raise ValueError("quant4_interval must be >= 1")
        if quant4_block_size < 2 or quant4_block_size % 2 != 0:
            raise ValueError("quant4_block_size must be an even integer >= 2")
        if gate_tier_max < 0 or gate_tier_max > 2:
            raise ValueError("gate_tier_max must be in [0, 2]")
        if not 0.0 < gate_min_scale <= 1.0:
            raise ValueError("gate_min_scale must be in (0, 1]")
        if gate_warmup_steps < 0:
            raise ValueError("gate_warmup_steps must be >= 0")
        if gate_update_interval < 1:
            raise ValueError("gate_update_interval must be >= 1")
        if not 0.0 <= min_lr_scale <= 1.0:
            raise ValueError("min_lr_scale must be in [0, 1]")
        if tiered_state and quant4:
            raise ValueError("tiered_state is incompatible with quant4")

        defaults = {"lr": lr, "weight_decay": weight_decay}
        super().__init__(params, defaults)

        self._tier_config = tier_config or TierConfig()
        if self._tier_config.ema_mismatch_weight < 0.0:
            raise ValueError("ema_mismatch_weight must be non-negative")
        if self._tier_config.min_tier_steps is not None and self._tier_config.min_tier_steps < 0:
            raise ValueError("min_tier_steps must be >= 0")
        if self._tier_config.signal_ema_decay is not None:
            if not 0.0 <= self._tier_config.signal_ema_decay < 1.0:
                raise ValueError("signal_ema_decay must be in [0, 1)")
        if self._tier_config.signal_margin < 0.0:
            raise ValueError("signal_margin must be >= 0")
        if (
            self._tier_config.force_tier2_warmup_steps is not None
            and self._tier_config.force_tier2_warmup_steps < 0
        ):
            raise ValueError("force_tier2_warmup_steps must be >= 0")
        if self._tier_config.full_sync_interval is not None:
            if self._tier_config.full_sync_interval < 1:
                raise ValueError("full_sync_interval must be >= 1")
        if self._tier_config.full_sync_tier < 0 or self._tier_config.full_sync_tier > 2:
            raise ValueError("full_sync_tier must be in [0, 2]")
        self._tier_lr_scales = (
            (1.0, 1.0, 1.0) if tier_lr_scales is None else tuple(float(x) for x in tier_lr_scales)
        )
        if len(self._tier_lr_scales) != 3:
            raise ValueError("tier_lr_scales must be a tuple of three floats")
        if any(scale < 0.0 for scale in self._tier_lr_scales):
            raise ValueError("tier_lr_scales must be non-negative")
        self._gate_tier_max = int(gate_tier_max)
        self._gate_min_scale = float(gate_min_scale)
        self._gate_warmup_steps = int(gate_warmup_steps)
        self._gate_update_interval = int(gate_update_interval)
        self._quant4 = quant4
        self._quant4_interval = int(quant4_interval)
        self._quant4_block_size = int(quant4_block_size)
        self._log_every = log_every
        self._logger = get_logger("cosmic")
        self._step = 0
        self._tiered_state = bool(tiered_state)
        self._use_fused_quant4 = (
            use_fused and quant4 and self._quant4_interval == 1 and not self._tiered_state
        )
        self._use_fused = use_fused and not quant4 and not self._tiered_state
        self._use_fused_tiered = use_fused and self._tiered_state
        self._min_lr_scale = float(min_lr_scale)

        interval = max(1, self._tier_config.reassignment_interval)
        self._tier_confirmations = 2
        self._tier_warmup_steps = (
            int(self._tier_config.warmup_steps)
            if self._tier_config.warmup_steps is not None
            else interval * 2
        )

        self._tier_budget_bytes = self._tier_config.budget_bytes
        if self._tier_config.tier2_budget_ratio is not None:
            if self._tier_config.tier2_budget_ratio < 0.0:
                raise ValueError("tier2_budget_ratio must be >= 0")
            total_bytes = 0
            for group in self.param_groups:
                for param in group["params"]:
                    total_bytes += param.numel() * param.element_size()
            self._tier_budget_bytes = int(total_bytes * self._tier_config.tier2_budget_ratio)

        self._param_name_map: dict[int, str] = {}
        if param_name_map:
            for key, name in param_name_map.items():
                if isinstance(key, torch.Tensor):
                    self._param_name_map[id(key)] = str(name)
                else:
                    self._param_name_map[int(key)] = str(name)
        for group in self.param_groups:
            names = group.get("param_names")
            if isinstance(names, list) and len(names) == len(group["params"]):
                for param, name in zip(group["params"], names, strict=False):
                    self._param_name_map[id(param)] = str(name)

        self._force_tier2_layers = tuple(force_tier2_layers or ())
        self._forced_tier2_ids: set[int] = set()
        if self._force_tier2_layers:
            for group in self.param_groups:
                for param in group["params"]:
                    name = self._param_name_map.get(id(param))
                    if name and any(token in name for token in self._force_tier2_layers):
                        self._forced_tier2_ids.add(id(param))
        self._force_tier2_warmup_steps = (
            int(self._tier_config.force_tier2_warmup_steps)
            if self._tier_config.force_tier2_warmup_steps is not None
            else None
        )

        self._ema_short_decay = float(ema_short_decay)
        self._ema_long_decay = float(ema_long_decay)
        self._ema_short_alpha = 1.0 - self._ema_short_decay
        self._ema_long_alpha = 1.0 - self._ema_long_decay
        self._beta2 = float(beta2)
        self._eps = float(eps)
        self._tier_blend = (0.0, 0.5, 1.0)

        self._gate_signal_decay = 0.9
        self._gate_signal_delta_threshold = 0.5
        self._gate_variance_threshold = 4.0
        self._gate_loss_spike_threshold = 0.25
        self._gate_eps = 1e-12
        self._gate_signal_ema: dict[int, float] = {}
        self._gate_prev_loss: float | None = None
        self._gate_events = 0
        self._gate_skips = 0
        self._gate_scale_raw_cache: dict[int, float] = {}

        self._sync_enabled = False
        self._sync_hook = None

        self._metrics_enabled = True
        self._tier_history: list[dict[str, int]] = []
        self._gate_scale_history: list[float] = []
        self._step_times: list[float] = []
        self._quant_bytes_saved: int = 0

    def get_metrics(self) -> dict[str, object]:
        """Get observability metrics for debugging and analysis.

        Returns dict with:
        - tier_occupancy: Current % params in each tier
        - tier_history: List of tier counts over time
        - gate_scale_stats: Statistics on gate scaling
        - quant_bytes_saved: Bytes saved by quantization
        - step_times: Step timing breakdown
        """
        tier_counts = [0, 0, 0]
        total_params = 0
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state.get(param, {})
                tier = int(state.get("tier", 0))
                tier_counts[tier] += 1
                total_params += 1

        tier_occupancy = {
            "tier0_fraction": tier_counts[0] / total_params if total_params > 0 else 0,
            "tier1_fraction": tier_counts[1] / total_params if total_params > 0 else 0,
            "tier2_fraction": tier_counts[2] / total_params if total_params > 0 else 0,
            "tier0_count": tier_counts[0],
            "tier1_count": tier_counts[1],
            "tier2_count": tier_counts[2],
            "total_params": total_params,
        }

        gate_stats: dict[str, float] = {}
        if self._gate_scale_history:
            import statistics

            gate_stats = {
                "gate_scale_mean": statistics.mean(self._gate_scale_history),
                "gate_scale_min": min(self._gate_scale_history),
                "gate_scale_max": max(self._gate_scale_history),
                "gate_scale_median": statistics.median(self._gate_scale_history),
                "gate_events": self._gate_events,
                "gate_skips": self._gate_skips,
            }
            if len(self._gate_scale_history) > 1:
                gate_stats["gate_scale_std"] = statistics.stdev(self._gate_scale_history)

        quant_stats = {
            "quant4_enabled": self._quant4,
            "quant_bytes_saved": self._quant_bytes_saved,
        }
        if self._quant4:
            total_state_bytes = 0
            quant_state_bytes = 0
            for group in self.param_groups:
                for param in group["params"]:
                    state = self.state.get(param, {})
                    for key in ["ema_short", "ema_long"]:
                        q = state.get(f"{key}_q")
                        scale = state.get(f"{key}_scale")
                        if isinstance(q, torch.Tensor):
                            quant_state_bytes += q.numel() * q.element_size()
                        if isinstance(scale, torch.Tensor):
                            quant_state_bytes += scale.numel() * scale.element_size()
                        elif key in state and torch.is_tensor(state[key]):
                            total_state_bytes += state[key].numel() * state[key].element_size()
            quant_stats["quant_state_bytes"] = quant_state_bytes
            quant_stats["fp_state_bytes"] = total_state_bytes

        return {
            "step": self._step,
            "tier_occupancy": tier_occupancy,
            "tier_history": self._tier_history[-100:],
            "gate_stats": gate_stats,
            "quant_stats": quant_stats,
        }

    def reset_metrics(self) -> None:
        """Reset accumulated metrics."""
        self._tier_history.clear()
        self._gate_scale_history.clear()
        self._step_times.clear()
        self._quant_bytes_saved = 0
        self._gate_events = 0
        self._gate_skips = 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1

        loss_value = self._loss_to_float(loss)

        with torch.no_grad():
            for group_index, group in enumerate(self.param_groups):
                params, grads = self._collect_group_tensors(group)
                if not params:
                    continue

                self._maybe_assign_tiers(params, grads)

                if self._tiered_state:
                    if self._use_fused_tiered:
                        self._step_fused_tiered(group_index, group, params, grads, loss_value)
                    else:
                        self._step_tiered(group_index, group, params, grads, loss_value)
                    continue
                if self._use_fused_quant4:
                    self._step_fused_quant4(group_index, group, params, grads, loss_value)
                    continue
                if self._use_fused:
                    self._step_fused(group_index, group, params, grads, loss_value)
                    continue

                momentum_grads, exp_avg_sq_list = self._apply_dual_ema(params, grads)
                tiers = [self.state[param].get("tier", 0) for param in params]
                gate_grads = [
                    grad
                    for grad, tier in zip(momentum_grads, tiers, strict=False)
                    if tier <= self._gate_tier_max
                ]
                gate_scale = self._get_gate_scale(group_index, gate_grads, loss_value)

                bias_correction2 = 1.0 - self._beta2**self._step if self._beta2 > 0 else 1.0
                eps = self._eps
                weight_decay = group["weight_decay"]
                use_adaptive = self._beta2 > 0

                for param, momentum, exp_avg_sq, tier in zip(
                    params, momentum_grads, exp_avg_sq_list, tiers, strict=False
                ):
                    tier_index = self._apply_full_sync_tier(max(0, min(2, int(tier))))
                    lr_scale = self._tier_lr_scales[tier_index]
                    effective_lr = group["lr"] * lr_scale
                    if tier_index <= self._gate_tier_max:
                        effective_lr *= gate_scale
                    effective_lr = self._apply_lr_floor(effective_lr, group["lr"])
                    if effective_lr <= 0.0:
                        continue

                    if weight_decay != 0:
                        param.add_(param, alpha=-effective_lr * weight_decay)

                    if use_adaptive:
                        denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
                        param.addcdiv_(momentum, denom, value=-effective_lr)
                    else:
                        param.add_(momentum, alpha=-effective_lr)

                for param in params:
                    state = self.state[param]
                    state["step"] = state.get("step", 0) + 1

        if self._log_every and self._step % self._log_every == 0:
            log_event(self._logger, "optimizer_step", step=self._step)

        if loss_value is not None:
            self._gate_prev_loss = loss_value

        return loss

    def _step_fused(
        self,
        group_index: int,
        group: dict[str, object],
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        loss_value: float | None,
    ) -> None:
        """Fused optimizer step using C++ kernel.

        This is the optimized path that does all operations in a single
        cache-efficient pass per parameter tensor.
        """
        tiers = [self.state[param].get("tier", 0) for param in params]
        gate_grads = [
            grad for grad, tier in zip(grads, tiers, strict=False) if tier <= self._gate_tier_max
        ]
        gate_scale = self._get_gate_scale(group_index, gate_grads, loss_value)

        ema_shorts: list[torch.Tensor] = []
        ema_longs: list[torch.Tensor] = []
        exp_avg_sqs: list[torch.Tensor] = []
        blends: list[float] = []
        effective_lrs: list[float] = []

        base_lr = group["lr"]
        weight_decay = group["weight_decay"]

        for param, _grad, tier in zip(params, grads, tiers, strict=False):
            state = self.state[param]

            ema_short = state.get("ema_short")
            if ema_short is None:
                ema_short = torch.zeros_like(param)
                state["ema_short"] = ema_short

            ema_long = state.get("ema_long")
            if ema_long is None:
                ema_long = torch.zeros_like(param)
                state["ema_long"] = ema_long

            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg_sq is None:
                exp_avg_sq = torch.zeros_like(param)
                state["exp_avg_sq"] = exp_avg_sq

            ema_shorts.append(ema_short)
            ema_longs.append(ema_long)
            exp_avg_sqs.append(exp_avg_sq)

            tier_index = self._apply_full_sync_tier(max(0, min(2, int(tier))))
            blends.append(self._tier_blend[tier_index])

            lr_scale = self._tier_lr_scales[tier_index]
            effective_lr = base_lr * lr_scale
            if tier_index <= self._gate_tier_max:
                effective_lr *= gate_scale
            effective_lr = self._apply_lr_floor(effective_lr, base_lr)
            effective_lrs.append(effective_lr)

            state["step"] = state.get("step", 0) + 1

        cosmic_fused_step(
            params,
            grads,
            ema_shorts,
            ema_longs,
            exp_avg_sqs,
            blends,
            effective_lrs,
            self._step,
            self._ema_short_decay,
            self._ema_long_decay,
            self._beta2,
            self._eps,
            weight_decay,
        )

    def _step_tiered(
        self,
        group_index: int,
        group: dict[str, object],
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        loss_value: float | None,
    ) -> None:
        """Tiered COSMIC step with reduced state outside Tier2."""
        tiers = [self.state[param].get("tier", 0) for param in params]
        gate_grads = [
            grad for grad, tier in zip(grads, tiers, strict=False) if tier <= self._gate_tier_max
        ]
        gate_scale = self._get_gate_scale(group_index, gate_grads, loss_value)

        base_lr = group["lr"]
        weight_decay = group["weight_decay"]
        beta2 = self._beta2
        use_adaptive = beta2 > 0
        bias_correction2 = 1.0 - beta2**self._step if beta2 > 0 else 1.0
        eps = self._eps
        short_decay = self._ema_short_decay
        long_decay = self._ema_long_decay
        short_alpha = self._ema_short_alpha
        long_alpha = self._ema_long_alpha

        for param, grad, tier in zip(params, grads, tiers, strict=False):
            state = self.state[param]

            tier_value = max(0, min(2, int(tier)))
            tier_index = self._apply_full_sync_tier(tier_value)
            lr_scale = self._tier_lr_scales[tier_index]
            effective_lr = base_lr * lr_scale
            if tier_index <= self._gate_tier_max:
                effective_lr *= gate_scale
            effective_lr = self._apply_lr_floor(effective_lr, base_lr)
            if effective_lr <= 0.0:
                state["step"] = state.get("step", 0) + 1
                continue

            if tier_value == 2:
                ema_short = state.get("ema_short")
                if ema_short is None:
                    ema_short = torch.zeros_like(param)
                    state["ema_short"] = ema_short
                ema_long = state.get("ema_long")
                if ema_long is None:
                    ema_long = torch.zeros_like(param)
                    state["ema_long"] = ema_long
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg_sq is None:
                    exp_avg_sq = torch.zeros_like(param)
                    state["exp_avg_sq"] = exp_avg_sq

                ema_short.mul_(short_decay).add_(grad, alpha=short_alpha)
                ema_long.mul_(long_decay).add_(grad, alpha=long_alpha)
                if use_adaptive:
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                blend = self._tier_blend[tier_value]
                if blend == 0.0:
                    momentum = ema_long
                elif blend == 1.0:
                    momentum = ema_short
                else:
                    momentum = ema_long.mul(1.0 - blend).add(ema_short, alpha=blend)

                if weight_decay != 0:
                    param.add_(param, alpha=-effective_lr * weight_decay)
                if use_adaptive:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
                    param.addcdiv_(momentum, denom, value=-effective_lr)
                else:
                    param.add_(momentum, alpha=-effective_lr)
                state.pop("cosmic_s", None)
            elif tier_value == 1:
                grad_scale = float(grad.abs().mean())
                scale = state.get("cosmic_s")
                if scale is None:
                    scale = grad_scale
                else:
                    scale = long_decay * float(scale) + (1.0 - long_decay) * grad_scale
                state["cosmic_s"] = scale
                for key in ("ema_short", "ema_long", "exp_avg_sq"):
                    state.pop(key, None)

                scaled_lr = effective_lr * scale
                if weight_decay != 0:
                    param.add_(param, alpha=-scaled_lr * weight_decay)
                param.add_(grad.sign(), alpha=-scaled_lr)
            else:
                for key in ("ema_short", "ema_long", "exp_avg_sq"):
                    state.pop(key, None)
                state.pop("cosmic_s", None)
                if weight_decay != 0:
                    param.add_(param, alpha=-effective_lr * weight_decay)
                param.add_(grad.sign(), alpha=-effective_lr)

            state["step"] = state.get("step", 0) + 1

    def _step_fused_tiered(
        self,
        group_index: int,
        group: dict[str, object],
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        loss_value: float | None,
    ) -> None:
        """Fused tiered COSMIC step with sign updates outside Tier2."""
        tiers = [self.state[param].get("tier", 0) for param in params]
        gate_grads = [
            grad for grad, tier in zip(grads, tiers, strict=False) if tier <= self._gate_tier_max
        ]
        gate_scale = self._get_gate_scale(group_index, gate_grads, loss_value)

        ema_shorts: list[torch.Tensor] = []
        ema_longs: list[torch.Tensor] = []
        exp_avg_sqs: list[torch.Tensor] = []
        blends: list[float] = []
        effective_lrs: list[float] = []
        cosmic_params: list[torch.Tensor] = []
        cosmic_grads: list[torch.Tensor] = []
        sign_params: list[torch.Tensor] = []
        sign_grads: list[torch.Tensor] = []
        sign_lrs: list[float] = []

        base_lr = group["lr"]
        weight_decay = group["weight_decay"]
        long_decay = self._ema_long_decay

        for param, grad, tier in zip(params, grads, tiers, strict=False):
            state = self.state[param]

            tier_value = max(0, min(2, int(tier)))
            tier_index = self._apply_full_sync_tier(tier_value)
            lr_scale = self._tier_lr_scales[tier_index]
            effective_lr = base_lr * lr_scale
            if tier_index <= self._gate_tier_max:
                effective_lr *= gate_scale
            effective_lr = self._apply_lr_floor(effective_lr, base_lr)
            if effective_lr <= 0.0:
                state["step"] = state.get("step", 0) + 1
                continue

            if tier_value == 2:
                ema_short = state.get("ema_short")
                if ema_short is None:
                    ema_short = torch.zeros_like(param)
                    state["ema_short"] = ema_short
                ema_long = state.get("ema_long")
                if ema_long is None:
                    ema_long = torch.zeros_like(param)
                    state["ema_long"] = ema_long
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg_sq is None:
                    exp_avg_sq = torch.zeros_like(param)
                    state["exp_avg_sq"] = exp_avg_sq

                cosmic_params.append(param)
                cosmic_grads.append(grad)
                ema_shorts.append(ema_short)
                ema_longs.append(ema_long)
                exp_avg_sqs.append(exp_avg_sq)
                blends.append(self._tier_blend[tier_value])
                effective_lrs.append(effective_lr)
                state.pop("cosmic_s", None)
            elif tier_value == 1:
                grad_scale = float(grad.abs().mean())
                scale = state.get("cosmic_s")
                if scale is None:
                    scale = grad_scale
                else:
                    scale = long_decay * float(scale) + (1.0 - long_decay) * grad_scale
                state["cosmic_s"] = scale
                for key in ("ema_short", "ema_long", "exp_avg_sq"):
                    state.pop(key, None)

                sign_params.append(param)
                sign_grads.append(grad)
                sign_lrs.append(effective_lr * scale)
            else:
                for key in ("ema_short", "ema_long", "exp_avg_sq"):
                    state.pop(key, None)
                state.pop("cosmic_s", None)

                sign_params.append(param)
                sign_grads.append(grad)
                sign_lrs.append(effective_lr)

            state["step"] = state.get("step", 0) + 1

        if cosmic_params:
            cosmic_fused_step(
                cosmic_params,
                cosmic_grads,
                ema_shorts,
                ema_longs,
                exp_avg_sqs,
                blends,
                effective_lrs,
                self._step,
                self._ema_short_decay,
                self._ema_long_decay,
                self._beta2,
                self._eps,
                weight_decay,
            )
        if sign_params:
            cosmic_fused_step_sign(
                sign_params,
                sign_grads,
                sign_lrs,
                weight_decay,
            )

    def _ensure_quant4_block_state(
        self,
        state: dict[str, object],
        key: str,
        param: torch.Tensor,
        grad: torch.Tensor,
        numel: int,
        num_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = state.get(f"{key}_q")
        scale = state.get(f"{key}_scale")
        if isinstance(q, torch.Tensor) and isinstance(scale, torch.Tensor):
            if q.numel() == (numel + 1) // 2 and scale.numel() == num_blocks:
                if not q.is_contiguous():
                    q = q.contiguous()
                    state[f"{key}_q"] = q
                if not scale.is_contiguous():
                    scale = scale.contiguous()
                    state[f"{key}_scale"] = scale
                return q, scale

        tensor = None
        fp = state.get(f"{key}_fp")
        if isinstance(fp, torch.Tensor):
            tensor = fp
        elif isinstance(state.get(key), torch.Tensor):
            tensor = state.get(key)
        elif (
            isinstance(q, torch.Tensor)
            and scale is not None
            and not isinstance(scale, torch.Tensor)
        ):
            try:
                numel_state = int(state.get(f"{key}_numel", numel))
                tensor = quant4_unpack(q, float(scale), numel_state).view_as(param)
            except Exception:
                tensor = None

        if isinstance(tensor, torch.Tensor):
            packed, scales, _numel, _block_size, _use_zp = quant4_pack_blocks(
                tensor, self._quant4_block_size, False
            )
            q = packed
            scale = scales
        else:
            q = torch.zeros((numel + 1) // 2, dtype=torch.uint8, device=grad.device)
            scale = torch.zeros(num_blocks, dtype=torch.float32, device=grad.device)

        state[f"{key}_q"] = q
        state[f"{key}_scale"] = scale
        state[f"{key}_block_size"] = int(self._quant4_block_size)
        state[f"{key}_use_zero_point"] = False
        for suffix in ("_numel", "_fp"):
            state.pop(f"{key}{suffix}", None)
        if isinstance(state.get(key), torch.Tensor):
            state.pop(key, None)

        return q, scale

    def _step_fused_quant4(
        self,
        group_index: int,
        group: dict[str, object],
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        loss_value: float | None,
    ) -> None:
        """Fused optimizer step using C++ kernel with quantized EMA state."""
        tiers = [self.state[param].get("tier", 0) for param in params]
        gate_grads = [
            grad for grad, tier in zip(grads, tiers, strict=False) if tier <= self._gate_tier_max
        ]
        gate_scale = self._get_gate_scale(group_index, gate_grads, loss_value)

        ema_short_qs: list[torch.Tensor] = []
        ema_short_scales: list[torch.Tensor] = []
        ema_long_qs: list[torch.Tensor] = []
        ema_long_scales: list[torch.Tensor] = []
        exp_avg_sqs: list[torch.Tensor] = []
        blends: list[float] = []
        effective_lrs: list[float] = []

        base_lr = group["lr"]
        weight_decay = group["weight_decay"]
        block_size = self._quant4_block_size

        for param, grad, tier in zip(params, grads, tiers, strict=False):
            state = self.state[param]
            numel = param.numel()
            num_blocks = (numel + block_size - 1) // block_size

            ema_short_q, ema_short_scale = self._ensure_quant4_block_state(
                state, "ema_short", param, grad, numel, num_blocks
            )
            ema_long_q, ema_long_scale = self._ensure_quant4_block_state(
                state, "ema_long", param, grad, numel, num_blocks
            )

            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg_sq is None:
                exp_avg_sq = torch.zeros_like(param)
                state["exp_avg_sq"] = exp_avg_sq

            ema_short_qs.append(ema_short_q)
            ema_short_scales.append(ema_short_scale)
            ema_long_qs.append(ema_long_q)
            ema_long_scales.append(ema_long_scale)
            exp_avg_sqs.append(exp_avg_sq)

            tier_index = self._apply_full_sync_tier(max(0, min(2, int(tier))))
            blends.append(self._tier_blend[tier_index])

            lr_scale = self._tier_lr_scales[tier_index]
            effective_lr = base_lr * lr_scale
            if tier_index <= self._gate_tier_max:
                effective_lr *= gate_scale
            effective_lr = self._apply_lr_floor(effective_lr, base_lr)
            effective_lrs.append(effective_lr)

            state["step"] = state.get("step", 0) + 1

        cosmic_fused_step_quant4(
            params,
            grads,
            ema_short_qs,
            ema_short_scales,
            ema_long_qs,
            ema_long_scales,
            exp_avg_sqs,
            blends,
            effective_lrs,
            self._step,
            self._ema_short_decay,
            self._ema_long_decay,
            self._beta2,
            self._eps,
            weight_decay,
            block_size,
        )

    def _collect_group_tensors(
        self, group: dict[str, object]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        params: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        for param in group["params"]:  # type: ignore
            if param.grad is None:
                continue
            self._validate_cpu(param, param.grad)
            params.append(param)
            grads.append(param.grad)
        return params, grads

    def _apply_dual_ema(
        self, params: list[torch.Tensor], grads: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Apply dual EMA momentum and return (momentum_grads, exp_avg_sq)."""
        momentum_grads: list[torch.Tensor] = []
        exp_avg_sq_list: list[torch.Tensor] = []
        short_decay = self._ema_short_decay
        long_decay = self._ema_long_decay
        short_alpha = self._ema_short_alpha
        long_alpha = self._ema_long_alpha
        beta2 = self._beta2
        tier_blend = self._tier_blend

        should_quantize = (
            self._quant4 and self._quant4_interval > 0 and self._step % self._quant4_interval == 0
        )
        ema_short_list: list[torch.Tensor] = []
        ema_long_list: list[torch.Tensor] = []
        state_list: list[dict[str, object]] = []

        for param, grad in zip(params, grads, strict=False):
            state = self.state[param]
            ema_short = self._load_ema_state(state, "ema_short", grad)
            ema_long = self._load_ema_state(state, "ema_long", grad)

            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg_sq is None:
                exp_avg_sq = torch.zeros_like(grad)
                state["exp_avg_sq"] = exp_avg_sq

            ema_short.mul_(short_decay).add_(grad, alpha=short_alpha)
            ema_long.mul_(long_decay).add_(grad, alpha=long_alpha)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            blend = tier_blend[state.get("tier", 0)]
            if blend == 0.0:
                momentum = ema_long
            elif blend == 1.0:
                momentum = ema_short
            else:
                momentum = ema_long.mul(1.0 - blend).add(ema_short, alpha=blend)

            momentum_grads.append(momentum)
            exp_avg_sq_list.append(exp_avg_sq)

            if self._quant4:
                if should_quantize:
                    ema_short_list.append(ema_short)
                    ema_long_list.append(ema_long)
                    state_list.append(state)
                else:
                    state["ema_short_fp"] = ema_short
                    state["ema_long_fp"] = ema_long
                    for suffix in ("_q", "_scale", "_numel"):
                        state.pop(f"ema_short{suffix}", None)
                        state.pop(f"ema_long{suffix}", None)
                    state.pop("ema_short", None)
                    state.pop("ema_long", None)
                    state.pop("ema_blend", None)
            else:
                state["ema_short"] = ema_short
                state["ema_long"] = ema_long
                state.pop("ema_blend", None)

        if self._quant4 and should_quantize and state_list:
            packed_short, scales_short, numels_short = quant4_pack_many(ema_short_list)
            packed_long, scales_long, numels_long = quant4_pack_many(ema_long_list)
            for idx, state in enumerate(state_list):
                state["ema_short_q"] = packed_short[idx]
                state["ema_short_scale"] = scales_short[idx]
                state["ema_short_numel"] = numels_short[idx]
                state["ema_long_q"] = packed_long[idx]
                state["ema_long_scale"] = scales_long[idx]
                state["ema_long_numel"] = numels_long[idx]
                state.pop("ema_short_fp", None)
                state.pop("ema_long_fp", None)
                state.pop("ema_short", None)
                state.pop("ema_long", None)
                state.pop("ema_blend", None)

        return momentum_grads, exp_avg_sq_list

    def _load_ema_state(
        self, state: dict[str, object], key: str, grad: torch.Tensor
    ) -> torch.Tensor:
        if not self._quant4:
            ema = state.get(key)
            if ema is None:
                ema = grad.detach().clone()
                state[key] = ema
            return ema

        ema_fp = state.get(f"{key}_fp")
        if isinstance(ema_fp, torch.Tensor):
            return ema_fp

        packed = state.get(f"{key}_q")
        if isinstance(packed, torch.Tensor):
            scale_obj = state.get(f"{key}_scale")
            numel = int(state.get(f"{key}_numel", grad.numel()))
            if isinstance(scale_obj, torch.Tensor):
                block_size = int(state.get(f"{key}_block_size", self._quant4_block_size))
                use_zero_point = bool(state.get(f"{key}_use_zero_point", False))
                ema = quant4_unpack_blocks(packed, scale_obj, numel, block_size, use_zero_point)
            else:
                scale = float(scale_obj) if scale_obj is not None else 1.0
                ema = quant4_unpack(packed, scale, numel)
            ema = ema.to(device=grad.device, dtype=grad.dtype)
            return ema.view_as(grad)

        return grad.detach().clone()

    def _store_quantized_state(
        self, state: dict[str, object], key: str, tensor: torch.Tensor
    ) -> None:
        packed, scale, numel = quant4_pack(tensor)
        state[f"{key}_q"] = packed
        state[f"{key}_scale"] = float(scale)
        state[f"{key}_numel"] = int(numel)

    def _store_quantized_state_blocks(
        self, state: dict[str, object], key: str, tensor: torch.Tensor
    ) -> None:
        packed, scales, numel, block_size, use_zero_point = quant4_pack_blocks(
            tensor, self._quant4_block_size, False
        )
        state[f"{key}_q"] = packed
        state[f"{key}_scale"] = scales
        state[f"{key}_numel"] = int(numel)
        state[f"{key}_block_size"] = int(block_size)
        state[f"{key}_use_zero_point"] = bool(use_zero_point)

    def _export_ema_state(
        self, state: dict[str, object], key: str, param: torch.Tensor
    ) -> torch.Tensor | None:
        ema_fp = state.get(f"{key}_fp")
        if isinstance(ema_fp, torch.Tensor):
            return ema_fp.detach().cpu().clone()
        ema = state.get(key)
        if isinstance(ema, torch.Tensor):
            return ema.detach().cpu().clone()

        packed = state.get(f"{key}_q")
        if isinstance(packed, torch.Tensor):
            scale_obj = state.get(f"{key}_scale")
            numel = int(state.get(f"{key}_numel", param.numel()))
            if isinstance(scale_obj, torch.Tensor):
                block_size = int(state.get(f"{key}_block_size", self._quant4_block_size))
                use_zero_point = bool(state.get(f"{key}_use_zero_point", False))
                unpacked = quant4_unpack_blocks(
                    packed, scale_obj, numel, block_size, use_zero_point
                )
            else:
                scale = float(scale_obj) if scale_obj is not None else 1.0
                unpacked = quant4_unpack(packed, scale, numel)
            return unpacked.view_as(param).detach().cpu()

        return None

    @staticmethod
    def _loss_to_float(loss) -> float | None:
        if loss is None:
            return None
        if isinstance(loss, torch.Tensor):
            return float(loss.detach().cpu().item())
        return float(loss)

    def _compute_grad_stats(self, grads: list[torch.Tensor]) -> tuple[float, float]:
        total_signal = 0.0
        total_var_ratio = 0.0
        count = len(grads) if grads else 1

        for grad in grads:
            mean_abs = float(grad.abs().mean())
            var = float(grad.var(unbiased=False))
            total_signal += mean_abs
            total_var_ratio += var / (mean_abs * mean_abs + self._gate_eps)

        return total_signal / count, total_var_ratio / count

    def _apply_gate_ramp(self, gate_scale_raw: float) -> tuple[float, float]:
        if self._gate_warmup_steps <= 0:
            return gate_scale_raw, 1.0
        if self._step <= self._gate_warmup_steps:
            return 1.0, 0.0
        ramp = min(
            1.0,
            (self._step - self._gate_warmup_steps) / float(self._gate_warmup_steps),
        )
        gate_scale = 1.0 - ramp * (1.0 - gate_scale_raw)
        return gate_scale, ramp

    def _gate_group_update(
        self, group_index: int, grads: list[torch.Tensor], loss_value: float | None
    ) -> tuple[float, dict[str, float]]:
        avg_signal, avg_var_ratio = self._compute_grad_stats(grads)

        ema = self._gate_signal_ema.get(group_index)
        if ema is None:
            ema = avg_signal
        else:
            ema = self._gate_signal_decay * ema + (1.0 - self._gate_signal_decay) * avg_signal
        self._gate_signal_ema[group_index] = ema

        signal_delta = abs(avg_signal - ema) / (ema + self._gate_eps)
        signal_lo = float(self._gate_signal_delta_threshold)
        variance_lo = float(self._gate_variance_threshold)
        signal_hi = signal_lo * 2.0
        variance_hi = variance_lo * 2.0
        min_scale = float(self._gate_min_scale)

        def linear_scale(value: float, lo: float, hi: float, floor: float) -> float:
            if value <= lo:
                return 1.0
            if hi <= lo or value >= hi:
                return floor
            t = (value - lo) / (hi - lo)
            return 1.0 + t * (floor - 1.0)

        signal_scale = linear_scale(signal_delta, signal_lo, signal_hi, min_scale)
        variance_scale = linear_scale(avg_var_ratio, variance_lo, variance_hi, min_scale)
        loss_scale = 1.0

        triggers: dict[str, float] = {}
        if signal_delta > signal_lo:
            triggers["signal_delta"] = signal_delta
        if avg_var_ratio > variance_lo:
            triggers["variance_ratio"] = avg_var_ratio
        if loss_value is not None and self._gate_prev_loss is not None:
            loss_ratio = loss_value / (self._gate_prev_loss + self._gate_eps)
            if loss_ratio > 1.0 + self._gate_loss_spike_threshold:
                triggers["loss_spike_ratio"] = loss_ratio
                loss_scale = min_scale

        gate_scale_raw = min(signal_scale, variance_scale, loss_scale)

        info: dict[str, float] = {
            "signal_delta": signal_delta,
            "variance_ratio": avg_var_ratio,
            "signal_delta_threshold": signal_lo,
            "signal_delta_threshold_hi": signal_hi,
            "variance_threshold": variance_lo,
            "variance_threshold_hi": variance_hi,
            "loss_spike_threshold": float(self._gate_loss_spike_threshold),
            "gate_min_scale": min_scale,
            "gate_warmup_steps": float(self._gate_warmup_steps),
            "signal_scale": signal_scale,
            "variance_scale": variance_scale,
            "loss_scale": loss_scale,
            "gate_scale_raw": gate_scale_raw,
        }
        if loss_value is not None:
            info["loss"] = loss_value
        if self._gate_prev_loss is not None:
            info["prev_loss"] = self._gate_prev_loss
        info.update(triggers)

        return gate_scale_raw, info

    def _get_gate_scale(
        self, group_index: int, grads: list[torch.Tensor], loss_value: float | None
    ) -> float:
        if not grads:
            return 1.0

        update_now = (
            self._gate_update_interval <= 1
            or self._step % self._gate_update_interval == 0
            or group_index not in self._gate_scale_raw_cache
        )

        gate_info: dict[str, float] = {}
        if update_now:
            gate_scale_raw, gate_info = self._gate_group_update(group_index, grads, loss_value)
            self._gate_scale_raw_cache[group_index] = gate_scale_raw
        else:
            gate_scale_raw = self._gate_scale_raw_cache.get(group_index, 1.0)

        gate_scale, ramp = self._apply_gate_ramp(gate_scale_raw)

        if update_now:
            if self._gate_warmup_steps > 0:
                if self._step <= self._gate_warmup_steps:
                    gate_info["gate_warmup"] = 1.0
                else:
                    gate_info["gate_ramp"] = ramp
            if gate_scale <= 0.0:
                self._gate_skips += 1
            if gate_scale < 1.0:
                self._gate_events += 1
                log_event(
                    self._logger,
                    "update_gated",
                    step=self._step,
                    group=group_index,
                    gate_scale=gate_scale,
                    gate_events=self._gate_events,
                    gate_skips=self._gate_skips,
                    **gate_info,
                )

        if self._metrics_enabled:
            self._gate_scale_history.append(gate_scale)

        return gate_scale

    def _apply_full_sync_tier(self, tier_index: int) -> int:
        interval = self._tier_config.full_sync_interval
        if interval and interval > 0 and self._step % interval == 0:
            return max(tier_index, int(self._tier_config.full_sync_tier))
        return tier_index

    def _apply_lr_floor(self, effective_lr: float, base_lr: float) -> float:
        if self._min_lr_scale > 0.0:
            return max(effective_lr, base_lr * self._min_lr_scale)
        return effective_lr

    @staticmethod
    def _validate_cpu(param: torch.Tensor, grad: torch.Tensor) -> None:
        if param.is_cuda or grad.is_cuda:
            raise RuntimeError("Cosmic optimizer is CPU-only; CUDA tensors are not supported")

    def _maybe_assign_tiers(self, params: list[torch.Tensor], grads: list[torch.Tensor]) -> None:
        interval = self._tier_config.reassignment_interval
        if interval <= 0 or self._step % interval != 0:
            return
        raw_signals = [tier_signal(grad) for grad in grads]
        signals: list[float] = []
        signal_decay = self._tier_config.signal_ema_decay
        if signal_decay is not None:
            for param, raw_signal in zip(params, raw_signals, strict=False):
                state = self.state[param]
                ema = state.get("tier_signal_ema", raw_signal)
                ema = signal_decay * float(ema) + (1.0 - signal_decay) * float(raw_signal)
                state["tier_signal_ema"] = ema
                signals.append(ema)
        else:
            signals = raw_signals
        if self._tier_config.ema_mismatch_weight > 0.0:
            self._augment_signals_with_ema_mismatch(params, signals)
        candidates = self._assign_tier_candidates(params, signals)

        promotions = 0
        demotions = 0
        tier_counts = [0, 0, 0]
        force_active = (
            self._force_tier2_warmup_steps is None or self._step <= self._force_tier2_warmup_steps
        )
        min_tier_steps = self._tier_config.min_tier_steps or 0
        signal_margin = float(self._tier_config.signal_margin)

        for param, signal, candidate in zip(params, signals, candidates, strict=False):
            state = self.state[param]
            if force_active and id(param) in self._forced_tier2_ids:
                current = state.get("tier", 0)
                if current != 2:
                    promotions += 1 if 2 > current else 0
                    demotions += 1 if 2 < current else 0
                state["tier"] = 2
                state["tier_last_change_step"] = self._step
                state["last_tier_signal"] = signal
                state.pop("tier_pending", None)
                state["tier_pending_count"] = 0
                tier_counts[2] += 1
                continue
            if "tier" not in state:
                state["tier"] = candidate
                state["tier_last_change_step"] = self._step
                state["last_tier_signal"] = signal
                state.pop("tier_pending", None)
                state["tier_pending_count"] = 0
                if candidate > 0:
                    promotions += 1
                tier_counts[state["tier"]] += 1
                continue

            current = state.get("tier", 0)
            last_signal = state.get("last_tier_signal", signal)
            state["last_tier_signal"] = signal

            if candidate == current:
                state.pop("tier_pending", None)
                state["tier_pending_count"] = 0
            else:
                if min_tier_steps > 0:
                    last_change = int(state.get("tier_last_change_step", 0))
                    if self._step - last_change < min_tier_steps:
                        candidate = current

                if self._step < self._tier_warmup_steps and candidate < current:
                    candidate = current
                else:
                    trend = signal - last_signal
                    if candidate > current and trend < 0:
                        candidate = current
                    elif candidate < current and trend > 0:
                        candidate = current

                if candidate != current and signal_margin > 0.0:
                    baseline = max(abs(last_signal), self._gate_eps)
                    upper = last_signal + signal_margin * baseline
                    lower = last_signal - signal_margin * baseline
                    if candidate > current and signal < upper:
                        candidate = current
                    elif candidate < current and signal > lower:
                        candidate = current

                if candidate != current:
                    pending = state.get("tier_pending")
                    if pending == candidate:
                        count = state.get("tier_pending_count", 0) + 1
                    else:
                        count = 1
                    state["tier_pending"] = candidate
                    state["tier_pending_count"] = count

                    if count >= self._tier_confirmations:
                        state["tier"] = candidate
                        state["tier_last_change_step"] = self._step
                        state.pop("tier_pending", None)
                        state["tier_pending_count"] = 0
                        if candidate > current:
                            promotions += 1
                        else:
                            demotions += 1
                else:
                    state.pop("tier_pending", None)
                    state["tier_pending_count"] = 0

            tier_counts[state.get("tier", 0)] += 1

        if promotions or demotions:
            log_event(
                self._logger,
                "tier_transition",
                step=self._step,
                promotions=promotions,
                demotions=demotions,
                tier0=tier_counts[0],
                tier1=tier_counts[1],
                tier2=tier_counts[2],
            )

    def _assign_tier_candidates(
        self, params: list[torch.Tensor], signals: list[float]
    ) -> list[int]:
        indexed = list(enumerate(zip(params, signals, strict=False)))
        indexed.sort(key=lambda item: (-item[1][1], item[0]))

        tier2_fraction = self._tier_config.tier2_fraction
        tier1_fraction = self._tier_config.tier1_fraction
        if self._step < self._tier_warmup_steps:
            if self._tier_config.tier2_fraction_warmup is not None:
                tier2_fraction = self._tier_config.tier2_fraction_warmup
            if self._tier_config.tier1_fraction_warmup is not None:
                tier1_fraction = self._tier_config.tier1_fraction_warmup

        tier2_target = max(0, int(round(len(params) * tier2_fraction)))
        tier1_target = max(0, int(round(len(params) * tier1_fraction)))

        tier_list = [0] * len(params)
        budget = self._tier_budget_bytes

        used_budget = 0
        tier2_assigned = 0
        tier1_assigned = 0

        forced_count = 0
        for idx, param in enumerate(params):
            if id(param) in self._forced_tier2_ids:
                tier_list[idx] = 2
                forced_count += 1
                if budget is not None:
                    used_budget += param.numel() * param.element_size()

        tier2_assigned = forced_count
        if forced_count > tier2_target:
            tier2_target = forced_count

        for param_index, (param, _signal) in indexed:
            if tier_list[param_index] == 2:
                continue
            param_bytes = param.numel() * param.element_size()
            if budget is not None and used_budget + param_bytes > budget:
                tier_list[param_index] = 0
                continue

            if tier2_assigned < tier2_target:
                tier_list[param_index] = 2
                tier2_assigned += 1
                used_budget += param_bytes
                continue

            if tier1_assigned < tier1_target:
                tier_list[param_index] = 1
                tier1_assigned += 1
                used_budget += param_bytes
                continue

            tier_list[param_index] = 0

        return tier_list

    def _get_ema_for_signal(
        self, state: dict[str, object], key: str, param: torch.Tensor
    ) -> torch.Tensor | None:
        ema_fp = state.get(f"{key}_fp")
        if isinstance(ema_fp, torch.Tensor):
            return ema_fp
        ema = state.get(key)
        if isinstance(ema, torch.Tensor):
            return ema
        packed = state.get(f"{key}_q")
        if isinstance(packed, torch.Tensor):
            numel = int(state.get(f"{key}_numel", param.numel()))
            scale_obj = state.get(f"{key}_scale")
            if isinstance(scale_obj, torch.Tensor):
                block_size = int(state.get(f"{key}_block_size", self._quant4_block_size))
                use_zero_point = bool(state.get(f"{key}_use_zero_point", False))
                return quant4_unpack_blocks(
                    packed, scale_obj, numel, block_size, use_zero_point
                ).view_as(param)
            scale = float(scale_obj) if scale_obj is not None else 1.0
            return quant4_unpack(packed, scale, numel).view_as(param)
        return None

    def _augment_signals_with_ema_mismatch(
        self, params: list[torch.Tensor], signals: list[float]
    ) -> None:
        weight = float(self._tier_config.ema_mismatch_weight)
        if weight <= 0.0:
            return
        for idx, param in enumerate(params):
            state = self.state.get(param, {})
            ema_short = self._get_ema_for_signal(state, "ema_short", param)
            ema_long = self._get_ema_for_signal(state, "ema_long", param)
            if ema_short is None or ema_long is None:
                continue
            mismatch = float((ema_short - ema_long).abs().mean().item())
            signals[idx] += weight * mismatch

    def export_sync_metadata(self) -> dict[str, object]:
        tiers: list[int] = []
        ema_short: list[torch.Tensor | None] = []
        ema_long: list[torch.Tensor | None] = []

        for group in self.param_groups:
            for param in group["params"]:
                state = self.state.get(param, {})
                tiers.append(int(state.get("tier", 0)))
                ema_short.append(self._export_ema_state(state, "ema_short", param))
                ema_long.append(self._export_ema_state(state, "ema_long", param))
        return {
            "step": int(self._step),
            "tiers": tiers,
            "ema_short": ema_short,
            "ema_long": ema_long,
            "ema_short_decay": float(self._ema_short_decay),
            "ema_long_decay": float(self._ema_long_decay),
            "quant4": bool(self._quant4),
        }

    def import_sync_metadata(self, metadata: dict[str, object]) -> None:
        self._step = int(metadata.get("step", self._step))  # type: ignore
        tiers_obj = metadata.get("tiers", [])
        ema_short_obj = metadata.get("ema_short", [])
        ema_long_obj = metadata.get("ema_long", [])

        tiers: list[int] = list(tiers_obj) if isinstance(tiers_obj, list) else []
        ema_short_list: list[torch.Tensor | None] = (
            list(ema_short_obj) if isinstance(ema_short_obj, list) else []
        )
        ema_long_list: list[torch.Tensor | None] = (
            list(ema_long_obj) if isinstance(ema_long_obj, list) else []
        )

        param_idx = 0
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                if param_idx < len(tiers):
                    state["tier"] = int(tiers[param_idx])

                if param_idx < len(ema_short_list):
                    ema_short = ema_short_list[param_idx]
                    if isinstance(ema_short, torch.Tensor):
                        ema_short = ema_short.to(param.device, dtype=param.dtype)
                        if self._quant4:
                            if self._use_fused_quant4:
                                self._store_quantized_state_blocks(state, "ema_short", ema_short)
                            else:
                                self._store_quantized_state(state, "ema_short", ema_short)
                        else:
                            state["ema_short"] = ema_short

                if param_idx < len(ema_long_list):
                    ema_long = ema_long_list[param_idx]
                    if isinstance(ema_long, torch.Tensor):
                        ema_long = ema_long.to(param.device, dtype=param.dtype)
                        if self._quant4:
                            if self._use_fused_quant4:
                                self._store_quantized_state_blocks(state, "ema_long", ema_long)
                            else:
                                self._store_quantized_state(state, "ema_long", ema_long)
                        else:
                            state["ema_long"] = ema_long

                if self._quant4:
                    state.pop("ema_short", None)
                    state.pop("ema_long", None)
                    state.pop("ema_blend", None)
                    state.pop("ema_short_fp", None)
                    state.pop("ema_long_fp", None)

                param_idx += 1

    def register_sync_hook(self, hook) -> None:
        self._sync_hook = hook

    def enable_sync(self, enabled: bool = True) -> None:
        self._sync_enabled = enabled

    def _maybe_sync(self, metadata: dict[str, object]) -> dict[str, object]:
        if not self._sync_enabled or self._sync_hook is None:
            return metadata
        return self._sync_hook(metadata)
