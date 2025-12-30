import torch

from cosmic.extension import tier_signal
from cosmic.optim import Cosmic, TierConfig


def test_tier_assignment_by_signal():
    params = [
        torch.nn.Parameter(torch.zeros(2)),
        torch.nn.Parameter(torch.zeros(2)),
        torch.nn.Parameter(torch.zeros(2)),
    ]
    grads = [
        torch.ones(2) * 1.0,
        torch.ones(2) * 0.1,
        torch.ones(2) * 0.01,
    ]

    optimizer = Cosmic(
        params,
        lr=1e-3,
        tier_config=TierConfig(tier2_fraction=0.34, tier1_fraction=0.34, reassignment_interval=1),
    )

    signals = [tier_signal(g) for g in grads]
    tier_list = optimizer._assign_tier_candidates(params, signals)

    assert tier_list == [2, 1, 0]
