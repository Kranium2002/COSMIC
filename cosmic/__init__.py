"""CPU-only PyTorch optimizer extension for research workflows."""

from cosmic._version import __version__
from cosmic.optim import Cosmic, TierConfig

__all__ = ["Cosmic", "TierConfig", "__version__"]
