"""Determinism helpers.

Every CLI entry point calls :func:`seed_everything` first (BUILD_PLAN §5) so the
synthetic CPU-smoke pipeline is reproducible. Pure ``numpy``/``torch``/``random``.
"""

from __future__ import annotations

import os
import random

__all__ = ["seed_everything", "set_seed"]


def seed_everything(seed: int = 1337, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch RNGs.

    Args:
        seed: The seed value applied to all RNGs.
        deterministic: If ``True``, also request deterministic cuDNN algorithms
            (slower, but bit-reproducible on GPU). No effect on CPU-only runs.

    Returns:
        The seed that was set (for logging/echoing).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:  # numpy is a core dep but guard anyway for ultra-minimal imports
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            torch.cuda.manual_seed_all(seed)
        if deterministic:  # pragma: no cover - GPU/cudnn path
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001
        pass

    return seed


# Alias matching the B0 export contract (``set_seed``); both names are public.
set_seed = seed_everything
