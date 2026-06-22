"""Model-contract tests for every registered model (BUILD_PLAN §3.1/§3.2).

For each of the six registered models this checks the frozen interface the
trainer / evaluator / inference engine depend on:

* ``build_model`` constructs from a tiny ``ModelConfig``;
* ``forward(SAMPLE)`` returns a ``ModelOutput`` whose reconstruction matches the
  target shape ``[B, C, H, W]``;
* ``loss`` returns a mapping containing ``"total"``;
* ``total.backward()`` produces finite gradients;
* ``predict`` runs under ``no_grad``.

Plus model-specific contracts: the GAN exposes disjoint generator/discriminator
parameter groups + a discriminator loss; the uncertainty model populates the
variance head.
"""

from __future__ import annotations

import pytest
import torch

from cloudremoval.config import Config
from cloudremoval.models.registry import build_model, list_models

_REGISTERED = list_models()


def _tiny_model_cfg(name: str) -> Config:
    """A small CPU config for ``name`` (few channels/steps to keep tests fast)."""
    cfg = Config()
    cfg.model.name = name
    cfg.model.base_channels = 16
    cfg.model.depth = 2
    cfg.model.sample_steps = 2  # diffusion DDIM steps at predict time
    cfg.model.timesteps = 8
    return cfg


def _batch(b: int = 2, c: int = 3, h: int = 32, w: int = 32) -> dict:
    """A batched SAMPLE dict (§2) with all keys populated (no temporal refs)."""
    g = torch.Generator().manual_seed(0)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=g, dtype=torch.float32)

    cloud_mask = (rand(b, 1, h, w) > 0.6).float()
    return {
        "optical_cloudy": rand(b, c, h, w),
        "optical_clear": rand(b, c, h, w),
        "cloud_mask": cloud_mask,
        "shadow_mask": (rand(b, 1, h, w) > 0.85).float(),
        "sar": rand(b, 2, h, w),
        "dem": rand(b, 1, h, w),
        "temporal_refs": torch.zeros(b, 0, c, h, w),
        "meta": [{"norm": "zscore", "has_target": True} for _ in range(b)],
    }


def test_six_models_registered() -> None:
    """The registry holds exactly the six contracted models (short names)."""
    assert set(_REGISTERED) == {
        "unet",
        "dsen2cr",
        "spagan",
        "restormer",
        "diffusion",
        "uncertainty",
    }


@pytest.mark.parametrize("name", _REGISTERED)
def test_model_forward_loss_backward_predict(name: str) -> None:
    """Each model satisfies the forward/loss/backward/predict contract."""
    from cloudremoval.models.base import ModelOutput

    cfg = _tiny_model_cfg(name)
    model = build_model(cfg)
    sample = _batch()

    # forward -> ModelOutput with matching reconstruction shape.
    output = model(sample)
    assert isinstance(output, ModelOutput)
    assert output.reconstruction.shape == sample["optical_clear"].shape
    assert torch.isfinite(output.reconstruction).all()

    # loss mapping must contain "total" (the back-propagated scalar).
    losses = model.loss(sample, output)
    assert "total" in losses
    total = losses["total"]
    assert total.requires_grad

    # backward -> finite gradients on at least some trainable params.
    model.zero_grad(set_to_none=True)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, f"{name}: no gradients flowed"
    assert all(torch.isfinite(g).all() for g in grads), f"{name}: non-finite grads"

    # predict runs without tracking gradients and returns the right shape.
    pred = model.predict(sample)
    assert pred.reconstruction.shape == sample["optical_clear"].shape
    assert not pred.reconstruction.requires_grad


def test_gan_disjoint_optimizer_hooks() -> None:
    """SPAGAN exposes disjoint generator/discriminator params + a disc loss."""
    model = build_model(_tiny_model_cfg("spagan"))
    assert hasattr(model, "generator_parameters")
    assert hasattr(model, "discriminator_parameters")
    assert hasattr(model, "discriminator_loss")

    gen_ids = {id(p) for p in model.generator_parameters()}
    disc_ids = {id(p) for p in model.discriminator_parameters()}
    assert gen_ids and disc_ids
    assert gen_ids.isdisjoint(disc_ids), "generator and discriminator share params"

    sample = _batch()
    output = model(sample)
    d_loss = model.discriminator_loss(sample, output)
    assert "total" in d_loss
    assert torch.isfinite(d_loss["total"]).all()


def test_uncertainty_variance_head_present() -> None:
    """The uncertainty model populates ``ModelOutput.uncertainty`` (non-negative)."""
    model = build_model(_tiny_model_cfg("uncertainty"))
    sample = _batch()
    output = model(sample)
    assert output.uncertainty is not None
    assert output.uncertainty.shape[0] == sample["optical_cloudy"].shape[0]
    assert output.uncertainty.shape[-2:] == sample["optical_cloudy"].shape[-2:]
    # Variance (exp of log-var) is finite and >= 0.
    assert torch.isfinite(output.uncertainty).all()
    assert bool((output.uncertainty >= 0).all())


def test_non_gan_models_have_no_discriminator_hooks() -> None:
    """Non-adversarial models do not advertise GAN hooks (``hasattr`` is False)."""
    model = build_model(_tiny_model_cfg("unet"))
    assert not hasattr(model, "discriminator_parameters")
    assert not hasattr(model, "discriminator_loss")
