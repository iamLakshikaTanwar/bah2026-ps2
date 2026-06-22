"""Synthetic cloud + shadow generation for paired training data (pure NumPy/torch).

This is the **core paired-data generator** for the LISS-IV pipeline: since matched
(cloudy, clear) LISS-IV pairs are scarce (``research/06`` §C.5, §E.5), we paste
physically-plausible clouds and cast shadows onto clear scenes to fabricate
supervised pairs. The implementation follows the *SatelliteCloudGenerator* idea
(``research/05`` §B.5) — controllable thin (alpha-blended, spectrally recoverable)
and thick (opaque) clouds plus geometry-offset cast shadows — but is written with
**only NumPy/torch** so it runs on the CPU-smoke stack. A self-contained
fractal **value-noise** (fBm) field stands in for the ``noise`` package, which is
intentionally *not* a dependency on the smoke path.

Public API
----------
:func:`add_synthetic_clouds`
    Functional one-shot: ``clear[C,H,W] -> (cloudy, cloud_mask, shadow_mask)``.
:class:`CloudSimulator`
    Configurable, deterministic generator with thickness/coverage controls and an
    optional ``sample`` of real cloud-alpha mattes (copy-paste, the lower
    synthetic->real gap path from ``research/05`` §B.5).

All masks are returned as ``[1, H, W]`` float arrays/tensors in ``{0, 1}`` (cloud)
and ``[0, 1]`` (the soft alpha is exposed separately). Determinism is controlled
by an explicit ``seed`` / ``numpy.random.Generator``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cloudremoval.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = [
    "CloudSimConfig",
    "CloudSimulator",
    "add_synthetic_clouds",
    "value_noise_2d",
    "fractal_noise_2d",
]

_log = get_logger(__name__)
_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Value-noise / fractal Brownian motion (pure NumPy; replaces the `noise` pkg)
# --------------------------------------------------------------------------- #
def value_noise_2d(
    shape: tuple[int, int],
    cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate smooth 2-D value noise by bilinear upsampling of a random lattice.

    A coarse ``(cells+1, cells+1)`` lattice of uniform random values is smoothly
    (smoothstep) interpolated up to ``shape``. This is the building block for the
    fractal cloud field and needs no external noise library.

    Args:
        shape: Output ``(H, W)``.
        cells: Number of lattice cells per axis (frequency); higher = finer.
        rng: NumPy random generator (determinism).

    Returns:
        ``[H, W]`` noise in ``[0, 1]`` (``float32``).
    """
    h, w = shape
    cells = max(1, int(cells))
    lattice = rng.random((cells + 1, cells + 1)).astype(np.float32)

    # Coordinates of each output pixel within the lattice.
    ys = np.linspace(0, cells, h, endpoint=True, dtype=np.float32)
    xs = np.linspace(0, cells, w, endpoint=True, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int64).clip(0, cells - 1)
    x0 = np.floor(xs).astype(np.int64).clip(0, cells - 1)
    ty = (ys - y0).reshape(h, 1)
    tx = (xs - x0).reshape(1, w)
    # Smoothstep for C1 continuity (no lattice artefacts).
    ty = ty * ty * (3 - 2 * ty)
    tx = tx * tx * (3 - 2 * tx)

    v00 = lattice[np.ix_(y0, x0)]
    v01 = lattice[np.ix_(y0, x0 + 1)]
    v10 = lattice[np.ix_(y0 + 1, x0)]
    v11 = lattice[np.ix_(y0 + 1, x0 + 1)]
    top = v00 * (1 - tx) + v01 * tx
    bot = v10 * (1 - tx) + v11 * tx
    return (top * (1 - ty) + bot * ty).astype(np.float32)


def fractal_noise_2d(
    shape: tuple[int, int],
    rng: np.random.Generator,
    octaves: int = 5,
    base_cells: int = 3,
    persistence: float = 0.55,
    lacunarity: float = 2.0,
) -> np.ndarray:
    """Fractal Brownian motion: sum of value-noise octaves (cloud texture).

    Args:
        shape: Output ``(H, W)``.
        rng: NumPy random generator.
        octaves: Number of noise layers summed.
        base_cells: Lattice cells of the lowest-frequency octave.
        persistence: Amplitude decay per octave (``< 1``).
        lacunarity: Frequency growth per octave (``> 1``).

    Returns:
        ``[H, W]`` fractal field normalised to ``[0, 1]`` (``float32``).
    """
    field = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    frequency = float(base_cells)
    total_amp = 0.0
    for _ in range(max(1, int(octaves))):
        field += amplitude * value_noise_2d(shape, int(round(frequency)), rng)
        total_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    field /= max(total_amp, _EPS)
    # Normalise to full [0, 1] range for predictable thresholding.
    lo, hi = float(field.min()), float(field.max())
    return ((field - lo) / (hi - lo + _EPS)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class CloudSimConfig:
    """Configuration for :class:`CloudSimulator`.

    Attributes:
        cloud_mode: ``"thin"`` (haze, alpha-blended), ``"thick"`` (opaque), or
            ``"mixed"`` (random per sample).
        coverage: Target fraction of pixels covered by cloud (0-1); the field
            threshold is chosen to approximate this.
        coverage_jitter: Uniform +/- jitter applied to ``coverage`` per sample.
        thickness: Peak cloud opacity in ``[0, 1]`` for thick clouds (alpha at
            the densest core).
        thin_max_alpha: Maximum alpha for thin/haze clouds (kept < 1 so the
            surface stays partially visible, i.e. spectrally recoverable).
        cloud_brightness: Reflectance the cloud blends *towards* (white ~ high).
        octaves: fBm octaves for the cloud field.
        base_cells: Lowest-frequency lattice size (smaller = larger blobs).
        edge_softness: Width (in field units) of the soft cloud edge.
        shadow_strength: Multiplicative darkening at the shadow core (0-1; 1 = no
            shadow, lower = darker).
        shadow_offset_frac: Cast-shadow displacement as a fraction of image size
            at the reference sun elevation (scaled by ``1/tan(elevation)``).
        add_shadows: Whether to cast shadows at all.
    """

    cloud_mode: str = "mixed"
    coverage: float = 0.35
    coverage_jitter: float = 0.15
    thickness: float = 1.0
    thin_max_alpha: float = 0.6
    cloud_brightness: float = 0.92
    octaves: int = 5
    base_cells: int = 3
    edge_softness: float = 0.12
    shadow_strength: float = 0.45
    shadow_offset_frac: float = 0.06
    add_shadows: bool = True


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class CloudSimulator:
    """Deterministic synthetic cloud + shadow generator (SatelliteCloudGenerator-style).

    Generates a fractal cloud opacity field, splits it into thin (alpha-blend)
    and thick (opaque) regimes, blends the cloud over a clear image towards a
    bright cloud colour, and casts a geometry-offset shadow that darkens the
    surface outside the cloud. Real cloud-alpha mattes may be supplied to
    ``__call__`` for the copy-paste path (``research/05`` §B.5).

    Example::

        sim = CloudSimulator(CloudSimConfig(coverage=0.4))
        cloudy, cmask, smask = sim(clear, sun_azimuth=135.0, sun_elevation=45.0, seed=0)
    """

    def __init__(self, cfg: CloudSimConfig | None = None) -> None:
        """Store configuration.

        Args:
            cfg: Generator configuration; defaults to :class:`CloudSimConfig`.
        """
        self.cfg = cfg or CloudSimConfig()

    # ---- helpers ------------------------------------------------------- #
    def _resolve_mode(self, rng: np.random.Generator) -> str:
        mode = self.cfg.cloud_mode
        if mode == "mixed":
            return str(rng.choice(["thin", "thick"]))
        return mode

    def _opacity_field(
        self,
        hw: tuple[int, int],
        rng: np.random.Generator,
        coverage: float,
    ) -> np.ndarray:
        """Build a soft cloud-opacity field in ``[0, 1]`` covering ~``coverage``."""
        field = fractal_noise_2d(
            hw, rng, octaves=self.cfg.octaves, base_cells=self.cfg.base_cells
        )
        # Threshold so that ~coverage fraction of pixels are "clouded".
        coverage = float(np.clip(coverage, 0.0, 1.0))
        if coverage <= 0.0:
            return np.zeros(hw, dtype=np.float32)
        if coverage >= 1.0:
            thresh = 0.0
        else:
            thresh = float(np.quantile(field, 1.0 - coverage))
        soft = self.cfg.edge_softness + _EPS
        # Smooth ramp from 0 at (thresh) to 1 at (thresh + soft).
        alpha = np.clip((field - thresh) / soft, 0.0, 1.0)
        return (alpha * alpha * (3 - 2 * alpha)).astype(np.float32)  # smoothstep

    @staticmethod
    def _shadow_offset(
        hw: tuple[int, int],
        sun_azimuth_deg: float,
        sun_elevation_deg: float,
        base_frac: float,
    ) -> tuple[int, int]:
        """Pixel (dy, dx) shadow displacement from sun geometry.

        Shadows fall *away* from the sun; length grows as ``1/tan(elevation)``.
        Azimuth is measured clockwise from north (0 = north, 90 = east).
        """
        h, w = hw
        elev = max(float(sun_elevation_deg), 5.0)
        length = base_frac * (1.0 / np.tan(np.deg2rad(elev)))
        # Shadow direction = opposite the sun azimuth.
        shadow_az = np.deg2rad((sun_azimuth_deg + 180.0) % 360.0)
        dx = int(round(length * w * np.sin(shadow_az)))
        dy = int(round(-length * h * np.cos(shadow_az)))
        return dy, dx

    @staticmethod
    def _shift(field: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """Shift a 2-D field by ``(dy, dx)``, zero-filling exposed borders."""
        out = np.zeros_like(field)
        h, w = field.shape
        ys_src = slice(max(0, -dy), h - max(0, dy))
        xs_src = slice(max(0, -dx), w - max(0, dx))
        ys_dst = slice(max(0, dy), h - max(0, -dy))
        xs_dst = slice(max(0, dx), w - max(0, -dx))
        out[ys_dst, xs_dst] = field[ys_src, xs_src]
        return out

    # ---- main entry ---------------------------------------------------- #
    def __call__(
        self,
        clear: np.ndarray,
        sun_azimuth: float = 135.0,
        sun_elevation: float = 45.0,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        real_alpha: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Add synthetic clouds + cast shadow to a clear ``[C, H, W]`` image.

        Args:
            clear: ``[C, H, W]`` clear reflectance image (any normalization; the
                cloud blends towards ``cfg.cloud_brightness`` in the same space).
            sun_azimuth: Solar azimuth in degrees (clockwise from north).
            sun_elevation: Solar elevation in degrees (drives shadow length).
            seed: Optional seed for a fresh generator (ignored if ``rng`` given).
            rng: Optional explicit NumPy generator (takes precedence over ``seed``).
            real_alpha: Optional ``[H, W]`` or ``[1, H, W]`` real cloud-alpha matte
                in ``[0, 1]`` to use instead of synthetic fBm (copy-paste path).

        Returns:
            Tuple ``(cloudy[C,H,W], cloud_mask[1,H,W], shadow_mask[1,H,W])`` —
            ``cloudy`` clipped to ``[0, 1]``, masks as ``float32`` (cloud_mask in
            ``{0, 1}``, shadow_mask in ``{0, 1}``).
        """
        arr = np.asarray(clear, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        c, h, w = arr.shape
        if rng is None:
            rng = np.random.default_rng(seed)

        mode = self._resolve_mode(rng)
        coverage = float(
            np.clip(
                self.cfg.coverage + rng.uniform(-1.0, 1.0) * self.cfg.coverage_jitter,
                0.0,
                1.0,
            )
        )

        if real_alpha is not None:
            alpha = np.asarray(real_alpha, dtype=np.float32)
            if alpha.ndim == 3:
                alpha = alpha[0]
            alpha = np.clip(alpha, 0.0, 1.0)
        else:
            alpha = self._opacity_field((h, w), rng, coverage)

        # Thin clouds cap the alpha (surface stays partly visible); thick clouds
        # push the dense core opaque.
        if mode == "thin":
            alpha = alpha * self.cfg.thin_max_alpha
        else:  # thick
            alpha = np.clip(alpha * self.cfg.thickness, 0.0, 1.0)
            # Make the densest cores fully opaque.
            alpha = np.where(alpha > 0.85, 1.0, alpha).astype(np.float32)

        alpha3 = alpha[None, ...]  # [1, H, W] broadcast over channels

        # Alpha-blend cloud over the surface towards a bright cloud colour.
        cloud_color = np.full_like(arr, float(self.cfg.cloud_brightness))
        cloudy = arr * (1.0 - alpha3) + cloud_color * alpha3

        # Binary cloud mask: any non-trivial opacity counts as cloud.
        cloud_mask = (alpha > 0.1).astype(np.float32)[None, ...]

        # Cast shadow: offset the (opaque part of the) cloud away from the sun and
        # darken the surface there, but not where the cloud itself sits.
        shadow_mask = np.zeros((1, h, w), dtype=np.float32)
        if self.cfg.add_shadows and coverage > 0.0:
            dy, dx = self._shadow_offset(
                (h, w), sun_azimuth, sun_elevation, self.cfg.shadow_offset_frac
            )
            shadow_alpha = self._shift(alpha, dy, dx)
            # Shadow only where there is no cloud overhead.
            shadow_alpha = shadow_alpha * (1.0 - alpha)
            darken = 1.0 - (1.0 - self.cfg.shadow_strength) * shadow_alpha[None, ...]
            cloudy = cloudy * darken
            shadow_mask = (shadow_alpha > 0.1).astype(np.float32)[None, ...]

        cloudy = np.clip(cloudy, 0.0, 1.0).astype(np.float32)
        return cloudy, cloud_mask, shadow_mask


# --------------------------------------------------------------------------- #
# Functional one-shot
# --------------------------------------------------------------------------- #
def add_synthetic_clouds(
    clear,
    cloud_mode: str = "mixed",
    coverage: float = 0.35,
    thickness: float = 1.0,
    sun_azimuth: float = 135.0,
    sun_elevation: float = 45.0,
    seed: int | None = None,
    real_alpha: np.ndarray | None = None,
):
    """Add synthetic clouds + shadow to a clear image (NumPy or torch in/out).

    Convenience wrapper around :class:`CloudSimulator` matching the B1 contract
    signature ``add_synthetic_clouds(clear[C,H,W], ...) -> (cloudy, cloud_mask,
    shadow_mask)``. The output dtype mirrors the input: a torch tensor in returns
    torch tensors out; a NumPy array in returns NumPy out.

    Args:
        clear: ``[C, H, W]`` clear image (``numpy.ndarray`` or ``torch.Tensor``).
        cloud_mode: ``"thin"`` | ``"thick"`` | ``"mixed"``.
        coverage: Target cloud-cover fraction (0-1).
        thickness: Peak opacity for thick clouds.
        sun_azimuth: Solar azimuth (degrees, clockwise from north).
        sun_elevation: Solar elevation (degrees).
        seed: Optional determinism seed.
        real_alpha: Optional real cloud-alpha matte (copy-paste path).

    Returns:
        ``(cloudy, cloud_mask, shadow_mask)`` with the same array type as
        ``clear``; masks are ``[1, H, W]`` in ``{0, 1}``.
    """
    is_torch = _is_torch_tensor(clear)
    arr = _to_numpy(clear)
    sim = CloudSimulator(
        CloudSimConfig(cloud_mode=cloud_mode, coverage=coverage, thickness=thickness)
    )
    cloudy, cloud_mask, shadow_mask = sim(
        arr,
        sun_azimuth=sun_azimuth,
        sun_elevation=sun_elevation,
        seed=seed,
        real_alpha=real_alpha,
    )
    if is_torch:
        import torch

        return (
            torch.from_numpy(cloudy),
            torch.from_numpy(cloud_mask),
            torch.from_numpy(shadow_mask),
        )
    return cloudy, cloud_mask, shadow_mask


# --------------------------------------------------------------------------- #
# Small dtype helpers (avoid importing torch unless needed)
# --------------------------------------------------------------------------- #
def _is_torch_tensor(obj) -> bool:
    """Return ``True`` if ``obj`` is a torch tensor (without forcing an import)."""
    return type(obj).__module__.startswith("torch") and type(obj).__name__ == "Tensor"


def _to_numpy(obj) -> np.ndarray:
    """Convert a torch tensor or array-like to a contiguous ``float32`` NumPy array."""
    if _is_torch_tensor(obj):
        return np.ascontiguousarray(obj.detach().cpu().numpy(), dtype=np.float32)
    return np.ascontiguousarray(np.asarray(obj), dtype=np.float32)
