"""Quantitative + qualitative assessment artifact: Markdown (and optional HTML) report.

:func:`generate_report` turns an evaluator results dict **or** a benchmark
leaderboard into a human-readable Markdown report — metric tables, a per-model
summary, and qualitative-section placeholders (triptychs, error/ΔNDVI heatmaps,
spectral-profile plots) per the protocol in
``research/05_metrics_preprocessing_deployment.md`` §A.7. ``matplotlib`` is
imported lazily for optional figures and skipped cleanly when absent.

Pure stdlib for the core Markdown path; nothing heavy is required to import or to
produce the text report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cloudremoval.utils.logging import get_logger

__all__ = ["generate_report", "leaderboard_to_markdown", "results_to_markdown"]

_log = get_logger(__name__)

# Display order + nicer headers for the common metrics.
_METRIC_ORDER = [
    "psnr",
    "ssim",
    "ms_ssim",
    "sam",
    "ergas",
    "sid",
    "rmse",
    "mae",
    "ndvi_mae",
    "spectral_correlation",
    "lpips",
]
_METRIC_LABEL = {
    "psnr": "PSNR↑",
    "ssim": "SSIM↑",
    "ms_ssim": "MS-SSIM↑",
    "sam": "SAM°↓",
    "ergas": "ERGAS↓",
    "sid": "SID↓",
    "rmse": "RMSE↓",
    "mae": "MAE↓",
    "ndvi_mae": "NDVI-MAE↓",
    "spectral_correlation": "BandCorr↑",
    "lpips": "LPIPS↓",
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_report(
    results: dict[str, Any] | list[dict[str, Any]],
    out_path: str | Path,
    title: str = "Cloud-Removal Evaluation Report",
    figures: list[dict[str, Any]] | None = None,
    html: bool = False,
) -> str:
    """Write a Markdown assessment report (and optionally an HTML copy).

    Args:
        results: Either an :class:`Evaluator` results dict (has a ``"metrics"``
            key) **or** a benchmark leaderboard (a ``list`` of per-model dicts).
        out_path: Output ``.md`` path (parent dirs created). An ``.html`` sibling
            is also written when ``html=True``.
        title: Report title.
        figures: Optional list of figure specs for the qualitative section. Each
            spec may contain ``{"title", "cloudy", "recon", "reference",
            "uncertainty"}`` with image arrays/tensors; if ``matplotlib`` is
            available a triptych PNG is rendered and linked, otherwise a
            placeholder note is emitted.
        html: Also emit a minimal HTML rendering next to the Markdown.

    Returns:
        The Markdown file path written (as a string).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    sections.append(f"# {title}\n")
    sections.append(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n")

    is_leaderboard = isinstance(results, list)
    if is_leaderboard:
        sections.append(leaderboard_to_markdown(results))  # type: ignore[arg-type]
    else:
        sections.append(results_to_markdown(results))  # type: ignore[arg-type]

    # Qualitative section (figures or placeholders).
    sections.append(_qualitative_section(out, figures))

    # Methodology note.
    sections.append(_methodology_note())

    markdown = "\n".join(sections).rstrip() + "\n"
    out.write_text(markdown, encoding="utf-8")
    _log.info("Wrote Markdown report to %s", out)

    if html:
        html_path = out.with_suffix(".html")
        html_path.write_text(_markdown_to_html(markdown, title), encoding="utf-8")
        _log.info("Wrote HTML report to %s", html_path)

    return str(out)


def results_to_markdown(results: dict[str, Any]) -> str:
    """Render a single-model evaluator results dict as Markdown tables."""
    lines: list[str] = []
    name = results.get("model", "model")
    n = results.get("num_samples", "?")
    lines.append(f"## Quantitative Results — `{name}`\n")
    lines.append(f"Evaluated on **{n}** samples.\n")

    metrics: dict[str, dict[str, float]] = results.get("metrics", {})
    if metrics:
        lines.append(_metrics_table(metrics))
    else:
        lines.append("_No metrics available._\n")

    bias: dict[str, list[float]] = results.get("per_band_bias", {})
    if bias:
        lines.append("\n### Per-band reflectance bias (mean pred − target)\n")
        lines.append(_bias_table(bias))
    return "\n".join(lines)


def leaderboard_to_markdown(leaderboard: list[dict[str, Any]]) -> str:
    """Render a benchmark leaderboard as a ranked Markdown comparison table."""
    lines: list[str] = ["## Comparative Assessment — Leaderboard\n"]
    if not leaderboard:
        lines.append("_No models evaluated._\n")
        return "\n".join(lines)

    rank_by = leaderboard[0].get("rank_by", "")
    if rank_by:
        lines.append(f"Ranked by **{rank_by}** (1 = best).\n")

    # Choose a compact, readable column set: rank, model, key cloud + whole metrics.
    headline = _headline_columns(leaderboard)
    header = ["Rank", "Model", "Params", "Latency(ms)", *[_pretty_col(c) for c in headline]]
    rows: list[list[str]] = []
    for row in leaderboard:
        cells = [
            str(row.get("rank", "")),
            f"`{row.get('model', '')}`",
            _fmt_int(row.get("params")),
            _fmt(row.get("latency_ms")),
            *[_fmt(row.get(c)) for c in headline],
        ]
        rows.append(cells)
    lines.append(_md_table(header, rows))

    # Full per-stratum tables per model (collapsible-ish via subheadings).
    lines.append("\n### Full metric panel per model\n")
    for row in leaderboard:
        lines.append(f"\n#### `{row.get('model','')}` (rank {row.get('rank','?')})\n")
        lines.append(_grouped_metric_table(row))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #
def _metrics_table(metrics: dict[str, dict[str, float]]) -> str:
    """Build a stratum × metric Markdown table from ``{stratum: {metric: v}}``."""
    strata = list(metrics.keys())
    present = [m for m in _METRIC_ORDER if any(m in metrics[s] for s in strata)]
    # Include any non-standard metrics at the end.
    extra = sorted(
        {m for s in strata for m in metrics[s] if m not in _METRIC_ORDER}
    )
    cols = present + extra
    header = ["Stratum", *[_METRIC_LABEL.get(c, c) for c in cols]]
    rows: list[list[str]] = []
    for stratum in strata:
        cells = [f"**{stratum}**", *[_fmt(metrics[stratum].get(c)) for c in cols]]
        rows.append(cells)
    return _md_table(header, rows)


def _bias_table(bias: dict[str, list[float]]) -> str:
    """Per-band bias table (one row per stratum, one column per band)."""
    max_bands = max((len(v) for v in bias.values()), default=0)
    band_labels = ["Green", "Red", "NIR"][:max_bands] + [
        f"B{i}" for i in range(3, max_bands)
    ]
    header = ["Stratum", *band_labels]
    rows: list[list[str]] = []
    for stratum, values in bias.items():
        cells = [f"**{stratum}**", *[_fmt(v) for v in values]]
        rows.append(cells)
    return _md_table(header, rows)


def _grouped_metric_table(row: dict[str, Any]) -> str:
    """Reconstruct a stratum × metric table from a flattened leaderboard row."""
    metrics: dict[str, dict[str, float]] = {}
    for key, value in row.items():
        if "/" not in key or key.endswith("_rank"):
            continue
        stratum, metric = key.split("/", 1)
        metrics.setdefault(stratum, {})[metric] = value
    if not metrics:
        return "_No per-stratum metrics recorded._\n"
    return _metrics_table(metrics)


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    """Assemble a GitHub-flavoured Markdown table."""
    sep = ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Qualitative section
# --------------------------------------------------------------------------- #
def _qualitative_section(out: Path, figures: list[dict[str, Any]] | None) -> str:
    """Render qualitative figures (if matplotlib + data) or placeholders."""
    lines = ["\n## Qualitative Assessment\n"]
    lines.append(
        "Per the protocol (research §A.7): side-by-side triptychs "
        "(cloudy | reconstruction | reference), absolute-error & ΔNDVI heatmaps, "
        "and sampled spectral-profile plots.\n"
    )
    if not figures:
        lines.append(
            "\n> _Placeholder._ No figure data was supplied. Pass `figures=[...]` "
            "with `cloudy`/`recon`/`reference` arrays to render triptychs, or attach "
            "qualitative panels exported during inference.\n"
        )
        lines.append("\n- [ ] Triptychs (cloudy | reconstruction | reference)\n")
        lines.append("- [ ] Absolute-error heatmaps (shared colorbar)\n")
        lines.append("- [ ] ΔNDVI error maps inside the cloud mask\n")
        lines.append("- [ ] Spectral-profile plots for sampled transects\n")
        return "\n".join(lines)

    fig_dir = out.parent / f"{out.stem}_figures"
    for i, spec in enumerate(figures):
        rel = _render_triptych(spec, fig_dir, i)
        title = spec.get("title", f"Sample {i}")
        if rel is not None:
            lines.append(f"\n**{title}**\n\n![{title}]({rel})\n")
        else:
            lines.append(
                f"\n**{title}** — _figure skipped (matplotlib unavailable or "
                f"missing data)._\n"
            )
    return "\n".join(lines)


def _render_triptych(spec: dict[str, Any], fig_dir: Path, index: int) -> str | None:
    """Render one triptych PNG; return a path relative to the report, or ``None``."""
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # numpy is a core dep
    except Exception as exc:  # noqa: BLE001
        _log.warning("matplotlib unavailable (%s); skipping figure %d.", exc, index)
        return None

    panels = [
        ("Cloudy", spec.get("cloudy")),
        ("Reconstruction", spec.get("recon")),
        ("Reference", spec.get("reference")),
    ]
    panels = [(t, a) for t, a in panels if a is not None]
    if not panels:
        return None

    def to_rgb(arr: Any) -> Any:
        a = _to_numpy(arr, np)
        if a is None:
            return None
        if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[0] < a.shape[-1]:
            a = np.transpose(a, (1, 2, 0))  # CHW -> HWC
        if a.ndim == 3 and a.shape[-1] >= 3:
            a = a[..., :3]
        elif a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        if hi > lo:
            a = (a - lo) / (hi - lo)
        return np.clip(a, 0.0, 1.0)

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (label, arr) in zip(axes, panels, strict=False):
        img = to_rgb(arr)
        if img is None:
            ax.axis("off")
            continue
        ax.imshow(img, cmap=None if img.ndim == 3 else "viridis")
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    png = fig_dir / f"triptych_{index:03d}.png"
    fig.savefig(png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f"{fig_dir.name}/{png.name}"


def _to_numpy(arr: Any, np_mod: Any) -> Any:
    """Best-effort conversion of a tensor/array-like to a NumPy array."""
    if arr is None:
        return None
    if hasattr(arr, "detach"):  # torch tensor
        arr = arr.detach().cpu()
        if arr.dim() == 4:  # take first item of a batch
            arr = arr[0]
        return arr.numpy()
    try:
        a = np_mod.asarray(arr)
    except Exception:  # noqa: BLE001
        return None
    if a.ndim == 4:
        a = a[0]
    return a


# --------------------------------------------------------------------------- #
# Formatting + misc
# --------------------------------------------------------------------------- #
def _headline_columns(leaderboard: list[dict[str, Any]]) -> list[str]:
    """Pick a compact headline column set present in the leaderboard rows."""
    candidates = [
        "cloud/psnr",
        "cloud/ssim",
        "cloud/sam",
        "cloud/ndvi_mae",
        "whole/psnr",
        "whole/ssim",
    ]
    present: list[str] = []
    for col in candidates:
        if any(_is_number(r.get(col)) for r in leaderboard):
            present.append(col)
    return present


def _pretty_col(col: str) -> str:
    """Human header for a ``"stratum/metric"`` column."""
    if "/" in col:
        stratum, metric = col.split("/", 1)
        return f"{_METRIC_LABEL.get(metric, metric)} ({stratum})"
    return _METRIC_LABEL.get(col, col)


def _is_number(value: Any) -> bool:
    """True for a finite int/float (not bool/NaN)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value


def _fmt(value: Any) -> str:
    """Format a numeric cell to 4 sig-ish digits; ``n/a`` for missing/NaN."""
    if value is None or (isinstance(value, float) and value != value):
        return "n/a"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1000 or (value != 0 and abs(value) < 1e-3):
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


def _fmt_int(value: Any) -> str:
    """Format an integer cell with thousands separators."""
    if value is None:
        return "n/a"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _methodology_note() -> str:
    """Standard methodology footnote explaining the masked-evaluation discipline."""
    return (
        "\n## Methodology Notes\n\n"
        "- **Masked evaluation is the headline.** Whole-image metrics are inflated "
        "by the already-clear majority of pixels; the **cloud** (and "
        "cloud-shadow) region columns are the real scores.\n"
        "- **Spectral fidelity gates analysis-readiness:** SAM / ERGAS / NDVI-MAE "
        "matter more than PSNR for downstream NDVI/LULC. PSNR can be 'won' by "
        "over-smoothing.\n"
        "- SSIM/MS-SSIM are computed **per band then averaged** (NIR drives NDVI).\n"
        "- LPIPS (if present) is on an RGB render and is a *secondary* realism "
        "signal only.\n"
    )


def _markdown_to_html(markdown: str, title: str) -> str:
    """Minimal Markdown→HTML (uses the ``markdown`` package if present).

    Falls back to a ``<pre>`` wrapper so an HTML file is always produced without a
    hard dependency.
    """
    try:
        import markdown as md  # type: ignore

        body = md.markdown(markdown, extensions=["tables", "fenced_code"])
    except Exception:  # noqa: BLE001 - graceful fallback
        from html import escape

        body = f"<pre>{escape(markdown)}</pre>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;"
        "padding:0 1rem}table{border-collapse:collapse}td,th{border:1px solid #ccc;"
        "padding:4px 8px}</style></head><body>"
        f"{body}</body></html>"
    )


# Keep a JSON dump helper handy for callers that want a machine-readable sidecar.
def dump_json(obj: Any, path: str | Path) -> str:
    """Write ``obj`` as pretty JSON to ``path`` (parent dirs created)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return str(p)
