"""Command-line interface for ``cloudremoval`` (Typer).

Single entry point (``cloudremoval``) exposing the pipeline subcommands
(BUILD_PLAN §3.10):

    download · preprocess · simulate · train · eval · benchmark · infer · serve

Each subcommand loads the :class:`Config` (with optional ``--set key=value``
overrides), seeds RNGs, then **lazily** imports and calls the corresponding
``scripts/<name>.py:main(cfg, **kwargs)`` (owned by B1/B3/B4). The lazy import
means a missing ``scripts`` module raises a clean *"not yet implemented"* message
for that one command instead of breaking the whole CLI — so ``cloudremoval
--help`` and every other command keep working during parallel development.

This module imports only Typer + the core package, so it loads on the CPU-smoke
stack.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import typer

from cloudremoval import __version__
from cloudremoval.config import Config, load_config
from cloudremoval.utils.logging import get_logger
from cloudremoval.utils.seed import seed_everything

app = typer.Typer(
    name="cloudremoval",
    help="GenAI cloud removal & reconstruction for LISS-IV imagery (BAH 2026 PS2).",
    add_completion=False,
    no_args_is_help=True,
)
_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _coerce(value: str) -> Any:
    """Best-effort coercion of a CLI override string to bool/int/float/str."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Turn ``["model.name=unet", "train.max_steps=2"]`` into a nested dict.

    Args:
        pairs: List of ``dotted.key=value`` strings (from ``--set``).

    Returns:
        A nested mapping suitable for :func:`load_config`'s ``overrides``.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise typer.BadParameter(f"--set expects key=value, got '{pair}'")
        key, raw = pair.split("=", 1)
        node = overrides
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(raw.strip())
    return overrides


def _load(config: Path, set_: list[str] | None) -> Config:
    """Load a :class:`Config` from ``config`` applying ``--set`` overrides, seed RNGs."""
    cfg = load_config(config, overrides=_parse_overrides(set_))
    seed_everything(cfg.train.seed)
    return cfg


def _ensure_scripts_importable() -> None:
    """Put the repository root on ``sys.path`` so ``import scripts.*`` resolves.

    The ``scripts/`` package lives at the repository root (not under ``src/``), so
    it is *not* installed by ``pip install -e .``. The installed ``cloudremoval``
    console script therefore runs with a ``sys.path`` that does not include the
    repo root, and ``importlib.import_module("scripts.<name>")`` would fail even
    though the file exists. We locate the repo root relative to this module
    (``…/src/cloudremoval/cli.py`` → repo root) and prepend it if it actually
    contains a ``scripts`` package. This is a no-op when running from the repo
    root (or when already on the path).
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    if (repo_root / "scripts" / "__init__.py").exists():
        root = str(repo_root)
        if root not in sys.path:
            sys.path.insert(0, root)


def _run_script(module_name: str, cfg: Config, **kwargs: Any) -> None:
    """Lazily import ``scripts.<module_name>`` and call its ``main(cfg, **kwargs)``.

    Args:
        module_name: The ``scripts`` module to run (e.g. ``"train"``).
        cfg: The loaded configuration.
        **kwargs: Extra keyword arguments forwarded to ``main``.

    Raises:
        typer.Exit: With a clear message if the script module (or its ``main``)
            does not exist yet — keeps the rest of the CLI usable.
    """
    _ensure_scripts_importable()
    try:
        module = importlib.import_module(f"scripts.{module_name}")
    except ModuleNotFoundError as exc:
        # Distinguish "scripts.<name> missing" from a genuine missing dependency
        # raised *inside* an existing script module.
        missing = getattr(exc, "name", "") or ""
        if missing == f"scripts.{module_name}" or missing == "scripts":
            typer.secho(
                f"[cloudremoval] '{module_name}' is not yet implemented "
                f"(scripts/{module_name}.py is missing). "
                "It will be provided by another build stage.",
                fg=typer.colors.YELLOW,
                err=True,
            )
            raise typer.Exit(code=2) from exc
        raise
    main = getattr(module, "main", None)
    if main is None:
        typer.secho(
            f"[cloudremoval] scripts/{module_name}.py defines no main(cfg, **kwargs).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    main(cfg, **kwargs)


# --------------------------------------------------------------------------- #
# Top-level callback (version)
# --------------------------------------------------------------------------- #
def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"cloudremoval {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """GenAI cloud removal for LISS-IV. Use a subcommand; see ``--help``."""


# --------------------------------------------------------------------------- #
# Subcommands (BUILD_PLAN §3.10)
# --------------------------------------------------------------------------- #
_CONFIG_OPT = typer.Option(..., "--config", "-c", help="Path to a YAML config.")
_SET_OPT = typer.Option(None, "--set", "-s", help="Override config: dotted.key=value (repeatable).")


@app.command()
def download(
    config: Path = _CONFIG_OPT,
    set_: list[str] = _SET_OPT,
    aoi: Path | None = typer.Option(None, "--aoi", help="Area-of-interest KML/GeoJSON."),
    source: str | None = typer.Option(
        None, "--source", help="bhoonidhi | stac | gee | sentinel | dem."
    ),
) -> None:
    """Download / fetch source imagery for an AOI (delegates to ``scripts.download``)."""
    cfg = _load(config, set_)
    _run_script("download", cfg, aoi=aoi, source=source)


@app.command()
def preprocess(config: Path = _CONFIG_OPT, set_: list[str] = _SET_OPT) -> None:
    """Radiometric/geometric preprocessing + masking (delegates to ``scripts.preprocess``)."""
    cfg = _load(config, set_)
    _run_script("preprocess", cfg)


@app.command()
def simulate(config: Path = _CONFIG_OPT, set_: list[str] = _SET_OPT) -> None:
    """Generate synthetic cloudy/clear paired tiles (delegates to ``scripts.simulate``)."""
    cfg = _load(config, set_)
    _run_script("simulate", cfg)


@app.command()
def train(
    config: Path = _CONFIG_OPT,
    set_: list[str] = _SET_OPT,
    ckpt: Path | None = typer.Option(None, "--ckpt", help="Resume from checkpoint."),
) -> None:
    """Train the configured model (delegates to ``scripts.train``)."""
    cfg = _load(config, set_)
    _run_script("train", cfg, ckpt=ckpt)


@app.command()
def eval(  # noqa: A001 - matches the CLI verb in the contract
    config: Path = _CONFIG_OPT,
    set_: list[str] = _SET_OPT,
    ckpt: Path | None = typer.Option(None, "--ckpt", help="Checkpoint to evaluate."),
) -> None:
    """Evaluate a model with masked/stratified MVES metrics (delegates to ``scripts.eval``)."""
    cfg = _load(config, set_)
    _run_script("eval", cfg, ckpt=ckpt)


@app.command()
def benchmark(config: Path = _CONFIG_OPT, set_: list[str] = _SET_OPT) -> None:
    """Benchmark all registered models into a leaderboard (delegates to ``scripts.benchmark``)."""
    cfg = _load(config, set_)
    _run_script("benchmark", cfg)


@app.command()
def infer(
    config: Path = _CONFIG_OPT,
    set_: list[str] = _SET_OPT,
    ckpt: Path | None = typer.Option(None, "--ckpt", help="Model checkpoint."),
    input: Path | None = typer.Option(None, "--input", help="Input scene / stack."),
    out: Path | None = typer.Option(None, "--out", help="Output COG path."),
) -> None:
    """Run tiled inference and write a reconstructed COG (delegates to ``scripts.infer``)."""
    cfg = _load(config, set_)
    _run_script("infer", cfg, ckpt=ckpt, input=input, out=out)


@app.command()
def serve(config: Path = _CONFIG_OPT, set_: list[str] = _SET_OPT) -> None:
    """Launch the FastAPI serving app (delegates to ``scripts.serve``)."""
    cfg = _load(config, set_)
    _run_script("serve", cfg)


def main() -> None:
    """Console-script entry point (``cloudremoval=cloudremoval.cli:main``-equiv)."""
    app()


if __name__ == "__main__":
    app()
