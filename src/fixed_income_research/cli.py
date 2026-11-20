"""Command line: ``uv run fir <command>``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from fixed_income_research.data.cache import ResearchCache
from fixed_income_research.pipelines.load_ice import load_ice
from fixed_income_research.pipelines.load_index_levels import load_index_levels
from fixed_income_research.quality.checks import BOND_CHECKS
from research_core.data.db.engine import get_engine
from research_core.data.quality import run_checks

app = typer.Typer(help="QuantResearch: the fixed income pipelines", no_args_is_help=True)


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log each file as it loads.")] = False) -> None:
    """QuantResearch: the fixed income pipelines (a command group: one sub-command per pipeline)."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(asctime)s  %(message)s")


@app.command("load-ice")
def load_ice_command(
    folder: Annotated[Path | None, typer.Option(help="Folder of the constituent files (default: EQR_ICE_DIR).")] = None,
) -> None:
    """Load ICE's index constituent files (new ones only): bonds, issuers, index weights, analytics."""
    typer.echo(json.dumps(load_ice(get_engine(), folder), indent=2, default=str))


@app.command("load-index-levels")
def load_index_levels_command() -> None:
    """Load each index's official total-return level from FRED (its last three years; what was loaded stays)."""
    typer.echo(json.dumps(load_index_levels(get_engine()), indent=2, default=str))


@app.command("checks")
def checks_command() -> None:
    """Run the fixed income data-quality gates."""
    failed = False
    for r in run_checks(get_engine(), BOND_CHECKS):
        typer.echo(f"{r.status:5s} {r.name}: {r.value:.6g} ({r.threshold})")
        failed |= r.status == "fail"
    if failed:
        raise typer.Exit(code=1)


@app.command("sync-cache")
def sync_cache() -> None:
    """Refresh your local fixed income research cache."""
    typer.echo(json.dumps(ResearchCache().sync(get_engine()), indent=2))
