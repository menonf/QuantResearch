"""Command line: ``uv run eqr <command>``."""

from __future__ import annotations

import datetime as dt
import json
import logging

import keyring
import typer

from equity_research.db.engine import get_engine, make_engine
from equity_research.db.migrate import upgrade_to_latest
from equity_research.pipelines.copy_database import copy_database
from equity_research.pipelines.load_benchmarks import load_benchmarks
from equity_research.pipelines.load_corrections import load_corrections
from equity_research.pipelines.load_estimates import STEPS as ESTIMATE_STEPS
from equity_research.pipelines.load_estimates import load_estimates
from equity_research.pipelines.load_estimates import probe as probe_estimates
from equity_research.pipelines.load_financedatabase import load_financedatabase
from equity_research.pipelines.load_lseg import START as LSEG_START
from equity_research.pipelines.load_lseg import load_lseg, probe, probe_uk
from equity_research.pipelines.load_norgate import load_norgate
from equity_research.pipelines.load_norgate import probe as probe_norgate
from equity_research.pipelines.load_rules import load_rules
from equity_research.pipelines.load_sec import load_sec
from equity_research.pipelines.load_sec_text import load_sec_text
from equity_research.pipelines.load_simfin import load_simfin
from equity_research.pipelines.load_simfin import probe as probe_simfin
from equity_research.pipelines.load_tiingo import load_tiingo
from equity_research.pipelines.load_tiingo import probe as probe_tiingo
from equity_research.pipelines.refresh import refresh
from equity_research.pipelines.sources import describe_sources, probe_sources, run_sources
from equity_research.quality.checks import CORE_CHECKS, run_checks
from equity_research.raw.store import prune
from equity_research.research.cache import ResearchCache
from equity_research.settings import KEYRING_SERVICE, REPO_ROOT, DatabaseSettings, get_settings

app = typer.Typer(help="EquityResearch pipelines", no_args_is_help=True)


@app.callback()
def main() -> None:
    """EquityResearch pipelines (a command group: one sub-command per pipeline)."""


def _logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("urllib3", "alembic.runtime.migration"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command("load-lseg")
def load_lseg_command(
    *,
    verbose: bool = False,
    since: str | None = None,
    steps: str = "",
    market: str = "US",
    start: str | None = None,
    indices: str = "",
) -> None:
    """Load from LSEG (needs LSEG Workspace running): the whole history, or from --since YYYY-MM-DD. Resumable.

    --market UK loads London (the FTSE UK indices); --start and --indices FTSE100 narrow a trial build; --steps
    defaults to the market's own.
    """
    _logging(verbose)
    report = load_lseg(
        get_engine(),
        start=start or LSEG_START,
        since=dt.date.fromisoformat(since) if since else None,
        steps=[s for s in steps.split(",") if s] or None,
        market=market,
        indices=[i for i in indices.split(",") if i] or None,
    )
    typer.echo(json.dumps(report, indent=2, default=str))


@app.command("probe-lseg")
def probe_lseg_command() -> None:
    """Ask LSEG one small question per dataset and check its answers read correctly (before a long load)."""
    typer.echo(probe().to_string(index=False))


@app.command("probe-lseg-uk")
def probe_lseg_uk_command() -> None:
    """Check what a London (FTSE UK) load relies on: members, closes, currencies, half-years, levels (~20 requests)."""
    typer.echo(probe_uk().to_string(index=False))


@app.command("load-estimates")
def load_estimates_command(
    *,
    verbose: bool = False,
    since: str | None = None,
    start: str | None = None,
    end: str | None = None,
    steps: str = ",".join(ESTIMATE_STEPS),
    market: str = "US",
) -> None:
    """Load analysts' estimates from LSEG (Workspace running): consensus, actuals, revisions. Resumable.

    The whole history (--start, default 2005) or a top-up from --since YYYY-MM-DD; --end limits a trial build.
    """
    _logging(verbose)
    report = load_estimates(
        get_engine(),
        start=start or LSEG_START,
        end=end,
        since=dt.date.fromisoformat(since) if since else None,
        steps=[s for s in steps.split(",") if s],
        market=market,
    )
    typer.echo(json.dumps(report, indent=2, default=str))


@app.command("probe-estimates")
def probe_estimates_command() -> None:
    """Ask LSEG one small question per estimates dataset and measure, read by the load's own code (~20 requests)."""
    typer.echo(probe_estimates().to_string(index=False))


@app.command("load-norgate")
def load_norgate_command(verbose: bool = False, since: str | None = None) -> None:
    """Load from Norgate Data (Norgate Data Updater running): members with history, listings, prices, actions."""
    _logging(verbose)
    report = load_norgate(get_engine(), since=dt.date.fromisoformat(since) if since else None)
    typer.echo(json.dumps(report, indent=2, default=str))


@app.command("probe-norgate")
def probe_norgate_command() -> None:
    """Ask Norgate questions with known answers and check the adapter reads them right."""
    typer.echo(probe_norgate().to_string(index=False))


@app.command("load-financedatabase")
def load_financedatabase_command(verbose: bool = False) -> None:
    """Enrich our listings from FinanceDatabase: codes, the country a company is run from, its sector labels."""
    _logging(verbose)
    typer.echo(json.dumps(load_financedatabase(get_engine()), indent=2, default=str))


@app.command("load-corrections")
def load_corrections_command(verbose: bool = False) -> None:
    """Write the repository's corporate-action corrections (events no vendor recorded) under vendor ``repository``."""
    _logging(verbose)
    typer.echo(json.dumps(load_corrections(get_engine()), indent=2, default=str))


@app.command("load-rules")
def load_rules_command(verbose: bool = False) -> None:
    """Write the repository's strategy rule sets to the shared tables (a set already loaded is left alone)."""
    _logging(verbose)
    typer.echo(json.dumps(load_rules(get_engine()), indent=2, default=str))


@app.command("set-key")
def set_key_command(vendor: str) -> None:
    """Store a vendor's API key (tiingo, simfin) in the OS keyring, typed without echo - never on the command line."""
    if vendor not in ("tiingo", "simfin"):
        raise typer.BadParameter("tiingo or simfin")
    key = typer.prompt(f"{vendor} API key", hide_input=True).strip()
    keyring.set_password(KEYRING_SERVICE, f"{vendor}_api_key", key)
    typer.echo(f"stored as {KEYRING_SERVICE} / {vendor}_api_key")


@app.command("load-tiingo")
def load_tiingo_command(verbose: bool = False, max_requests: int | None = None) -> None:
    """Price the index members from Tiingo (API key in the keyring); stops at --max-requests or Tiingo's allowance."""
    _logging(verbose)
    typer.echo(json.dumps(load_tiingo(get_engine(), max_requests=max_requests), indent=2, default=str))


@app.command("probe-tiingo")
def probe_tiingo_command() -> None:
    """Ask Tiingo questions with known answers (one request) and check the adapter reads them right."""
    typer.echo(probe_tiingo().to_string(index=False))


@app.command("load-simfin")
def load_simfin_command(verbose: bool = False, max_age_days: float = 7.0) -> None:
    """Load SimFin's quarterly statements and sector labels for the companies in the database."""
    _logging(verbose)
    typer.echo(json.dumps(load_simfin(get_engine(), max_age_days=max_age_days), indent=2, default=str))


@app.command("probe-simfin")
def probe_simfin_command() -> None:
    """Download SimFin's companies and income files and check the adapter reads them right."""
    typer.echo(probe_simfin().to_string(index=False))


@app.command("load-sec")
def load_sec_command(verbose: bool = False, max_age_days: float = 7.0) -> None:
    """Load SEC XBRL filings and financial facts (as reported) for the issuers in the database."""
    _logging(verbose)
    report = load_sec(get_engine(), max_age_days=max_age_days)
    typer.echo(json.dumps(report, indent=2, default=str))


@app.command("load-sec-text")
def load_sec_text_command(
    *,
    verbose: bool = False,
    since: str = "2003-01-01",
    max_age_days: float = 30.0,
    issuers: str = typer.Option("", help="comma-separated issuer ids: a trial on some companies only"),
    workers: int = typer.Option(0, help="processes measuring text (0: all CPUs but one)"),
    tone: bool = typer.Option(False, help="also score MD&A tone with FinBERT (dependency group text)"),
    tone_device: str = typer.Option("cpu", help="where FinBERT runs: cpu, or cuda with a CUDA build of torch"),
) -> None:
    """Load SEC filing indexes and annual reports, and measure their text (dictionary counts, similarity)."""
    _logging(verbose)
    chosen = [int(x) for x in issuers.split(",") if x.strip()] or None
    report = load_sec_text(
        get_engine(),
        since=since,
        max_age_days=max_age_days,
        issuers=chosen,
        workers=workers or None,
        tone=tone,
        tone_device=tone_device,
    )
    typer.echo(json.dumps(report, indent=2, default=str))


@app.command("load-benchmarks")
def load_benchmarks_command(verbose: bool = False) -> None:
    """Load S&P index levels and the SPY / RSP ETFs from Yahoo Finance."""
    _logging(verbose)
    typer.echo(json.dumps(load_benchmarks(get_engine()), indent=2, default=str))


@app.command("checks")
def checks_command(verbose: bool = False) -> None:
    """Run the data quality checks and record them."""
    _logging(verbose)
    for r in run_checks(get_engine(), CORE_CHECKS):
        typer.echo(f"{r.status:5} {r.name}: {r.value:.6g} ({r.threshold})")


@app.command("probe")
def probe_command() -> None:
    """Ask every source of this build (EQR_SOURCES) its questions with known answers, before a long load."""
    typer.echo(probe_sources().to_string(index=False))


@app.command("sources")
def sources_command() -> None:
    """Every data source: what it supplies, what it needs, whether this build runs it."""
    typer.echo(describe_sources().to_string(index=False))


@app.command("audit")
def audit_command(all_rules: bool = False) -> None:
    """Check the data against the architecture and its fitness for backtesting (read-only; --all-rules: every rule)."""
    import pandas as pd  # noqa: PLC0415

    from equity_research.quality.audit import audit  # noqa: PLC0415

    findings = audit(get_engine())
    shown = findings if all_rules else findings[~findings["ok"]]
    with pd.option_context("display.width", 250, "display.max_colwidth", 140):
        typer.echo(shown.to_string(index=False) if len(shown) else "every rule holds")
    typer.echo(f"{int((~findings['ok']).sum())} of {len(findings)} rules do not hold")


@app.command("build")
def build_command(verbose: bool = False) -> None:
    """Load every source from nothing (or continue an unfinished build) - see EQR_SOURCES."""
    _logging(verbose)
    typer.echo(json.dumps(run_sources(get_engine(), "build"), indent=2, default=str))


@app.command("refresh")
def refresh_command(verbose: bool = False) -> None:
    """Bring everything up to date: every source, the quality gates, the cache, the raw-zone clean-up (daily)."""
    _logging(verbose)
    typer.echo(json.dumps(refresh(get_engine()), indent=2, default=str))


@app.command("copy-database")
def copy_database_command(source_server: str, source_name: str, verbose: bool = False) -> None:
    """Copy a local database (Windows authentication) into the configured one - empty - and check every table."""
    _logging(verbose)
    settings = get_settings()
    source_db = DatabaseSettings(source_server, source_name, None, None, settings.database.driver)
    target = get_engine()
    typer.echo(f"copying {source_server}/{source_name} -> {settings.database.server}/{settings.database.name}")
    typer.echo(f"target migrated to {upgrade_to_latest(target)}")
    report = copy_database(
        make_engine(source_db),
        target,
        source_raw=REPO_ROOT / "data" / source_name / "raw",
        target_raw=settings.raw_dir,
    )
    for t in report:
        typer.echo(f"{'ok  ' if t.ok else 'DIFF'} {t.table}: {t.source_rows:,} -> {t.target_rows:,}")
    if not all(t.ok for t in report):
        raise typer.Exit(code=1)


@app.command("prune-raw")
def prune_raw_command(dry_run: bool = False) -> None:
    """Remove old copies of raw files the vendor publishes again anyway (the SEC bulk file); keeps the newest two."""
    with get_engine().begin() as conn:
        pruned = prune(conn, dry_run=dry_run)
    for p in pruned:
        typer.echo(
            f"{'would remove' if dry_run else 'removed'} batch {p.batch_id} {p.vendor}/{p.dataset}: {p.path.name}"
        )
    typer.echo(f"{len(pruned)} files, {sum(p.bytes for p in pruned) / 1e6:,.1f} MB")


@app.command("sync-cache")
def sync_cache(verbose: bool = False, rebuild: list[str] | None = None) -> None:
    """Pull new and changed rows into the local research cache.

    ``--rebuild TABLE`` copies a table afresh - after a repair deleted rows (``--rebuild fund.financial_fact``).
    """
    _logging(verbose)
    typer.echo(json.dumps(ResearchCache().sync(get_engine(), rebuild=tuple(rebuild or ())), indent=2))


if __name__ == "__main__":
    app()
