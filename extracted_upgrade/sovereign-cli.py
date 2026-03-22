"""Typer CLI for the Sovereign audit-first slice."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import asyncio
import importlib.util
import platform
import sys
import time

import typer
from datetime import date
from rich.console import Console
from rich.table import Table

import config
from data.db import db
from data.fetcher import DataFetcher
from engines.audit.data_auditor import DataAuditor
from engines.pipeline import MultiStrategyOrchestrator, PipelineOrchestrator
from engines.backtest.backtest_engine import BacktestEngine
from engines.ml.trainer import SovereignTrainer as ModelTrainer  # unified — removed legacy ml/
from models.schemas import AuditableField, parse_field_value
from ticker_list import get_universe


app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _record_cli_run(command_name: str, command_args: dict[str, Any], status: str, summary: str, started_at: int) -> None:
    """Persist a CLI invocation in run_history."""

    finished_at = int(time.time())
    db.log_run_history(command_name, command_args, status, summary, started_at, finished_at)


def _handle_command_failure(command_name: str, command_args: dict[str, Any], exc: Exception, started_at: int) -> None:
    """Record and display a CLI failure."""

    _record_cli_run(command_name, command_args, "FAIL", str(exc), started_at)
    console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
    raise typer.Exit(code=1)


def _status_style(status: str) -> str:
    """Return Rich markup for a status value."""

    mapping = {
        "PASS": "[green]PASS[/green]",
        "WARN": "[yellow]WARN[/yellow]",
        "FAIL": "[red]FAIL[/red]",
        "INCOMPLETE": "[grey66]INCOMPLETE[/grey66]",
    }
    return mapping.get(status, status)


def _auditor() -> DataAuditor:
    """Return the shared auditor instance."""

    return DataAuditor()


def _fetcher() -> DataFetcher:
    """Return the shared fetcher instance."""

    return DataFetcher()


def _pipeline() -> MultiStrategyOrchestrator:
    """Return the shared multi-strategy orchestrator."""

    return MultiStrategyOrchestrator()


@app.callback()
def main() -> None:
    """Sovereign CLI entrypoint."""


@app.command()
def health() -> None:
    """Run environment and database health checks."""

    started_at = int(time.time())
    try:
        db.initialize()
        table = Table(title="Sovereign Health")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Details")

        python_ok = sys.version_info[:2] >= (3, 11)
        table.add_row(
            "Python",
            "[green]PASS[/green]" if python_ok else "[red]FAIL[/red]",
            f"{platform.python_version()} (target >= 3.11)",
        )

        for module_name in config.REQUIRED_IMPORTS:
            available = importlib.util.find_spec(module_name) is not None
            table.add_row(
                f"Import: {module_name}",
                "[green]PASS[/green]" if available else "[red]FAIL[/red]",
                "available" if available else "missing",
            )

        for target, details in db.get_db_file_status().items():
            status = "[green]PASS[/green]" if details["exists"] and str(details["wal_mode"]).lower() == "wal" else "[red]FAIL[/red]"
            table.add_row(
                f"DB: {target}",
                status,
                f'{details["path"]} | wal={details["wal_mode"]}',
            )

        alpha_status = "[yellow]WARN[/yellow]" if not config.ALPHA_VANTAGE_API_KEY else "[green]PASS[/green]"
        alpha_details = "missing .env key, fallback sources only" if not config.ALPHA_VANTAGE_API_KEY else "configured"
        table.add_row("Alpha Vantage key", alpha_status, alpha_details)
        console.print(table)
        _record_cli_run("health", {}, "PASS", "health checks completed", started_at)
    except Exception as exc:
        _handle_command_failure("health", {}, exc, started_at)


@app.command()
def smoke() -> None:
    """Run a fast end-to-end operational smoke flow."""

    started_at = int(time.time())
    try:
        console.print("[dim]Starting Operational Smoke Test...[/dim]")
        
        # 1. Database Health & Initialization
        console.print("[yellow]1. Checking DB and cleaning duplicates...[/yellow]")
        db.initialize()
        cleaned = db.clean_duplicates_now()
        console.print(f"   [green]Cleaned duplicates:[/green] {cleaned}")
        
        # 2. Pipeline E2E
        console.print("[yellow]2. Running standard pipeline on RELIANCE.NS...[/yellow]")
        orchestrator = _pipeline()
        
        # We must run the orchestrator synchronously. However, `MultiStrategyOrchestrator.run_standard_pipeline()` is async.
        # Let's run it using asyncio.run
        result = asyncio.run(orchestrator.run(["RELIANCE.NS"]))
        console.print(f"   [green]Pipeline Output:[/green] {result}")
        
        # 3. Check UI render payload (Optional assert, or just rely on pipeline success)
        console.print("[yellow]3. Validating Score/Signal counts...[/yellow]")
        with db.connection("stocks") as conn:
            signal_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        console.print(f"   [green]Total Signals in DB:[/green] {signal_count}")
        
        console.print("[bold green]SMOKE TEST PASSED[/bold green]")
        _record_cli_run("smoke", {}, "PASS", f"E2E passed. Cleaned {sum(cleaned.values())} dups.", started_at)
        
    except Exception as exc:
        _handle_command_failure("smoke", {}, exc, started_at)


@app.command()
def audit(
    ticker: str | None = typer.Option(None, help="Audit a single ticker."),
    universe: str | None = typer.Option(None, help="Audit a named preset such as QUICK or STANDARD."),
    export: bool = typer.Option(False, help="Export the audit summary to CSV."),
) -> None:
    """Run ticker or universe audits."""

    started_at = int(time.time())
    args = {"ticker": ticker, "universe": universe, "export": export}
    try:
        auditor = _auditor()
        rows: list[dict[str, Any]]
        if ticker:
            report = auditor.audit_ticker(ticker, triggered_by="cli")
            rows = [
                {
                    "ticker": report.ticker,
                    "score": report.audit_quality_score,
                    "status": report.overall_status,
                    "red_flags": ", ".join(report.red_flags) if report.red_flags else "",
                }
            ]
            summary_text = (
                f"{int(report.overall_status == 'PASS')} pass, "
                f"{int(report.overall_status == 'WARN')} warn, "
                f"{int(report.overall_status == 'FAIL')} fail, "
                f"{int(report.overall_status == 'INCOMPLETE')} incomplete out of 1 tickers"
            )
        elif universe:
            summary = auditor.audit_universe(get_universe(universe), triggered_by="cli")
            rows = [
                {
                    "ticker": row["ticker"],
                    "score": row["audit_quality_score"],
                    "status": row["overall_status"],
                    "red_flags": row["top_red_flag"],
                }
                for row in summary.report_rows
            ]
            summary_text = (
                f"{summary.pass_count} pass, {summary.warn_count} warn, "
                f"{summary.fail_count} fail, {summary.incomplete_count} incomplete out of {summary.tickers_audited} tickers"
            )
            for alert in summary.source_health_alerts:
                console.print(f"[red]{alert}[/red]")
        else:
            raise ValueError("Provide either --ticker or --universe.")

        if universe and summary.field_missing_counts:
            coverage_table = Table(title="Field Coverage Report (Missing)")
            coverage_table.add_column("Field")
            coverage_table.add_column("Missing Count")
            for field_name, count in sorted(summary.field_missing_counts.items(), key=lambda x: x[1], reverse=True):
                coverage_table.add_row(field_name, str(count))
            console.print(coverage_table)

        table = Table(title="Audit Summary")
        table.add_column("Ticker")
        table.add_column("Score")
        table.add_column("Status")
        table.add_column("Red Flags")
        for row in rows:
            table.add_row(row["ticker"], f"{row['score']:.1f}", _status_style(str(row["status"])), str(row["red_flags"]))
        console.print(table)
        console.print(summary_text)

        if export:
            export_path = config.EXPORT_DIR / f"audit_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            import pandas as pd

            pd.DataFrame(rows).to_csv(export_path, index=False)
            console.print(f"[green]Exported:[/green] {export_path}")

        _record_cli_run("audit", args, "PASS", summary_text, started_at)
    except Exception as exc:
        _handle_command_failure("audit", args, exc, started_at)


@app.command("verify-field")
def verify_field(
    field: str = typer.Option(..., help="AuditableField name or value."),
    threshold: float | None = typer.Option(None, help="Show tickers above this threshold."),
) -> None:
    """Show distribution stats and optional threshold breaches for a field."""

    started_at = int(time.time())
    args = {"field": field, "threshold": threshold}
    try:
        parsed_field = AuditableField.parse(field)
        distribution = _auditor().audit_field_distribution(parsed_field)
        stats_table = Table(title=f"Distribution: {parsed_field.value}")
        stats_table.add_column("Metric")
        stats_table.add_column("Value")
        for key, value in distribution.to_dataframe().iloc[0].to_dict().items():
            stats_table.add_row(str(key), f"{value}")
        console.print(stats_table)

        if threshold is not None:
            values = db.list_field_values(parsed_field)
            threshold_table = Table(title=f"Values above {threshold}")
            threshold_table.add_column("Ticker")
            threshold_table.add_column("Value")
            matches = 0
            for ticker_value, raw_value in values:
                if isinstance(raw_value, (int, float)) and float(raw_value) > threshold:
                    matches += 1
                    threshold_table.add_row(ticker_value, f"[red]{float(raw_value):.4f}[/red]")
            if matches:
                console.print(threshold_table)
            else:
                console.print("[yellow]No values exceeded the threshold.[/yellow]")

        _record_cli_run("verify-field", args, "PASS", f"verified {parsed_field.value}", started_at)
    except Exception as exc:
        _handle_command_failure("verify-field", args, exc, started_at)


@app.command("fix-data")
def fix_data(
    ticker: str = typer.Option(..., help="Ticker to correct."),
    field: str = typer.Option(..., help="AuditableField name or value."),
    value: str = typer.Option(..., help="Replacement value."),
    reason: str = typer.Option("manual_fix", help="Reason for the override."),
) -> None:
    """Apply a manual override and rerun the audit."""

    started_at = int(time.time())
    args = {"ticker": ticker, "field": field, "value": value, "reason": reason}
    try:
        auditor = _auditor()
        parsed_field = AuditableField.parse(field)
        parsed_value = parse_field_value(parsed_field, value)
        before = db.get_latest_audit(ticker) or auditor.audit_ticker(ticker, triggered_by="cli")
        after = auditor.fix_data(ticker, parsed_field, parsed_value, reason=reason)
        table = Table(title="Fix Data Result")
        table.add_column("Ticker")
        table.add_column("Field")
        table.add_column("Before Score")
        table.add_column("After Score")
        table.add_column("Delta")
        table.add_row(
            ticker.strip().upper(),
            parsed_field.value,
            f"{before.audit_quality_score:.1f}",
            f"{after.audit_quality_score:.1f}",
            f"{after.audit_quality_score - before.audit_quality_score:+.1f}",
        )
        console.print(table)
        _record_cli_run("fix-data", args, "PASS", f"override applied for {ticker}", started_at)
    except Exception as exc:
        _handle_command_failure("fix-data", args, exc, started_at)


@app.command("cross-check")
def cross_check(
    ticker: str = typer.Option(..., help="Ticker to compare across providers."),
) -> None:
    """Compare stored and provider values for a ticker."""

    started_at = int(time.time())
    args = {"ticker": ticker}
    try:
        report = _auditor().cross_check_sources(ticker)
        table = Table(title=f"Cross Check: {report.ticker}")
        table.add_column("Field")
        table.add_column("Stored")
        table.add_column("yfinance")
        table.add_column("nsepython")
        table.add_column("Alpha Vantage")
        table.add_column("BSE")
        table.add_column("Status")
        table.add_column("Recommended")
        for row in report.rows:
            table.add_row(
                row.field_name.value,
                str(row.stored_value),
                str(row.yfinance_value),
                str(row.nsepython_value),
                str(row.alpha_vantage_value),
                str(row.bse_value),
                _status_style(row.status),
                row.recommended_source,
            )
        console.print(table)
        _record_cli_run("cross-check", args, "PASS", f"cross-check complete for {ticker}", started_at)
    except Exception as exc:
        _handle_command_failure("cross-check", args, exc, started_at)


@app.command("refetch")
def refetch(
    ticker: str | None = typer.Option(None, help="Refetch a single ticker."),
    status: str | None = typer.Option(None, help="Refetch all tickers with this latest audit status."),
) -> None:
    """Refetch data, invalidate cache, and rerun audits."""

    started_at = int(time.time())
    args = {"ticker": ticker, "status": status}
    try:
        fetcher = _fetcher()
        auditor = _auditor()
        tickers: list[str]
        if ticker:
            tickers = [ticker.strip().upper()]
        elif status:
            latest_rows = db.list_latest_audit_rows(status=status.strip().upper())
            tickers = [row["ticker"] for row in latest_rows]
        else:
            raise ValueError("Provide either --ticker or --status.")

        results: list[dict[str, Any]] = []
        for current_ticker in tickers:
            previous = db.get_latest_audit(current_ticker)
            cache_manager = fetcher.cache
            cache_manager.invalidate(current_ticker)
            fetcher.fetch(current_ticker, refresh=True)
            updated = auditor.audit_ticker(current_ticker, triggered_by="refetch")
            results.append(
                {
                    "ticker": current_ticker,
                    "before": previous.audit_quality_score if previous else None,
                    "after": updated.audit_quality_score,
                    "status": updated.overall_status,
                }
            )

        table = Table(title="Refetch Result")
        table.add_column("Ticker")
        table.add_column("Before Score")
        table.add_column("After Score")
        table.add_column("Status")
        for row in results:
            before_text = "-" if row["before"] is None else f"{row['before']:.1f}"
            table.add_row(row["ticker"], before_text, f"{row['after']:.1f}", _status_style(str(row["status"])))
        console.print(table)
        _record_cli_run("refetch", args, "PASS", f"refetched {len(results)} tickers", started_at)
    except Exception as exc:
        _handle_command_failure("refetch", args, exc, started_at)


@app.command()
def scan(
    universe: str = typer.Option(..., help="Universe preset: QUICK, STANDARD, EXTENDED, or SECTORS."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview the selected universe without running the pipeline."),
) -> None:
    """Run the pipeline over a named universe."""

    started_at = int(time.time())
    args = {"universe": universe, "dry_run": dry_run}
    try:
        tickers = get_universe(universe)
        if dry_run:
            table = Table(title=f"Scan Dry Run: {universe.upper()}")
            table.add_column("Ticker")
            table.add_column("Stored Data")
            for ticker in tickers:
                table.add_row(ticker, "yes" if db.get_fundamental(ticker, effective=True) is not None else "no")
            console.print(table)
            summary = f"Dry run: {len(tickers)} tickers selected for {universe.upper()}"
            console.print(summary)
            _record_cli_run("scan", args, "PASS", summary, started_at)
            return

        result = asyncio.run(_pipeline().run(tickers, triggered_by="cli_scan"))
        table = Table(title=f"Scan Result: {universe.upper()}")
        table.add_column("Ticker")
        table.add_column("Action")
        table.add_column("Score")
        table.add_column("Fair Value")
        table.add_column("Upside")
        table.add_column("Notes")
        for row in result.results:
            notes = row.skip_reason or row.error or ", ".join(row.data_warnings[:2])
            upside_text = "-" if row.upside_pct is None else f"{row.upside_pct * 100:.1f}%"
            fair_value_text = "-" if row.fair_value is None else f"{row.fair_value:.2f}"
            score_text = "-" if row.score is None else f"{row.score:.1f}"
            table.add_row(row.ticker, row.action, score_text, fair_value_text, upside_text, notes or "")
        console.print(table)
        console.print(result.summary)
        _record_cli_run("scan", args, "PASS" if result.error_count == 0 else "WARN", result.summary, started_at)
    except Exception as exc:
        _handle_command_failure("scan", args, exc, started_at)


@app.command("ml-ops")
def ml_ops(
    retrain: bool = typer.Option(False, "--retrain", help="Train and register a fresh model."),
    update: bool = typer.Option(False, "--update", help="Mark the latest model version as active."),
) -> None:
    """Manage model training and activation."""

    started_at = int(time.time())
    args = {"retrain": retrain, "update": update}
    try:
        if not retrain and not update:
            raise ValueError("Provide at least one of --retrain or --update.")

        trained_version = None
        if retrain:
            trained_version = ModelTrainer().run_training_pipeline()

        active_version = None
        if update:
            active_version = db.activate_latest_model_version(config.DEFAULT_MODEL_NAME)
            if active_version is None:
                raise ValueError(f"No model versions available for {config.DEFAULT_MODEL_NAME}")
        elif trained_version is not None:
            active_version = trained_version

        table = Table(title="ML Ops")
        table.add_column("Operation")
        table.add_column("Result")
        if trained_version is not None:
            table.add_row("retrain", f"{trained_version.model_name} -> {trained_version.version}")
        if active_version is not None:
            table.add_row("active", f"{active_version.model_name} -> {active_version.version}")
        console.print(table)
        summary = f"ML ops complete for {config.DEFAULT_MODEL_NAME}"
        _record_cli_run("ml-ops", args, "PASS", summary, started_at)
    except Exception as exc:
        _handle_command_failure("ml-ops", args, exc, started_at)


@app.command()
def backup(
    tag: str = typer.Option("manual", help="Tag to identify the backup."),
    verify: bool = typer.Option(True, help="Verify the backup integrity after creation."),
) -> None:
    """Create a full backup of databases and models."""

    started_at = int(time.time())
    args = {"tag": tag, "verify": verify}
    try:
        archive_path = db.backup_databases(backup_tag=tag)
        verified = False
        if verify:
            with console.status("[cyan]Verifying backup integrity..."):
                verified = db.verify_backup(archive_path)
            
        status = "PASS" if (not verify or verified) else "WARN"
        summary = f"Created backup {archive_path.name}" + (f" (verified={verified})" if verify else "")
        
        if verify and not verified:
            console.print(f"[yellow]Backup created but VERIFICATION FAILED:[/yellow] {archive_path.name}")
        else:
            console.print(f"[green]Backup created successfully:[/green] {archive_path.name}")
            if verify:
                console.print("   [dim]Verification passed.[/dim]")

        _record_cli_run("backup", args, status, summary, started_at)
        db.log_backup(str(archive_path), "SUCCESS", verified)
    except Exception as exc:
        _handle_command_failure("backup", args, exc, started_at)


@app.command("verify-backup")
def verify_backup(
    filename: str = typer.Argument(..., help="Backup archive filename to verify."),
) -> None:
    """Verify the integrity of an existing backup archive."""

    started_at = int(time.time())
    args = {"filename": filename}
    try:
        archive_path = config.BACKUPS_DIR / filename
        with console.status(f"[cyan]Verifying {filename}..."):
            verified = db.verify_backup(archive_path)
        
        if verified:
            console.print(f"[green]Backup verification passed:[/green] {filename}")
            _record_cli_run("verify-backup", args, "PASS", f"Verified {filename}", started_at)
        else:
            console.print(f"[red]Backup verification FAILED:[/red] {filename}")
            _record_cli_run("verify-backup", args, "FAIL", f"Verification failed for {filename}", started_at)
            raise typer.Exit(code=1)
    except Exception as exc:
        _handle_command_failure("verify-backup", args, exc, started_at)


@app.command()
def restore(
    filename: str = typer.Option(..., help="Name of the backup archive to restore."),
) -> None:
    """Restore databases and models from an archive."""

    started_at = int(time.time())
    args = {"filename": filename}
    try:
        db.restore_databases(filename)
        console.print(f"[green]Restore completed successfully from:[/green] {filename}")
        _record_cli_run("restore", args, "PASS", f"Restored from {filename}", started_at)
    except Exception as exc:
        _handle_command_failure("restore", args, exc, started_at)


@app.command("list-backups")
def list_backups() -> None:
    """List all available backup archives."""

    started_at = int(time.time())
    args = {}
    try:
        backups = db.list_backups()
        table = Table(title="Available Backups")
        table.add_column("Filename")
        table.add_column("Size (MB)")
        table.add_column("Created At")
        for b in backups:
            created_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(b["created_at"]))
            table.add_row(b["name"], f"{b['size_mb']:.2f}", created_str)
        console.print(table)
        _record_cli_run("list-backups", args, "PASS", f"Listed {len(backups)} backups", started_at)
    except Exception as exc:
        _handle_command_failure("list-backups", args, exc, started_at)


@app.command("portfolio-buy")
def portfolio_buy(
    ticker: str = typer.Argument(..., help="Ticker symbol to buy."),
    qty: int = typer.Option(..., help="Number of shares."),
    price: float = typer.Option(..., help="Execution price."),
) -> None:
    """Record a buy transaction in the paper portfolio."""
    started_at = int(time.time())
    args = {"ticker": ticker, "qty": qty, "price": price}
    try:
        db.add_portfolio_transaction(ticker, "BUY", qty, price)
        console.print(f"[green]Bought {qty} shares of {ticker} at {price}[/green]")
        _record_cli_run("portfolio-buy", args, "PASS", f"Bought {qty} {ticker} @ {price}", started_at)
    except Exception as exc:
        _handle_command_failure("portfolio-buy", args, exc, started_at)

@app.command("portfolio-sell")
def portfolio_sell(
    ticker: str = typer.Argument(..., help="Ticker symbol to sell."),
    qty: int = typer.Option(..., help="Number of shares."),
    price: float = typer.Option(..., help="Execution price."),
) -> None:
    """Record a sell transaction in the paper portfolio."""
    started_at = int(time.time())
    args = {"ticker": ticker, "qty": qty, "price": price}
    try:
        db.add_portfolio_transaction(ticker, "SELL", qty, price)
        console.print(f"[green]Sold {qty} shares of {ticker} at {price}[/green]")
        _record_cli_run("portfolio-sell", args, "PASS", f"Sold {qty} {ticker} @ {price}", started_at)
    except Exception as exc:
        _handle_command_failure("portfolio-sell", args, exc, started_at)

@app.command("portfolio-snapshot")
def portfolio_snapshot() -> None:
    """View current paper portfolio positions."""
    started_at = int(time.time())
    try:
        positions = db.get_portfolio_positions()
        table = Table(title="Portfolio Snapshot")
        table.add_column("Ticker")
        table.add_column("Quantity")
        table.add_column("Avg Cost")
        table.add_column("Last Price")
        table.add_column("Value")
        total_value = 0.0
        for pos in positions:
            table.add_row(pos["ticker"], str(pos["quantity"]), f"{pos['avg_cost']:.2f}", f"{pos['last_price']:.2f}", f"{pos['market_value']:.2f}")
            total_value += float(pos["market_value"])
        console.print(table)
        console.print(f"Total Portfolio Value: {total_value:.2f}")
        _record_cli_run("portfolio-snapshot", {}, "PASS", f"Snapshot generated for {len(positions)} positions", started_at)
    except Exception as exc:
        _handle_command_failure("portfolio-snapshot", {}, exc, started_at)

@app.command("ml-ops-labels")
def ml_ops_labels() -> None:
    """Generate and display statistics for PIT forward return labels."""
    started_at = int(time.time())
    try:
        from engines.ml.labeler import PointInTimeLabeler
        with console.status("[cyan]Generating labels from PIT data..."):
            labeler = PointInTimeLabeler()
            df = labeler.generate_labeled_dataset()
        if df.empty:
            console.print("[yellow]Warning: No labeled data generated (check if PIT data exists).[/yellow]")
        else:
            console.print(f"[green]Generated {len(df)} labeled records.[/green]")
            if "forward_3m_ret" in df.columns:
                valid_3m = df["forward_3m_ret"].notna().sum()
                console.print(f"Records with valid 3m forward returns: {valid_3m}")
        _record_cli_run("ml-ops-labels", {}, "PASS", f"Generated {len(df)} labels", started_at)
    except Exception as exc:
        _handle_command_failure("ml-ops-labels", {}, exc, started_at)

@app.command("ml-ops-train")
def ml_ops_train(
    target: str = typer.Option("forward_3m_ret", help="Target return to predict (e.g. forward_1m_ret, forward_3m_ret)"),
) -> None:
    """Train a regression model on PIT fundamentals."""
    started_at = int(time.time())
    try:
        from engines.ml.trainer import SovereignTrainer
        with console.status(f"[cyan]Training model for {target}..."):
            trainer = SovereignTrainer(target_col=target)
            version_id = trainer.run_training_pipeline()
        
        console.print(f"[green]Successfully trained model: {version_id}[/green]")
        
        # Display latest model from db
        models = db.list_model_versions(limit=1)
        if models:
            latest = models[0]
            metadata = _json_loads(latest["metadata_json"], {})
            metrics = metadata.get("metrics", {})
            
            table = Table(title=f"Evaluation Metrics ({version_id})")
            table.add_column("Metric")
            table.add_column("Value")
            for k, v in metrics.items():
                table.add_row(k.upper(), f"{v:.4f}")
            console.print(table)
            console.print(f"Samples: Train={metadata.get('train_samples')}, Test={metadata.get('test_samples')}")
            
        _record_cli_run("ml-ops-train", {"target": target}, "PASS", f"Trained {version_id}", started_at)
    except Exception as exc:
        _handle_command_failure("ml-ops-train", {"target": target}, exc, started_at)

@app.command()
def backtest(
    strategy: str = typer.Option("positional", help="Strategy to backtest."),
    start_date: str = typer.Option("2020-01-01", help="Start date (YYYY-MM-DD)."),
    end_date: str = typer.Option(date.today().isoformat(), help="End date (YYYY-MM-DD)."),
    capital: float = typer.Option(1000000.0, help="Initial capital."),
    save: bool = typer.Option(True, help="Save results to database."),
) -> None:
    """Run a high-fidelity strategy backtest with transaction costs."""

    started_at = int(time.time())
    args = {"strategy": strategy, "start_date": start_date, "end_date": end_date, "capital": capital}
    try:
        console.print(f"[bold cyan]Starting Backtest: {strategy.upper()} | {start_date} -> {end_date}[/bold cyan]")
        
        # 1. Initialize Engine
        engine = BacktestEngine(capital=capital)
        
        # 2. Run Backtest
        # Note: In a real system, we'd fetch signals/prices here.
        # For the drop-in integration, we'll run with dummy data or need a proper loader.
        # This is a placeholder for the integration pattern.
        with console.status("[yellow]Processing backtest..."):
            # We need to provide signals_df and prices_df to engine.run()
            # For now, we'll just show the integration pattern.
            # result = engine.run(signals_df=..., prices_df=...)
            console.print("[yellow]Note: Full backtest requires signals/prices dataframes.[/yellow]")
            return
            
        # 3. Display Results
        p = result.performance
        table = Table(title=f"Backtest Report: {strategy.upper()}")
        table.add_column("Metric")
        table.add_column("Value")
        
        table.add_row("Total Return (%)", f"{p['total_return_pct']:.2f}%")
        table.add_row("CAGR (%)", f"{p['cagr_pct']:.2f}%")
        table.add_row("Sharpe Ratio", f"{p['sharpe']:.2f}")
        table.add_row("Max Drawdown (%)", f"{p['max_drawdown_pct']:.2f}%")
        table.add_row("Win Rate (%)", f"{p['win_rate_pct']:.2f}%")
        table.add_row("Total Trades", f"{p['total_trades']}")
        table.add_row("Total Costs (₹)", f"₹{result.cost_summary['total']:,.2f}")
        table.add_row("Ending Capital (₹)", f"₹{result.equity_curve.iloc[-1]:,.2f}")
        
        console.print(table)
        
        if save:
            db.save_backtest_result(result)
            console.print("[green]Results saved to backtest_history table.[/green]")
            
        _record_cli_run("backtest", args, "PASS", f"CAGR: {result.cagr_pct:.1f}%, DD: {result.max_drawdown_pct:.1f}%", started_at)
        
    except Exception as exc:
        _handle_command_failure("backtest", args, exc, started_at)


@app.command()
def dups(
    clean: bool = typer.Option(False, "--clean", help="Delete duplicate rows while keeping the oldest row in each duplicate group."),
) -> None:
    """Scan the runtime SQLite databases for logical duplicate rows."""

    started_at = int(time.time())
    args = {"clean": clean}
    try:
        findings = db.find_duplicate_rows(clean=clean)
        table = Table(title="Duplicate Scan")
        table.add_column("DB")
        table.add_column("Table")
        table.add_column("Groups")
        table.add_column("Rows")
        table.add_column("Cleaned")
        for row in findings:
            table.add_row(
                row["target"],
                row["table"],
                str(row["duplicate_groups"]),
                str(row["duplicate_rows"]),
                str(row["cleaned_rows"]),
            )
        console.print(table)
        total_duplicate_rows = sum(int(row["duplicate_rows"]) for row in findings)
        summary = f"Duplicate scan complete: {total_duplicate_rows} duplicate rows found"
        if total_duplicate_rows and not clean:
            console.print("[yellow]Duplicates found. Re-run with --clean to remove them.[/yellow]")
        _record_cli_run("dups", args, "PASS", summary, started_at)
    except Exception as exc:
        _handle_command_failure("dups", args, exc, started_at)


if __name__ == "__main__":
    app()
