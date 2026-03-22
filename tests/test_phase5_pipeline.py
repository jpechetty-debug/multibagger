"""Tests for Phase 5 pipeline, alerts, scheduler, and persistence."""

from __future__ import annotations

import asyncio
import time

import config
from data.db import db
from engines.alert_engine import AlertEngine
from engines.pipeline import PipelineDependencies, PipelineOrchestrator
from app.scheduler import AppScheduler
from models.schemas import (
    AuditReport,
    CorrelationFilterResult,
    EarningsRevisionAnalysis,
    FundamentalAnalysis,
    FundamentalData,
    GateResult,
    LiquidityAnalysis,
    MarketRegime,
    MomentumAnalysis,
    OwnershipAnalysis,
    PipelineTickerResult,
    RegimeResult,
    RiskMetricsAnalysis,
    ScoreFeatureVector,
    ScoreResult,
    SectorRankAnalysis,
    SignalResult,
    ValuationResult,
    VixFilterResult,
    VixState,
)


def _cleanup_ticker(ticker: str) -> None:
    """Remove test rows for a Phase 5 ticker."""

    with db.connection("stocks") as conn:
        conn.execute("DELETE FROM manual_overrides WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM fundamentals WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM scores WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM score_history WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM valuations WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM signals WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM analysis_snapshots WHERE ticker = ?", (ticker,))
    with db.connection("ops") as conn:
        conn.execute("DELETE FROM audit_findings WHERE run_id IN (SELECT run_id FROM audit_runs WHERE ticker = ?)", (ticker,))
        conn.execute("DELETE FROM audit_runs WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM audit_actions WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM audit_log WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM logs WHERE component = 'engines.alert_engine' AND context_json LIKE ?", (f"%{ticker}%",))


def _sample_record(ticker: str = "TESTPH5") -> FundamentalData:
    """Return a reusable Phase 5 record."""

    return FundamentalData(
        ticker=ticker,
        company_name="Phase5 Test",
        sector="Financial Services",
        price=100.0,
        market_cap=50_000_000.0,
        avg_volume=100_000.0,
        roe_5y=0.18,
        roe_ttm=0.16,
        sales_growth_5y=0.12,
        eps_growth_ttm=0.10,
        cfo_to_pat=1.0,
        debt_equity=0.4,
        peg_ratio=1.0,
        pe_ratio=15.0,
        piotroski_score=7,
        promoter_pct=60.0,
        pledge_pct=0.0,
        fii_delta=1.0,
        dii_delta=0.5,
        updated_at=int(time.time()),
        ingestion_quality_score=95.0,
    )


class _StaticFetcher:
    """Deterministic fetcher test double."""

    def __init__(self, record: FundamentalData) -> None:
        self.record = record

    def fetch(self, ticker: str, refresh: bool = False) -> FundamentalData:
        return self.record


class _StaticAuditor:
    """Deterministic auditor test double."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def audit_ticker(self, ticker: str, triggered_by: str = "test") -> AuditReport:
        return AuditReport(
            ticker=ticker,
            run_id=f"audit-{ticker}",
            timestamp=int(time.time()),
            overall_status="PASS",
            audit_quality_score=95.0,
            fail_count=0,
            warn_count=0,
            missing_count=0,
            field_results=[],
            red_flags=[],
            suggested_fixes=[],
            triggered_by=triggered_by,
        )


class _StaticGate:
    """Deterministic pre-scan gate."""

    def check(self, ticker: str, data: FundamentalData, latest_audit=None) -> GateResult:
        return GateResult(passed=True, effective_quality_score=95.0, warnings=[])


class _StaticAnalyzer:
    """Deterministic analyzer test double."""

    def __init__(self, result) -> None:
        self.result = result

    def analyze(self, ticker: str, data: FundamentalData | None = None):
        return self.result


class _StaticScoreEngine:
    """Score engine double that also persists the score contract."""

    def __init__(self, result: ScoreResult) -> None:
        self.result = result

    def score_ticker(self, ticker: str, data: FundamentalData | None = None) -> ScoreResult:
        db.save_score_result(self.result)
        return self.result


class _StaticValuationEngine:
    """Valuation engine double that persists the valuation contract."""

    def __init__(self, result: ValuationResult) -> None:
        self.result = result

    def value_ticker(self, ticker: str, data: FundamentalData | None = None) -> ValuationResult:
        db.save_valuation_result(self.result)
        return self.result


class _StaticSignalEngine:
    """Signal engine double that persists the signal contract."""

    def __init__(self, result: SignalResult) -> None:
        self.result = result

    def evaluate(self, ticker: str, data: FundamentalData | None = None) -> SignalResult:
        db.save_signal_result(self.result)
        return self.result


class _StaticRegimeDetector:
    """Regime detector test double."""

    def detect(self) -> RegimeResult:
        return RegimeResult(
            regime=MarketRegime.BULL,
            india_vix=14.0,
            breadth_ratio=1.4,
            advance_count=1500,
            decline_count=900,
            reason="test",
            as_of=int(time.time()),
        )


class _StaticVixFilter:
    """VIX filter test double."""

    def evaluate(self) -> VixFilterResult:
        return VixFilterResult(vix_value=14.0, state=VixState.NORMAL, position_multiplier=1.0, reason="ok", as_of=int(time.time()))


class _StaticCorrelation:
    """Correlation analyzer test double."""

    def correlation_matrix(self, tickers: list[str]) -> CorrelationFilterResult:
        return CorrelationFilterResult(
            tickers=tickers,
            correlation_matrix={ticker: {other: 0.0 for other in tickers} for ticker in tickers},
            allowed_tickers=tickers,
            as_of=int(time.time()),
        )


class _StaticAlertEngine:
    """Alert engine test double."""

    def __init__(self) -> None:
        self.seen: list[PipelineTickerResult] = []

    def process_run(self, results: list[PipelineTickerResult]) -> list[dict[str, object]]:
        self.seen.extend(results)
        return [{"count": len(results)}]


class _FakeJob:
    """Minimal scheduled job representation."""

    def __init__(self, job_id: str) -> None:
        self.id = job_id
        self.next_run_time = "stub"


class _FakeScheduler:
    """Minimal scheduler double for registration tests."""

    def __init__(self) -> None:
        self.jobs: list[_FakeJob] = []

    def add_job(self, func, trigger, id: str, replace_existing: bool = True) -> None:
        self.jobs.append(_FakeJob(id))

    def get_jobs(self) -> list[_FakeJob]:
        return self.jobs


def test_pipeline_run_persists_snapshots_and_logs() -> None:
    """Pipeline should persist market snapshots, analysis snapshots, and run history."""

    ticker = "TESTPH5PIPE"
    _cleanup_ticker(ticker)
    record = _sample_record(ticker)
    db.upsert_fundamental(record)
    now_ts = int(time.time())
    dependencies = PipelineDependencies(
        fetcher=_StaticFetcher(record),
        auditor=_StaticAuditor(ticker),
        pre_scan_gate=_StaticGate(),
        fundamentals=_StaticAnalyzer(FundamentalAnalysis(ticker=ticker, roe_5y=0.18, eps_growth_ttm=0.10, cfo_to_pat=1.0, piotroski_score=7, piotroski_checks={}, score=82.0, as_of=now_ts)),
        earnings_revision=_StaticAnalyzer(EarningsRevisionAnalysis(ticker=ticker, beat_streak=2, revision_signal="UPGRADE", estimate_trend_pct=0.05, surprise_mean=7.0, score=70.0, as_of=now_ts)),
        momentum=_StaticAnalyzer(MomentumAnalysis(ticker=ticker, price_return_3m=0.12, benchmark_return_3m=0.08, relative_strength_3m=0.04, price_vs_50dma_pct=0.05, volume_acceleration=1.1, above_50dma=True, score=68.0, as_of=now_ts)),
        ownership=_StaticAnalyzer(OwnershipAnalysis(ticker=ticker, promoter_pct=60.0, pledge_pct=0.0, fii_delta=1.0, dii_delta=0.5, ownership_clean=True, score=80.0, as_of=now_ts)),
        sector_rank=_StaticAnalyzer(SectorRankAnalysis(ticker=ticker, sector="Financial Services", peer_count=4, sector_rank=1, rank_percentile=1.0, top_3=True, dupont={}, common_size={}, score=90.0, as_of=now_ts)),
        liquidity=_StaticAnalyzer(LiquidityAnalysis(ticker=ticker, avg_daily_volume_20d=100_000.0, turnover_value_20d=10_000_000.0, delivery_pct=None, liquidity_ok=True, score=75.0, as_of=now_ts)),
        risk_metrics=_StaticAnalyzer(RiskMetricsAnalysis(ticker=ticker, volatility_6m=0.22, beta_vs_nifty=0.95, max_drawdown_6m=-0.10, score=72.0, as_of=now_ts)),
        regime=_StaticRegimeDetector(),
        vix_filter=_StaticVixFilter(),
        correlation=_StaticCorrelation(),
        score_engine=_StaticScoreEngine(ScoreResult(ticker=ticker, regime=MarketRegime.BULL, weighted_score=78.0, meta_model_score=70.0, total_score=76.0, action="WATCH", feature_vector=ScoreFeatureVector(ticker=ticker, feature_names=["x"], values=[1.0]), generated_at=now_ts)),
        valuation_engine=_StaticValuationEngine(ValuationResult(ticker=ticker, dcf_value=130.0, eps_value=125.0, graham_value=120.0, peg_value=118.0, fair_value=118.0, margin_of_safety_pct=0.15, undervalued=True, generated_at=now_ts)),
        signal_engine=_StaticSignalEngine(SignalResult(ticker=ticker, action="BUY", confidence_score=85.0, reason_code="test", generated_at=now_ts)),
        alert_engine=_StaticAlertEngine(),
    )
    result = asyncio.run(PipelineOrchestrator(dependencies=dependencies).run([ticker], triggered_by="test"))
    assert result.processed_count == 1
    assert db.get_analysis_snapshot(ticker, "momentum") is not None
    assert db.get_latest_market_snapshot("regime") is not None
    assert db.get_latest_signal(ticker)["action"] == "BUY"
    assert db.list_run_history(limit=1)[0]["command_name"] == "pipeline.run"


def test_alert_engine_logs_buy_signal_without_telegram() -> None:
    """Alert engine should log alerts even when Telegram credentials are absent."""

    ticker = "TESTPH5ALERT"
    _cleanup_ticker(ticker)
    db.upsert_fundamental(_sample_record(ticker))
    before = len(db.list_logs(limit=200, component_prefix="engines.alert_engine"))
    result = PipelineTickerResult(ticker=ticker, action="BUY", score=82.0, fair_value=None, generated_at=int(time.time()))
    AlertEngine().process_run([result])
    after_logs = db.list_logs(limit=200, component_prefix="engines.alert_engine")
    assert len(after_logs) >= before + 1
    assert any("BUY signal" in row["message"] for row in after_logs)


def test_scheduler_registers_phase5_jobs() -> None:
    """Scheduler should register the expected recurring jobs."""

    scheduler = _FakeScheduler()
    app_scheduler = AppScheduler(
        pipeline=object(),  # type: ignore[arg-type]
        trainer=object(),  # type: ignore[arg-type]
        alert_engine=object(),  # type: ignore[arg-type]
        scheduler=scheduler,  # type: ignore[arg-type]
    )
    job_ids = {job.id for job in scheduler.get_jobs()}
    assert job_ids == {
        "daily_scan",
        "swing_scan",
        "multibagger_scan",
        "weekly_retrain",
        "daily_report",
        "weekly_report",
        "cache_evict",
        "db_optimize",
        "daily_backup",
    }
    assert len(app_scheduler.describe_jobs()) == 9


def test_backup_database_creates_archive_and_lists_it(tmp_path, monkeypatch) -> None:
    """Backup helper should create an archive in the configured backup directory."""

    monkeypatch.setattr(config, "BACKUPS_DIR", tmp_path / "backups")
    config.ensure_runtime_dirs()

    archive_path = db.backup_databases(backup_tag="phase5test")
    backups = db.list_backups()

    assert archive_path.exists()
    assert archive_path.parent == config.BACKUPS_DIR
    assert any(row["name"] == archive_path.name for row in backups)
