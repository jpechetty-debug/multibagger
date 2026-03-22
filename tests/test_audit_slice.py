"""Focused tests for the audit-first vertical slice."""

from __future__ import annotations

import time

import pytest

from data.cache import CacheManager
from data.db import db
from data.fetcher import DataFetcher
from engines.audit.data_auditor import DataAuditor
from engines.audit.pre_scan_gate import PreScanGate
from models.schemas import AuditReport, AuditableField, FundamentalData, SourceSnapshot, SourceSnapshotBundle


def _cleanup_ticker(ticker: str) -> None:
    """Remove test rows for a ticker from all relevant tables."""

    with db.connection("stocks") as conn:
        conn.execute("DELETE FROM manual_overrides WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM fundamentals WHERE ticker = ?", (ticker,))
    with db.connection("ops") as conn:
        conn.execute("DELETE FROM audit_findings WHERE run_id IN (SELECT run_id FROM audit_runs WHERE ticker = ?)", (ticker,))
        conn.execute("DELETE FROM audit_runs WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM audit_actions WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM audit_log WHERE ticker = ?", (ticker,))


class FakeFetcher:
    """Small test double for audit flows that should not hit live providers."""

    def __init__(self, record: FundamentalData) -> None:
        """Store the record returned by this fake fetcher."""

        self.record = record

    def fetch(self, ticker: str, refresh: bool = False) -> FundamentalData:
        """Return the prebuilt record."""

        return self.record

    def fetch_source_snapshots(self, ticker: str, refresh: bool = False) -> SourceSnapshotBundle:
        """Return only the yfinance price snapshot needed by the auditor."""

        return SourceSnapshotBundle(
            ticker=ticker,
            snapshots=[
                SourceSnapshot(
                    source="yfinance",
                    ticker=ticker,
                    fetched_at=int(time.time()),
                    fields={"price": self.record.price, "source_updated_at": int(time.time())},
                )
            ],
        )


def test_db_initialization_creates_expected_tables() -> None:
    """Database initialization should route migrations to the correct files."""

    db.initialize()
    expected_tables = {
        "stocks": {"fundamentals", "manual_overrides"},
        "pit": {"fundamentals_pit"},
        "cache": {"cache_entries"},
        "ops": {"run_history", "logs", "audit_log", "audit_runs", "audit_findings", "audit_actions", "audit_universe_runs"},
    }
    for target, tables in expected_tables.items():
        with db.connection(target) as conn:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert tables.issubset(names)


def test_manual_override_merges_effective_record() -> None:
    """Effective reads should include the latest active override without mutating the base row."""

    ticker = "TESTOVERRIDE"
    _cleanup_ticker(ticker)
    db.upsert_fundamental(
        FundamentalData(
            ticker=ticker,
            company_name="Override Test",
            sector="Financial Services",
            price=100.0,
            market_cap=20_000_000.0,
            avg_volume=50_000.0,
            roe_5y=0.15,
            roe_ttm=0.14,
            sales_growth_5y=0.12,
            eps_growth_ttm=0.10,
            cfo_to_pat=1.1,
            debt_equity=0.4,
            peg_ratio=1.5,
            pe_ratio=15.0,
            piotroski_score=7,
            promoter_pct=55.0,
            pledge_pct=0.0,
            updated_at=int(time.time()),
            ingestion_quality_score=95.0,
        )
    )
    db.add_override(ticker, AuditableField.PRICE, 120.0, "test_override")
    base = db.get_fundamental(ticker, effective=False)
    effective = db.get_fundamental(ticker, effective=True)
    assert base is not None and effective is not None
    assert base.price == 100.0
    assert effective.price == 120.0


def test_cache_manager_roundtrip() -> None:
    """Cache manager should store, retrieve, and invalidate entries cleanly."""

    cache = CacheManager()
    cache.invalidate("TESTCACHE")
    cache.set("TESTCACHE", "yfinance", {"price": 321.0}, ttl=60)
    assert cache.get("TESTCACHE", "yfinance") == {"price": 321.0}
    cache.invalidate("TESTCACHE", "yfinance")
    assert cache.get("TESTCACHE", "yfinance") is None


def test_prescan_gate_uses_audit_score_when_fresh() -> None:
    """Pre-scan gate should prioritize a fresh audit score over ingestion quality."""

    record = FundamentalData(
        ticker="TESTGATE",
        company_name="Gate Test",
        sector="Financial Services",
        price=250.0,
        market_cap=50_000_000.0,
        avg_volume=75_000.0,
        roe_5y=0.10,
        roe_ttm=0.09,
        sales_growth_5y=0.12,
        eps_growth_ttm=0.11,
        cfo_to_pat=0.8,
        debt_equity=0.5,
        peg_ratio=1.2,
        pe_ratio=13.2,
        piotroski_score=6,
        promoter_pct=60.0,
        pledge_pct=0.0,
        updated_at=int(time.time()),
        ingestion_quality_score=90.0,
    )
    latest_audit = AuditReport(
        ticker="TESTGATE",
        run_id="audit-1",
        timestamp=int(time.time()),
        overall_status="FAIL",
        audit_quality_score=40.0,
        fail_count=1,
        warn_count=0,
        missing_count=0,
        field_results=[],
        red_flags=[],
        suggested_fixes=[],
        triggered_by="test",
    )
    result = PreScanGate().check("TESTGATE", record, latest_audit=latest_audit)
    assert result.passed is False
    assert result.effective_quality_score == 40.0
    assert result.skip_reason == "LOW_DATA_QUALITY"


def test_auditor_scores_and_persists_report() -> None:
    """Audits should persist runs and findings for seeded data."""

    ticker = "TESTAUDIT"
    _cleanup_ticker(ticker)
    record = FundamentalData(
        ticker=ticker,
        company_name="Audit Test",
        sector="Financial Services",
        price=100.0,
        market_cap=30_000_000.0,
        avg_volume=80_000.0,
        roe_5y=0.20,
        roe_ttm=0.18,
        sales_growth_5y=0.14,
        eps_growth_ttm=0.12,
        cfo_to_pat=0.9,
        debt_equity=0.4,
        peg_ratio=1.1,
        pe_ratio=13.2,
        piotroski_score=7,
        promoter_pct=58.0,
        pledge_pct=35.0,
        updated_at=int(time.time()),
        ingestion_quality_score=92.0,
        source_metadata={field.value: "seed" for field in AuditableField if field.value != "updated_at"},
    )
    db.upsert_fundamental(record)
    auditor = DataAuditor(fetcher=FakeFetcher(record))
    report = auditor.audit_ticker(ticker, triggered_by="test")
    assert report.overall_status == "FAIL"
    assert report.fail_count >= 1
    latest = db.get_latest_audit(ticker)
    assert latest is not None
    assert latest.run_id == report.run_id


def test_audit_universe_reports_score_distribution() -> None:
    """Universe audit summaries should expose score bands and aggregate stats."""

    ticker = "TESTAUDSUM"
    _cleanup_ticker(ticker)
    record = FundamentalData(
        ticker=ticker,
        company_name="Audit Summary Test",
        sector="Financial Services",
        price=100.0,
        market_cap=30_000_000.0,
        avg_volume=80_000.0,
        roe_5y=0.20,
        roe_ttm=0.18,
        sales_growth_5y=0.14,
        eps_growth_ttm=0.12,
        cfo_to_pat=0.9,
        debt_equity=0.4,
        peg_ratio=1.1,
        pe_ratio=13.2,
        piotroski_score=7,
        promoter_pct=58.0,
        pledge_pct=0.0,
        updated_at=int(time.time()),
        ingestion_quality_score=92.0,
    )
    db.upsert_fundamental(record)
    summary = DataAuditor(fetcher=FakeFetcher(record)).audit_universe([ticker], triggered_by="test", refresh_live=False)
    assert summary.tickers_audited == 1
    assert summary.average_score is not None
    assert summary.median_score is not None
    assert sum(summary.score_distribution.values()) == 1
    latest_summary = db.list_universe_audit_runs(limit=1)
    assert latest_summary
    assert latest_summary[0]["tickers"] == [ticker]


def test_fetcher_parses_bse_promoter_and_pledge_fallback(monkeypatch) -> None:
    """BSE ownership fallback should parse promoter and pledge percentages."""

    def _fake_bse_api(self, endpoint: str, params: dict[str, object]):
        if endpoint == "SHPQNewFormat/w":
            return {"Table": [{"qtrid": 128.0, "xbrlurl": None}]}
        if endpoint == "shpSecSummery_New/w":
            return {
                "Data": (
                    "<table><tr><td>Whether any shares held by promoters are pledge or otherwise encumbered?</td>"
                    "<td>No</td></tr><tr><td>(A) Promoter & Promoter Group</td><td>10</td><td>1000</td>"
                    "<td>-</td><td>1000</td><td>61.25</td></tr></table>"
                )
            }
        if endpoint == "ConsolidatePledge/w":
            return {"Table": [{"F_NewCol": 2.5, "FLAg_Pledge": "Y"}]}
        raise AssertionError(endpoint)

    monkeypatch.setattr(DataFetcher, "_resolve_bse_scrip_code", lambda self, ticker, preferred_name=None: "500001")
    monkeypatch.setattr(DataFetcher, "_bse_api_get", _fake_bse_api)
    snapshot = DataFetcher()._fetch_bse_ownership("TEST", preferred_name="Test Company")
    assert snapshot.fields["promoter_pct"] == 61.25
    assert snapshot.fields["pledge_pct"] == 2.5


def test_fetcher_penalizes_conflicting_sources() -> None:
    """Fetcher should detect conflicts across sources and apply a penalty."""

    import config
    bundle = SourceSnapshotBundle(
        ticker="CONFLICT",
        snapshots=[
            SourceSnapshot(
                source=config.SOURCE_NAME_YFINANCE,
                ticker="CONFLICT",
                fetched_at=int(time.time()),
                fields={"pe_ratio": 15.0, "source_updated_at": int(time.time())},
            ),
            SourceSnapshot(
                source=config.SOURCE_NAME_NSEPYTHON,
                ticker="CONFLICT",
                fetched_at=int(time.time()),
                fields={"pe_ratio": 20.0, "source_updated_at": int(time.time())},
            ),
        ],
    )
    record = DataFetcher()._resolve_canonical_record(bundle)
    assert "CONFLICT_PE_RATIO" in record.ingestion_issues
    # Assuming INGESTION_CONFLICT_PENALTY reduces score from baseline
    assert record.ingestion_quality_score < 100


def test_fetcher_penalizes_stale_source() -> None:
    """Fetcher should apply field-level staleness penalties."""

    import config
    stale_time = int(time.time()) - (config.WARN_STALE_DAYS + 1) * 86400
    bundle = SourceSnapshotBundle(
        ticker="STALE",
        snapshots=[
            SourceSnapshot(
                source=config.SOURCE_NAME_YFINANCE,
                ticker="STALE",
                fetched_at=int(time.time()),
                fields={"price": 100.0, "source_updated_at": stale_time},
            )
        ],
    )
    record = DataFetcher()._resolve_canonical_record(bundle)
    assert "STALE_PRICE" in record.ingestion_issues
    assert "STALE_SOURCE_DATA" in record.ingestion_issues
