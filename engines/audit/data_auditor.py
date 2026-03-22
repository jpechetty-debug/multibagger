"""Audit engine for canonical ticker data."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any
import time
import uuid

import pandas as pd

import config
from data.cache import cache_manager
from data.db import db
from data.fetcher import DataFetcher, DataQualitySkip
from models.schemas import (
    AuditReport,
    AuditableField,
    FieldAudit,
    FieldDistribution,
    FundamentalData,
    SourceSnapshotBundle,
    SourceComparisonReport,
    SourceComparisonRow,
    UniverseAuditSummary,
)


SEVERITY_ORDER = {"PASS": 0, "MISSING": 1, "WARN": 2, "FAIL": 3}


class DataAuditor:
    """Audits canonical stored data, produces reports, and manages overrides."""

    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        """Initialize the auditor."""

        self.fetcher = fetcher or DataFetcher()
        db.initialize()

    def audit_ticker(
        self,
        ticker: str,
        triggered_by: str = "manual",
        refresh_live: bool = True,
    ) -> AuditReport:
        """Audit a single ticker and persist the result."""

        normalized_ticker = ticker.strip().upper()
        if refresh_live:
            try:
                record = self.fetcher.fetch(normalized_ticker, refresh=True)
            except DataQualitySkip:
                record = db.get_fundamental(normalized_ticker, effective=True)
                if record is None:
                    raise
            live_bundle = self.fetcher.fetch_source_snapshots(normalized_ticker, refresh=False)
        else:
            record = db.get_fundamental(normalized_ticker, effective=True)
            if record is None:
                record = self.fetcher.fetch(normalized_ticker)
            live_bundle = SourceSnapshotBundle(ticker=normalized_ticker, snapshots=[])
        live_by_source = live_bundle.by_source()
        live_price = live_by_source.get(config.SOURCE_NAME_YFINANCE, None)
        live_price_value = None
        if live_price:
            live_price_value = live_price.fields.get("price")

        findings = self._initialize_findings(record)
        red_flags: list[str] = []
        suggested_fixes: list[str] = []

        self._apply_range_checks(record, findings)
        self._apply_cross_checks(record, live_price_value, findings, red_flags)
        report = self._build_report(record, findings, red_flags, suggested_fixes, triggered_by)
        db.save_audit_report(report)
        db.log_engine_event(
            "INFO",
            "engines.audit.data_auditor",
            "ticker audited",
            {
                "ticker": normalized_ticker,
                "status": report.overall_status,
                "score": report.audit_quality_score,
            },
        )
        return report

    def audit_universe(
        self,
        tickers: list[str],
        triggered_by: str = "manual",
        refresh_live: bool = True,
    ) -> UniverseAuditSummary:
        """Audit a universe of tickers and aggregate the results."""

        reports = [
            self.audit_ticker(ticker, triggered_by=triggered_by, refresh_live=refresh_live)
            for ticker in tickers
        ]
        pass_count = sum(report.overall_status == "PASS" for report in reports)
        warn_count = sum(report.overall_status == "WARN" for report in reports)
        fail_count = sum(report.overall_status == "FAIL" for report in reports)
        incomplete_count = sum(report.overall_status == "INCOMPLETE" for report in reports)
        field_fail_counts: Counter[str] = Counter()
        field_warn_counts: Counter[str] = Counter()
        field_missing_counts: Counter[str] = Counter()
        report_rows: list[dict[str, Any]] = []
        pledge_values: list[float] = []
        scores: list[float] = []

        for report in reports:
            top_red_flag = report.red_flags[0] if report.red_flags else ""
            scores.append(float(report.audit_quality_score))
            report_rows.append(
                {
                    "ticker": report.ticker,
                    "overall_status": report.overall_status,
                    "audit_quality_score": report.audit_quality_score,
                    "fail_count": report.fail_count,
                    "warn_count": report.warn_count,
                    "missing_count": report.missing_count,
                    "red_flags_count": len(report.red_flags),
                    "top_red_flag": top_red_flag,
                }
            )
            for finding in report.field_results:
                if finding.status == "FAIL":
                    field_fail_counts[finding.field_name.value] += 1
                elif finding.status == "WARN":
                    field_warn_counts[finding.field_name.value] += 1
                elif finding.status == "MISSING":
                    field_missing_counts[finding.field_name.value] += 1
                if finding.field_name is AuditableField.PLEDGE_PCT and isinstance(finding.stored_value, (int, float)):
                    pledge_values.append(float(finding.stored_value))

        source_health_alerts: list[str] = []
        if reports and pledge_values and all(value == 0.0 for value in pledge_values):
            source_health_alerts.append("All pledge values are zero - BSE source likely broken")

        score_distribution = {
            "90_100": sum(score >= 90 for score in scores),
            "80_89": sum(80 <= score < 90 for score in scores),
            "70_79": sum(70 <= score < 80 for score in scores),
            "below_70": sum(score < 70 for score in scores),
        }

        summary = UniverseAuditSummary(
            tickers_audited=len(reports),
            pass_count=pass_count,
            warn_count=warn_count,
            fail_count=fail_count,
            incomplete_count=incomplete_count,
            field_fail_counts=dict(field_fail_counts),
            field_warn_counts=dict(field_warn_counts),
            field_missing_counts=dict(field_missing_counts),
            source_health_alerts=source_health_alerts,
            average_score=float(mean(scores)) if scores else None,
            median_score=float(median(scores)) if scores else None,
            score_distribution=score_distribution,
            report_rows=sorted(report_rows, key=lambda row: (row["audit_quality_score"], row["ticker"])),
        )
        db.save_universe_audit_summary(summary, tickers=tickers, triggered_by=triggered_by)
        return summary

    def audit_field_distribution(self, field: AuditableField) -> FieldDistribution:
        """Return distribution statistics for a numeric field."""

        if field.is_string:
            raise ValueError(f"Field distribution is only available for numeric fields, not {field.value}")
        values = [(ticker, value) for ticker, value in db.list_field_values(field) if isinstance(value, (int, float))]
        if not values:
            raise ValueError(f"No stored numeric values found for {field.value}")
        numeric_values = [float(value) for _, value in values]
        series = pd.Series(numeric_values)
        p1 = float(series.quantile(0.01))
        p5 = float(series.quantile(0.05))
        p95 = float(series.quantile(0.95))
        p99 = float(series.quantile(0.99))
        outlier_tickers = [ticker for ticker, value in values if float(value) < p1 or float(value) > p99]
        return FieldDistribution(
            field=field,
            min=float(min(numeric_values)),
            max=float(max(numeric_values)),
            mean=float(mean(numeric_values)),
            median=float(median(numeric_values)),
            p1=p1,
            p5=p5,
            p95=p95,
            p99=p99,
            outlier_tickers=outlier_tickers,
        )

    def cross_check_sources(self, ticker: str) -> SourceComparisonReport:
        """Compare stored values against source snapshots and recommend sources."""

        normalized_ticker = ticker.strip().upper()
        record = db.get_fundamental(normalized_ticker, effective=True)
        if record is None:
            record = self.fetcher.fetch(normalized_ticker)
        bundle = self.fetcher.fetch_source_snapshots(normalized_ticker, refresh=True)
        by_source = bundle.by_source()
        rows: list[SourceComparisonRow] = []

        for field in AuditableField:
            yfinance_value = by_source.get(config.SOURCE_NAME_YFINANCE, None)
            nse_value = by_source.get(config.SOURCE_NAME_NSEPYTHON, None)
            alpha_value = by_source.get(config.SOURCE_NAME_ALPHA_VANTAGE, None)
            bse_value = by_source.get(config.SOURCE_NAME_BSE, None)
            stored_value = getattr(record, field.value)

            row = SourceComparisonRow(
                field_name=field,
                stored_value=stored_value,
                yfinance_value=yfinance_value.fields.get(field.value) if yfinance_value else None,
                nsepython_value=nse_value.fields.get(field.value) if nse_value else None,
                alpha_vantage_value=alpha_value.fields.get(field.value) if alpha_value else None,
                bse_value=bse_value.fields.get(field.value) if bse_value else None,
                status="PASS",
                recommended_source=self._recommended_source(field),
                details="Aligned across available sources",
            )
            if field is AuditableField.UPDATED_AT:
                row.details = "Stored freshness timestamp"
                rows.append(row)
                continue
            comparison_values = [row.yfinance_value, row.nsepython_value, row.alpha_vantage_value, row.bse_value]
            available = [value for value in comparison_values if value is not None]
            if not available:
                row.status = "MISSING"
                row.details = "No live source values available"
            elif all(isinstance(value, (int, float)) for value in available):
                baseline = float(available[0])
                max_delta = 0.0
                if baseline != 0:
                    for value in available[1:]:
                        max_delta = max(max_delta, abs(float(value) - baseline) / abs(baseline))
                if max_delta > 0.05:
                    row.status = "WARN"
                    row.details = f"Provider disagreement {max_delta:.2%}"
            else:
                unique_values = {str(value).strip().lower() for value in available}
                if len(unique_values) > 1:
                    row.status = "WARN"
                    row.details = "Provider values differ"
            rows.append(row)

        return SourceComparisonReport(ticker=normalized_ticker, generated_at=int(time.time()), rows=rows)

    def fix_data(
        self,
        ticker: str,
        field: AuditableField,
        value: Any,
        reason: str = "manual_fix",
    ) -> AuditReport:
        """Create an override for a field, invalidate cache, and rerun the audit."""

        normalized_ticker = ticker.strip().upper()
        self._validate_override_value(field, value)
        before_record = db.get_fundamental(normalized_ticker, effective=True)
        if before_record is None:
            raise ValueError(f"No stored data found for ticker {normalized_ticker}")
        override_result = db.add_override(normalized_ticker, field, value, reason)
        cache_manager.invalidate(normalized_ticker)
        db.log_audit_action(
            normalized_ticker,
            field,
            override_result["old_value"],
            override_result["new_value"],
            "fix_data",
            config.SOURCE_NAME_OVERRIDE,
        )
        return self.audit_ticker(normalized_ticker, triggered_by="fix_data")

    def _initialize_findings(self, record: FundamentalData) -> dict[AuditableField, dict[str, Any]]:
        """Create the default finding state for each auditable field."""

        findings: dict[AuditableField, dict[str, Any]] = {}
        for field in AuditableField:
            findings[field] = {
                "stored_value": getattr(record, field.value),
                "resolved_live_value": getattr(record, field.value),
                "source_name": record.source_metadata.get(field.value, config.SOURCE_NAME_STORED),
                "status": "PASS",
                "reason": "OK",
                "numeric_delta": None,
            }
        return findings

    def _apply_range_checks(self, record: FundamentalData, findings: dict[AuditableField, dict[str, Any]]) -> None:
        """Apply field-level range validations."""

        for field in AuditableField:
            value = getattr(record, field.value)
            if field is AuditableField.SECTOR:
                if value is None:
                    self._bump_finding(findings, field, "MISSING", "MISSING: sector is missing")
                elif str(value) not in config.NSE_SECTOR_LIST:
                    self._bump_finding(findings, field, "FAIL", "SECTOR_INVALID: sector not in NSE sector list")
                continue
            if field is AuditableField.UPDATED_AT:
                if value is None:
                    self._bump_finding(findings, field, "MISSING", "MISSING: updated_at is missing")
                continue
            if value is None:
                if field in {AuditableField.PEG_RATIO, AuditableField.PE_RATIO}:
                    self._bump_finding(findings, field, "PASS", "PASS: nullable for loss-making or unavailable case")
                else:
                    self._bump_finding(findings, field, "MISSING", f"MISSING: {field.value} is missing")
                continue

            if field is AuditableField.PIOTROSKI_SCORE:
                if not isinstance(value, int):
                    self._bump_finding(findings, field, "FAIL", "PIOTROSKI_TYPE_INVALID: score must be integer")
                    continue
                bounds = config.FIELD_AUDIT_RULES[field.value]
                if value < bounds["min"] or value > bounds["max"]:
                    self._bump_finding(findings, field, "FAIL", "PIOTROSKI_RANGE_INVALID: score outside 0-9")
                continue

            if field is AuditableField.PEG_RATIO:
                numeric_value = float(value)
                if numeric_value < 0:
                    self._bump_finding(findings, field, "FAIL", "INVALID_NEGATIVE_PEG: peg_ratio is negative")
                continue

            if field is AuditableField.PROMOTER_PCT:
                numeric_value = float(value)
                if numeric_value < 0 or numeric_value > 100:
                    self._bump_finding(findings, field, "FAIL", "PROMOTER_RANGE_INVALID: promoter_pct must be between 0 and 100")
                continue

            if field in {
                AuditableField.ROE_5Y,
                AuditableField.ROE_TTM,
                AuditableField.SALES_GROWTH_5Y,
                AuditableField.EPS_GROWTH_TTM,
                AuditableField.CFO_TO_PAT,
                AuditableField.DEBT_EQUITY,
                AuditableField.PE_RATIO,
            }:
                continue

            numeric_value = float(value)
            rules = config.FIELD_AUDIT_RULES.get(field.value)
            if not rules:
                continue
            min_exclusive = rules.get("min_exclusive")
            if min_exclusive is not None and numeric_value <= min_exclusive:
                self._bump_finding(findings, field, "FAIL", f"MIN_EXCLUSIVE_FAIL: must be > {min_exclusive}")
            if "min" in rules and numeric_value < rules["min"]:
                self._bump_finding(findings, field, "FAIL", f"MIN_FAIL: below minimum {rules['min']}")
            if "max" in rules and numeric_value > rules["max"]:
                self._bump_finding(findings, field, "FAIL", f"MAX_FAIL: above maximum {rules['max']}")
            if field not in {AuditableField.PROMOTER_PCT, AuditableField.CFO_TO_PAT, AuditableField.EPS_GROWTH_TTM} and "warn_high" in rules and numeric_value > rules["warn_high"]:
                self._bump_finding(findings, field, "WARN", f"WARN_HIGH: above warning threshold {rules['warn_high']}")
            if field not in {AuditableField.PROMOTER_PCT, AuditableField.CFO_TO_PAT} and "warn_low" in rules and numeric_value < rules["warn_low"]:
                self._bump_finding(findings, field, "WARN", f"WARN_LOW: below warning threshold {rules['warn_low']}")
            if "fail_high" in rules and numeric_value > rules["fail_high"]:
                self._bump_finding(findings, field, "FAIL", f"FAIL_HIGH: above fail threshold {rules['fail_high']}")

    def _apply_cross_checks(
        self,
        record: FundamentalData,
        live_price_value: Any,
        findings: dict[AuditableField, dict[str, Any]],
        red_flags: list[str],
    ) -> None:
        """Apply cross-field validation rules."""

        if record.price is not None and isinstance(live_price_value, (int, float)) and live_price_value:
            delta = abs(record.price - float(live_price_value)) / float(live_price_value)
            findings[AuditableField.PRICE]["resolved_live_value"] = float(live_price_value)
            findings[AuditableField.PRICE]["source_name"] = config.SOURCE_NAME_YFINANCE
            findings[AuditableField.PRICE]["numeric_delta"] = delta
            if delta > 0.02:
                self._bump_finding(findings, AuditableField.PRICE, "WARN", "PRICE_DRIFT: stored price differs from live price by >2%")

        if record.roe_5y is not None and record.roe_ttm is not None and abs(record.roe_5y - record.roe_ttm) > 0.20:
            reason = "ROE_INCONSISTENCY: abs(roe_5y - roe_ttm) > 0.20"
            self._bump_finding(findings, AuditableField.ROE_5Y, "WARN", reason)
            self._bump_finding(findings, AuditableField.ROE_TTM, "WARN", reason)

        if record.pledge_pct is not None and record.promoter_pct is not None and record.pledge_pct > 5 and record.promoter_pct < 40:
            red_flags.append("PROMOTER_PLEDGE_DOUBLE_RED_FLAG")
            self._bump_finding(findings, AuditableField.PLEDGE_PCT, "WARN", "PROMOTER_PLEDGE_DOUBLE_RED_FLAG: pledge > 5 and promoter < 40")
            self._bump_finding(findings, AuditableField.PROMOTER_PCT, "WARN", "PROMOTER_PLEDGE_DOUBLE_RED_FLAG: pledge > 5 and promoter < 40")

        recalculated_peg = None
        if record.pe_ratio is not None and record.eps_growth_ttm is not None and record.eps_growth_ttm > 0:
            recalculated_peg = record.pe_ratio / (record.eps_growth_ttm * 100)
            findings[AuditableField.PEG_RATIO]["resolved_live_value"] = recalculated_peg
            findings[AuditableField.PEG_RATIO]["source_name"] = "calculated"
            if record.peg_ratio is not None and recalculated_peg != 0:
                delta = abs(record.peg_ratio - recalculated_peg) / abs(recalculated_peg)
                findings[AuditableField.PEG_RATIO]["numeric_delta"] = delta
                if delta > 0.10:
                    self._bump_finding(findings, AuditableField.PEG_RATIO, "WARN", "PEG_CALCULATION_MISMATCH: stored PEG differs by >10%")

        age_seconds = int(time.time()) - int(record.updated_at)
        findings[AuditableField.UPDATED_AT]["resolved_live_value"] = int(record.updated_at)
        if age_seconds > config.FAIL_STALE_DAYS * 86400:
            self._bump_finding(findings, AuditableField.UPDATED_AT, "FAIL", "DATA_TOO_OLD: updated_at older than fail threshold")
        elif age_seconds > config.WARN_STALE_DAYS * 86400:
            self._bump_finding(findings, AuditableField.UPDATED_AT, "WARN", "STALE_DATA: updated_at older than warn threshold")

        for field in (AuditableField.PLEDGE_PCT, AuditableField.DEBT_EQUITY, AuditableField.PRICE, AuditableField.MARKET_CAP, AuditableField.AVG_VOLUME):
            if findings[field]["status"] == "FAIL":
                red_flags.append(f"{field.value.upper()}_FAIL")

        for issue in record.ingestion_issues:
            if issue.startswith("STALE_") and issue != "STALE_SOURCE_DATA":
                field_str = issue.replace("STALE_", "").lower()
                try:
                    field = AuditableField[field_str.upper()]
                    self._bump_finding(findings, field, "WARN", f"STALE_SOURCE_DATA: {field.value} is stale")
                except KeyError:
                    pass
            elif issue.startswith("CONFLICT_"):
                field_str = issue.replace("CONFLICT_", "").lower()
                try:
                    field = AuditableField[field_str.upper()]
                    self._bump_finding(findings, field, "WARN", f"CONFLICTING_SOURCES: high deviation across providers for {field.value}")
                except KeyError:
                    pass

        red_flags[:] = list(dict.fromkeys(red_flags))

    def _build_report(
        self,
        record: FundamentalData,
        findings: dict[AuditableField, dict[str, Any]],
        red_flags: list[str],
        suggested_fixes: list[str],
        triggered_by: str,
    ) -> AuditReport:
        """Construct an audit report from final finding states."""

        field_results = [
            FieldAudit(
                field_name=field,
                stored_value=payload["stored_value"],
                resolved_live_value=payload["resolved_live_value"],
                source_name=payload["source_name"],
                status=payload["status"],
                reason=payload["reason"],
                numeric_delta=payload["numeric_delta"],
            )
            for field, payload in findings.items()
        ]
        fail_count = sum(result.status == "FAIL" for result in field_results)
        warn_count = sum(result.status == "WARN" for result in field_results)
        missing_count = sum(result.status == "MISSING" for result in field_results)
        audit_quality_score = max(0.0, 100 - fail_count * config.FAIL_PENALTY - warn_count * config.WARN_PENALTY)

        if missing_count > 3:
            overall_status = "INCOMPLETE"
        elif fail_count:
            overall_status = "FAIL"
        elif warn_count:
            overall_status = "WARN"
        else:
            overall_status = "PASS"

        for result in field_results:
            if result.status == "MISSING":
                suggested_fixes.append(
                    f"Run `python sovereign-cli.py refetch --ticker {record.ticker}` to repopulate missing field `{result.field_name.value}`."
                )
            elif result.status in {"WARN", "FAIL"}:
                if "CONFLICTING_SOURCES" in result.reason:
                    suggested_fixes.append(
                        f"Review cross-provider values for `{result.field_name.value}` via `python sovereign-cli.py cross-check --ticker {record.ticker}`."
                    )
                else:
                    suggested_fixes.append(
                        f"Review `{result.field_name.value}` for {record.ticker}; if the stored value is confirmed wrong use `python sovereign-cli.py fix-data --ticker {record.ticker} --field {result.field_name.value} --value ...`."
                    )
        suggested_fixes = list(dict.fromkeys(suggested_fixes))
        return AuditReport(
            ticker=record.ticker,
            run_id=str(uuid.uuid4()),
            timestamp=int(time.time()),
            overall_status=overall_status,
            audit_quality_score=audit_quality_score,
            fail_count=fail_count,
            warn_count=warn_count,
            missing_count=missing_count,
            field_results=field_results,
            red_flags=red_flags,
            suggested_fixes=suggested_fixes,
            triggered_by=triggered_by,
        )

    def _bump_finding(
        self,
        findings: dict[AuditableField, dict[str, Any]],
        field: AuditableField,
        status: str,
        reason: str,
    ) -> None:
        """Promote a finding to a more severe status or append a reason."""

        payload = findings[field]
        if SEVERITY_ORDER[status] > SEVERITY_ORDER[payload["status"]]:
            payload["status"] = status
            payload["reason"] = reason
        elif payload["reason"] == "OK":
            payload["reason"] = reason
        elif reason not in payload["reason"]:
            payload["reason"] = f"{payload['reason']} | {reason}"

    def _recommended_source(self, field: AuditableField) -> str:
        """Return the recommended source for a field."""

        if field in {AuditableField.PRICE, AuditableField.MARKET_CAP, AuditableField.AVG_VOLUME, AuditableField.SECTOR, AuditableField.PE_RATIO}:
            return " -> ".join(config.SOURCE_PRECEDENCE["market"])
        if field in {AuditableField.PROMOTER_PCT, AuditableField.PLEDGE_PCT, AuditableField.FII_DELTA, AuditableField.DII_DELTA}:
            return " -> ".join(config.SOURCE_PRECEDENCE["ownership"])
        if field in {AuditableField.PEG_RATIO, AuditableField.PIOTROSKI_SCORE}:
            return "calculated"
        return " -> ".join(config.SOURCE_PRECEDENCE["fundamentals"])

    def _validate_override_value(self, field: AuditableField, value: Any) -> None:
        """Validate an override value against field expectations."""

        if field.is_string and not isinstance(value, str):
            raise ValueError(f"{field.value} expects a string value")
        if field.is_integer and not isinstance(value, int):
            raise ValueError(f"{field.value} expects an integer value")
        if field.is_numeric and not field.is_integer and not isinstance(value, (int, float)):
            raise ValueError(f"{field.value} expects a numeric value")


if __name__ == "__main__":
    auditor = DataAuditor()
    for ticker in ["RELIANCE", "TCS", "HDFCBANK"]:
        report = auditor.audit_ticker(ticker)
        print(f"{ticker}: {report.overall_status} | score={report.audit_quality_score}")
        for flag in report.red_flags:
            print(f"  RED FLAG: {flag}")
