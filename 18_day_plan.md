# Execution Table

| Phase | Focus | Days | Key Deliverable | Exit Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Data Trust Layer | 4 | Provider-confidence scoring, stale/conflict quarantine, explicit skip reasons | STANDARD scan completes with auditable pass/skip/error reasons and no silent provider failures |
| **2** | Valuation & Signal Calibration | 5 | Sector-aware valuation bounds, outlier quarantine, more realistic fair values | Sample review across 50-100 tickers shows plausible fair values and signal reasons |
| **3** | Production Proof & Observability | 4 | Automated smoke checks, scheduler proof, backup/restore proof, ops visibility | Daily health/smoke path is one-command reproducible and failures are visible in CLI/UI |
| **4** | Strategy Validation & Execution Quality | 5 | Walk-forward validation, ML quality scorecard, portfolio lifecycle proof | Backtest, ML, and portfolio behavior are measured by regime and ready for paper-trading review |

**Total:** 18 days

## Phase 1 Workstreams
*   `data/fetcher.py`: add field-level trust weights, freshness penalties, source conflict detection, and stricter canonical resolution.
*   `engines/audit/data_auditor.py`: expand audits for stale data, provider disagreement, and missing critical fields; emit clearer remediation actions.
*   `engines/audit/pre_scan_gate.py` and `engines/pipeline.py`: quarantine bad records before scoring and persist explicit skip codes.
*   `data/db.py` and `models/schemas.py`: store trust score, provider disagreement metadata, and audit summaries for UI/CLI use.
*   `tests/test_audit_slice.py` and `tests/test_pipeline.py`: add coverage for stale-source, low-quality, and cross-provider-conflict cases.

## Phase 2 Workstreams
*   `engines/valuation_engine.py`: add sector templates, bounded assumptions, fallback ordering, and hard sanity checks for absurd outputs.
*   `engines/signal_engine.py` and `engines/score_engine/model.py`: separate valuation confidence from action confidence so weak valuations cannot dominate signals.
*   `engines/analysis/fundamentals.py` and `engines/analysis/sector_rank.py`: expose peer context for valuation comparison and sector baselines.
*   `app/streamlit_app.py`: show valuation confidence, outlier warnings, and “quarantined” fair values instead of misleading numbers.
*   `tests/test_phase4_modules.py` plus new calibration fixtures: validate DCF/EPS/Graham/PEG outputs across representative sectors.

## Phase 3 Workstreams
*   `sovereign-cli.py`: add a single smoke command or strengthen current operational commands to run health, scan, dups, backup, and restore checks in sequence.
*   `app/scheduler.py`: add explicit job-run result logging, duration tracking, and failure counters.
*   `engines/alert_engine.py` and `data/db.py`: promote operational events into structured alerts and durable run-history summaries.
*   `app/streamlit_app.py`: add an Ops page or strengthen Logs/Alerts pages with scheduler status, last backup, last restore test, duplicate counts, and recent failures.
*   `scripts/smoke_test_ui.ps1`, `scripts/smoke_test_ui.sh`, and `tests/test_phase5_pipeline.py`: formalize repeatable operational smoke coverage.

## Phase 4 Workstreams
*   `ml/train.py`, `engines/ml/trainer.py`, and `engines/ml/labeler.py`: measure training-set quality, label coverage, and model drift; stop treating “trained” as equivalent to “useful.”
*   `ml/features.py` and `ml/predict.py`: audit feature leakage risk and expose prediction confidence.
*   `backtest/swing_backtest.py` and `backtest/multibagger_simulator.py`: add walk-forward and regime-sliced reporting, not just aggregate outcomes.
*   `engines/portfolio_engine.py` and `engines/portfolio_simulator.py`: verify entries, exits, sizing, realized P&L, and cash handling under real signal sequences.
*   `tests/test_phase4_modules.py` and `tests/test_pipeline.py`: add end-to-end cases for paper-trading consistency, backtest realism, and regime-aware strategy metrics.

---

# 18-Day Plan

| Day | Phase | Tasks | Files | Acceptance Check |
| :--- | :--- | :--- | :--- | :--- |
| **1** | 1 | Baseline data-trust audit. Inventory missing/stale/conflicting fields across QUICK and STANDARD; define trust-score rules and skip codes. | `data/fetcher.py`, `engines/audit/data_auditor.py`, `models/schemas.py` | Audit output exists for both universes; top failure modes are documented; trust-score formula is fixed. |
| **2** | 1 | Implement field-level trust/freshness/conflict scoring in canonical resolution; persist source disagreement metadata. | `data/fetcher.py`, `models/schemas.py`, `data/db.py` | Fetched records now carry trust metadata; unit/integration tests cover stale and conflicting provider cases. |
| **3** | 1 | Tighten pre-scan gating and pipeline skip handling; make every skip/error reason explicit and queryable. | `engines/audit/pre_scan_gate.py`, `engines/pipeline.py`, `data/db.py` | A failed ticker never disappears silently; run history/logs show exact gate or provider reason. |
| **4** | 1 | Expose trust diagnostics in CLI/UI and lock tests. | `sovereign-cli.py`, `app/streamlit_app.py`, `tests/test_audit_slice.py`, `tests/test_pipeline.py` | STANDARD scan completes with auditable pass/skip/error counts and no unexplained failures. |
| **5** | 2 | Baseline valuation calibration. Measure current fair-value error/outliers across 50-100 names and identify broken formulas/assumptions. | `engines/valuation_engine.py`, `engines/analysis/fundamentals.py` | Sample sheet of current valuations exists; absurd outliers are categorized by root cause. |
| **6** | 2 | Add sector-aware valuation templates and bounded assumptions for DCF/EPS/Graham/PEG. | `engines/valuation_engine.py`, `config.py` | High-level sectors use different baselines; same input no longer produces obviously impossible fair values. |
| **7** | 2 | Add valuation sanity filters and quarantine logic; prevent poisoned fair values from driving signals. | `engines/valuation_engine.py`, `engines/signal_engine.py`, `engines/score_engine/model.py` | Outlier valuations are flagged or excluded; final action confidence degrades when valuation confidence is weak. |
| **8** | 2 | Add peer-context and confidence display in UI; show quarantined/low-confidence valuations clearly. | `app/streamlit_app.py`, `engines/analysis/sector_rank.py` | Fair Value page shows confidence and warnings instead of misleading numbers. |
| **9** | 2 | Freeze valuation regression tests and rerun sample scan. | `tests/test_phase4_modules.py`, `tests/test_pipeline.py` | Manual review over 50-100 names shows plausible ranges; no major outlier class remains unguarded. |
| **10**| 3 | Build one-command operational smoke flow covering health, scan, UI, dups, backup. | `sovereign-cli.py`, `scripts/smoke_test_ui.ps1`, `scripts/smoke_test_ui.sh` | A single smoke path can be run on demand and returns pass/fail by subsystem. |
| **11**| 3 | Improve scheduler observability with job durations, last result, consecutive failures, and next-run visibility. | `app/scheduler.py`, `data/db.py` | Each scheduler job records status, duration, and error payloads in ops storage. |
| **12**| 3 | Strengthen backup/restore proof and duplicate hygiene; record restore-check results in ops logs. | `data/db.py`, `sovereign-cli.py` | Backup, restore, and duplicate cleanup are reproducible and logged with explicit outcomes. |
| **13**| 3 | Add an operations surface in the UI for health, scheduler, backups, duplicate counts, and recent failures. | `app/streamlit_app.py`, `engines/alert_engine.py` | UI exposes current operational state without opening the DB manually. |
| **14**| 4 | Audit ML data quality: label coverage, leakage risk, feature availability, and regime distribution. | `ml/train.py`, `engines/ml/labeler.py`, `ml/features.py` | ML report exists with usable sample count, label null rate, and leakage review. |
| **15**| 4 | Retrain with explicit metrics capture and active-model promotion rules; reject weak models. | `ml/train.py`, `ml/registry.py`, `engines/ml/trainer.py` | Training outputs metrics and only promotes a model if it beats the current baseline. |
| **16**| 4 | Upgrade backtests to walk-forward and regime-sliced reporting for swing and multibagger strategies. | `backtest/swing_backtest.py`, `backtest/multibagger_simulator.py` | Backtest results are broken down by period and regime, not just aggregate return. |
| **17**| 4 | Validate portfolio lifecycle: entries, exits, stop-loss handling, realized P&L, sizing, and cash consistency. | `engines/portfolio_engine.py`, `engines/portfolio_simulator.py` | Portfolio state remains internally consistent through buy/sell/rebalance scenarios. |
| **18**| 4 | Final integrated proof: run smoke flow, STANDARD scan, model check, backtest summary, and paper-trading readiness review. | `tests/test_pipeline.py`, `tests/test_phase5_pipeline.py`, `README.md` | Final sign-off pack exists: operational status, valuation sanity, ML metrics, backtest summary, and known-risk list. |

## Phase Gates
*   **Phase 1 complete on Day 4** if scans are auditable and bad data is explicitly gated.
*   **Phase 2 complete on Day 9** if fair values are plausible and outliers are quarantined.
*   **Phase 3 complete on Day 13** if ops health is visible and reproducible.
*   **Phase 4 complete on Day 18** if strategy quality is measured, not assumed.
