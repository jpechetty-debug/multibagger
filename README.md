# Sovereign AI Trading Engine (Multibagger-Ultraversion)

An end-to-end, institutional-grade multi-strategy research and execution stack for Indian equities. Sovereign supports **Swing**, **Positional**, and **Multibagger** strategies driven by an automated, unified orchestrator, utilizing audit-first data ingestion, an advanced ML Meta-Model Ensemble, and a formal Feature Store with full data lineage.

---

## 🚀 Key Highlights

*   **7-Dimension "Multibagger" Scorer**
    A sophisticated fundamental scoring engine evaluating **Fundamentals, Earnings Revision, Momentum, Valuation, Ownership, Sector Strength, and Risk Metrics** with 6+ critical safety gates (Pledge, D/E, ROIC, Audit).
*   **Institutional ML & Model Registry**
    A robust **XGBoost Ensemble** pipeline with automated walk-forward validation and a formal **Model Registry** for experiment tracking, regression detection, and auto-promotion.
*   **Performance Tracking (OOS vs Live)**
    A closed-loop feedback system that monitors live trade outcomes against out-of-sample backtest expectations, triggering emergency retraining if model decay (Sharpe/Correlation) is detected.
*   **8-State Market Regime Tracker**
    High-fidelity state detection (Crisis, Panic, Bull, Quality, etc.) with volatility-aware hysteresis to recalibrate position sizing and strategy tilts.
*   **Advanced Risk Controls (VaR & Drawdown)**
    Real-time **95% Value-at-Risk (VaR)** computation and a **Drawdown Guard** with soft/hard breach rules to protect capital during market stress.
*   **Kelly Criterion Position Sizing**
    An institutional sizing overlay that adjusts MVO weights based on empirical win probability and win-loss ratios, explicitly preserving cash when edge is uncertain.
*   **High-Fidelity Backtesting**
    An event-driven backtest engine with realistic **Transaction Cost Modelling** (STT, GST, Brokerage, Slippage) for strategy validation.

---

## 📁 System Architecture

```text
Multibagger-Ultraversion/
├── app/                  # Streamlit UI dashboard and background APScheduler
├── data/                 # Data core and Feature Store:
│   ├── feature_store/    # Canonical spec (features_v1), Snapshotting, and Provenance
│   ├── db.py             # Main operational database (SQLAlchemy/SQLite)
│   └── fetcher.py        # Enriched fetchers with historical (4y) trajectories
├── engines/              # The core logic matrix:
│   ├── score_engine/     # 7-dimension weighted scoring & ML feature building
│   ├── ml/               # XGBoost Ensemble, Model Registry, Performance Tracker
│   ├── regime/           # 8-state Tracker (BULL, QUALITY, SIDEWAYS, PANIC, etc.)
│   ├── backtest/         # Event-driven backtest with realistic cost models
│   ├── portfolio/        # MVO Optimizer, Kelly Sizer (Post-MVO Overlay)
│   ├── risk/             # VaR Engine, Drawdown Guard, Factor Audit, and Vol Scaling
│   ├── multibagger/      # Specialized quality/TAM filters for long-term bets
│   └── pipeline.py       # Main E2E Orchestrator (Data -> Score -> Signal)
├── runtime/              # Active DBs (.db), model artifacts (.pkl), and backups
├── sovereign-cli.py      # Operations CLI (Health, Scans, Backups, Backtest, Retrain)
├── ticker_list.py        # Ticker universe management (307 tickers)
└── config.py             # Global constants and threshold definitions
```

---

## 🧠 Institutional ML & Governance

### 1. The Feature Store Contract
ML never reads raw database rows. All data flows through the **Feature Store Layer** with full cryptographic hash chains ensuring 100% reproducibility.

### 2. Model Registry & Promotion
- **Version Tracking**: Every model is logged with its dataset snapshot and performance metrics.
- **Regression Check**: New models are compared against the "Promoted" version; they are only activated if they show a performance delta > threshold.

### 3. Performance Tracker & Feedback Loop
The engine closes the loop by backfilling live outcomes (`record_outcome`) and comparing them to backtest (`OOS`) benchmarks:
- **Sharpe Decay**: Triggers ALERT if live Sharpe drops > 40% vs OOS.
- **Return Correlation**: Triggers RETRAIN if the correlation between predicted and actual returns falls below 0.30.

---

## 📈 Advanced Quant & Risk Management

### 1. "Triple-Gate" Exposure Reduction
The engine uses three independent, compounding gates to protect capital:
1.  **Regime Multiplier**: Reduces total exposure (e.g., to 25%) in BEAR or PANIC states.
2.  **Kelly Sizer**: Adjusts individual weights based on "Edge." If Kelly weights sum to < 1.0, the remainder stays as cash.
3.  **Volatility Scaling**: Inverses position size based on realised vol to equalise rupee-risk.

### 2. Risk Audit & VaR
- **Value-at-Risk (VaR)**: Real-time 95% historical VaR relative to portfolio equity.
- **Drawdown Guard**: Monitors live equity curve; triggers "Hard Breach" exit if max-DD threshold is crossed.
- **Factor Audit**: Comparison of portfolio factor loadings (Quality, Momentum, Size) against the Nifty 500.

---

## 🎯 Usage Guide

### Institutional Terminal (UI)
```bash
streamlit run app/streamlit_app.py
```

### Operations CLI (`sovereign-cli.py`)
- **Full Market Scan:** `python sovereign-cli.py scan --universe ALL`
- **High-Fidelity Backtest:** `python sovereign-cli.py backtest --strategy positional`
- **Model Retrain:** `python sovereign-cli.py ml-ops --retrain`
- **Health Check:** `python sovereign-cli.py health`

---

## 🧪 Verification & Auditing

Run individual module assessments to verify the architecture:
- `python ml_verify.py` (XGBoost Ensemble & SHAP)
- `python fs_verify.py` (Feature Store Hash Integrity)
- `python mb_verify.py` (Multibagger Safety Gates)
- `python quant_verify.py` (MVO, VaR, & Sector Constraints)

---

## 📝 Maintenance Notes
- **WAL Mode:** Databases use Write-Ahead Logging for high-concurrency access.
- **Auto-Fallback**: Data fetchers automatically fallback from Fyers to yfinance on failure.
- **Regime Resilience**: Logic handles invalid regime strings with safe defaults (QUALITY).
