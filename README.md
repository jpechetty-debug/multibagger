# Sovereign AI Trading Engine (Multibagger-Ultraversion)

An end-to-end, institutional-grade multi-strategy research and execution stack for Indian equities. Sovereign supports **Swing**, **Positional**, and **Multibagger** strategies driven by an automated, unified orchestrator, utilizing audit-first data ingestion, an advanced ML Meta-Model Ensemble, and a formal Feature Store with full data lineage.

---

## ⚡ Key Highlights

*   **Industrial Precision UI (Upgraded)**
    A high-contrast, professional-grade terminal aesthetic with **Acid Green** accents and **Geist Mono** typography. Features a real-time **Ticker Tape**, unified VIX monitoring, and a premium search interface with keyboard shortcuts (`/`).
*   **Live Broker Integration (Fyers)**
    Transforms research into reality with a direct execution bridge to Fyers. Supports **MARKET, LIMIT, SL, and SL_M** orders with live audit logging and position confirmation.
*   **7-Dimension "Multibagger" Scorer**
    A sophisticated fundamental scoring engine evaluating **Fundamentals, Earnings Revision, Momentum, Valuation, Ownership, Sector Strength, and Risk Metrics** with 6+ critical safety gates.
*   **Advanced Risk Controls (VaR & Drawdown)**
    Real-time **Value-at-Risk (VaR)** computation and a multi-tiered **Drawdown Guard** (Soft, Medium, Hard, and Circuit Breaker) to protect capital during market stress.
*   **8-State Market Regime Tracker**
    High-fidelity state detection (Crisis, Panic, Bull, Quality, etc.) with volatility-aware hysteresis to recalibrate position sizing and strategy tilts.
*   **High-Fidelity Backtesting & Circuit Filters**
    Event-driven backtest engine with realistic **NSE Circuit Breaker** handling to ensure fills are only processed at tradeable prices.

---

## 📁 System Architecture

```text
Multibagger-Ultraversion/
├── app/                  # Streamlit UI (Industrial Precision) and APScheduler
├── data/                 # Data core and Feature Store:
│   ├── feature_store/    # Canonical spec, Snapshotting, and Provenance
│   ├── db.py             # Main operational database (SQLAlchemy/SQLite)
│   └── fetcher.py        # Enriched fetchers with historical (4y) trajectories
├── engines/              # The core logic matrix:
│   ├── execution/        # [NEW] Live trading via fyers_client.py
│   ├── monitoring/       # [NEW] Prometheus metrics.py for Grafana observability
│   ├── backtest/         # Event-driven backtest with CircuitFilter and cost models
│   ├── risk/             # [UPGRADED] Merged VaR, DrawdownGuard, and LiquidityFilter
│   ├── regime/           # 8-state Tracker (BULL, QUALITY, SIDEWAYS, etc.)
│   ├── score_engine/     # 7-dimension weighted scoring & ML feature building
│   ├── ml/               # XGBoost Ensemble, Model Registry, Performance Tracker
│   ├── pipeline.py       # Main E2E Orchestrator (Data -> Score -> Signal)
│   └── portfolio/        # MVO Optimizer, Kelly Sizer (Post-MVO Overlay)
├── runtime/              # Active DBs (.db), model artifacts (.pkl), and backups
├── sovereign-cli.py      # Operations CLI (Health, Scans, Backtest, Retrain, Metrics)
└── config.py             # Global constants and threshold definitions
```

---

## 🧠 Institutional ML & Governance

### 1. The Feature Store Contract
ML never reads raw database rows. All data flows through the **Feature Store Layer** with full cryptographic hash chains ensuring 100% reproducibility.

### 2. Performance Tracker & Feedback Loop
The engine closes the loop by comparing live outcomes against out-of-sample backtest expectations:
- **Sharpe Decay**: Triggers ALERT if live Sharpe drops > 40% vs OOS.
- **Return Correlation**: Triggers RETRAIN if the correlation between predicted and actual returns falls below 0.30.

---

## 📈 Advanced Quant & Risk Management

### 1. "Triple-Gate" Exposure Reduction
The engine uses three independent, compounding gates to protect capital:
1.  **Regime Multiplier**: Reduces total exposure in BEAR or PANIC states.
2.  **Kelly Sizer**: Adjusts weights based on "Edge" vs cash preservation.
3.  **Volatility Scaling**: Inverses position size based on realised vol.

### 2. Risk Audit & VaR
- **Value-at-Risk (VaR)**: Real-time 95% historical VaR relative to portfolio equity.
- **Drawdown Guard**: Monitors live equity curve; triggers tiered Stop rules or a **Circuit Breaker** (halt all new entries) if thresholds are crossed.
- **Liquidity Filter**: Ensures no position exceeds a fraction of the stock's Average Daily Volume (ADV).

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
- **Start Metrics Server:** `python engines/monitoring/metrics.py` (Default: port 9090)

---

## 🧪 Verification & Auditing
- `python ml_verify.py` (XGBoost Ensemble & SHAP)
- `python fs_verify.py` (Feature Store Hash Integrity)
- `python mb_verify.py` (Multibagger Safety Gates)
- `python quant_verify.py` (MVO, VaR, & Sector Constraints)

---

## 📝 Maintenance Notes
- **Fyers Integration**: Requires `FYERS_APP_ID` and `FYERS_ACCESS_TOKEN` in `.env`.
- **Metrics**: Exposes engine health as a `/metrics` HTTP endpoint for Grafana.
- **NSE Circuits**: Backtest engine uses `CircuitFilter` to simulate realistic fills.
