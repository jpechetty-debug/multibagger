# Sovereign AI Trading Engine (Multibagger-Ultraversion)

An end-to-end, institutional-grade multi-strategy research and execution stack for Indian equities. Sovereign supports **Swing**, **Positional**, and **Multibagger** strategies driven by an automated, unified orchestrator, utilizing audit-first data ingestion, an advanced ML Meta-Model Ensemble, and a formal Feature Store with full data lineage.

---

## ⚡ Key Highlights

*   **Industrial Precision UI (Upgraded)**
    A high-contrast terminal aesthetic with **Acid Green** accents and **Geist Mono** typography. Features a real-time **Ticker Tape**, unified VIX monitoring, and a premium search interface with keyboard shortcuts (`/`).
*   **NSE Cycle Detector (Phase 2)**
    A new intelligence layer identifying the 4 phases of the NSE economic cycle: **Recovery, Expansion, Peak, and Contraction**. Automatically biases sector weights to capture rotation alpha.
*   **ML Model Guard (Phase 2)**
    Automatic reliability protection. If ML model decay (AUC/Correlation) is detected, the engine transparently falls back to rule-based scoring until a fresh model is promoted.
*   **Live Broker Integration (Fyers)**
    Direct execution bridge to Fyers. Supports **MARKET, LIMIT, SL, and SL_M** orders with live audit logging and position confirmation.
*   **Advanced Risk Controls (VaR & Drawdown)**
    Real-time **Value-at-Risk (VaR)** and a multi-tiered **Drawdown Guard** (Soft, Medium, Hard, and Circuit Breaker) to protect capital during market stress.
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
│   ├── analysis/         # [NEW] NSE CycleDetector and sector rotation logic
│   ├── ml/               # [UPGRADED] ModelGuard, ModelRegistry, and DecayTracker
│   ├── execution/        # Live trading via fyers_client.py
│   ├── monitoring/       # Prometheus metrics.py for Grafana observability
│   ├── backtest/         # Event-driven backtest with CircuitFilter and cost models
│   ├── regime/           # 8-state Tracker (BULL, QUALITY, SIDEWAYS, etc.)
│   ├── score_engine/     # [REFINED] ScoreEngine with ModelGuard integration
│   ├── pipeline.py       # Main E2E Orchestrator (Data -> Score -> Signal)
│   └── portfolio/        # MVO Optimizer, Kelly Sizer (Post-MVO Overlay)
├── runtime/              # Active DBs (.db), model artifacts (.pkl), and backups
├── sovereign-cli.py      # Operations CLI (Health, Scans, Backtest, Retrain, Metrics)
└── config.py             # Global constants and threshold definitions
```

---

## 🧠 Institutional ML & Governance

### 1. Model Guarding & Fallback
The engine uses a **ModelGuard** singleton to monitor the health of the active XGBoost ensemble. If the live performance (AUC) drops > 5% below the out-of-sample baseline, the system automatically activates a rule-based fallback mode to protect the portfolio.

### 2. Performance Feedback Loop
The engine closes the loop by comparing live outcomes against OOS expectations:
- **Sharpe Decay**: Triggers alert if live Sharpe drops > 40% vs OOS.
- **Correlation**: Triggers emergency retrain if prediction correlation falls below 0.30.

---

## 📈 Advanced Quant & Risk Management

### 1. NSE Cycle awareness
The **CycleDetector** biases position sizing by mapping ticker sectors to the current market phase. 
- **Recovery**: Biases toward Financials and Industrials.
- **Contraction**: Biases toward FMCG, Healthcare, and Defensive IT.

### 2. Triple-Gate Risk Audit
1.  **Regime Multiplier**: Total exposure reduction based on volatility state.
2.  **Drawdown Guard**: tiered exposure cuts (Hard Breach = exit all).
3.  **Liquidity Filter**: Position caps based on Average Daily Volume (ADV).

---

## 🎯 Usage Guide

### Institutional Terminal (UI)
```bash
streamlit run app/streamlit_app.py
```

### Operations CLI (`sovereign-cli.py`)
- **Full Market Scan:** `python sovereign-cli.py scan --universe ALL`
- **High-Fidelity Backtest:** `python sovereign-cli.py backtest --strategy positional`
- **Model Health:** `python sovereign-cli.py ml-ops --retrain`
- **Health Check:** `python sovereign-cli.py health`
- **Metrics Endpoint:** `python engines/monitoring/metrics.py` (Exposes port 9090)

---

## 🧪 Verification & Maintenance
- `python ml_verify.py` (ML Guard & Ensemble)
- `python fs_verify.py` (Feature Store Integrity)
- `python engines/analysis/cycle_detector.py` (Cycle Detector test run)
- **NSE Circuits**: Backtest engine uses `CircuitFilter` to simulate realistic fills.
- **Metrics**: Exposes engine health as a `/metrics` HTTP endpoint for Grafana.
