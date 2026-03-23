# Sovereign AI Trading Engine (Multibagger-Ultraversion)

An end-to-end, institutional-grade multi-strategy research and execution stack for Indian equities. Sovereign supports **Swing**, **Positional**, and **Multibagger** strategies driven by an automated, unified orchestrator, utilizing audit-first data ingestion, an advanced ML Meta-Model Ensemble, and a formal Feature Store with full data lineage.

---

## ⚡ Key Highlights

*   **Industrial Precision UI (Upgraded)**
    A high-contrast terminal aesthetic with **Acid Green** accents and **Geist Mono** typography. Features a real-time **Ticker Tape**, unified VIX monitoring, and a premium search interface with keyboard shortcuts (`/`).
*   **8-State Regime Tracker (Upgraded)**
    A new intelligence layer identifying 8 distinct market states (BULL, BEAR, SIDEWAYS, QUALITY, etc.). Uses live **Nifty 50 (^NSEI)** data for volatility regime classification, replacing single-stock proxies for cleaner signals.
*   **NSE Cycle Detector (Phase 2)**
    Identifies the 4 phases of the NSE economic cycle: **Recovery, Expansion, Peak, and Contraction**. Automatically biases sector weights to capture rotation alpha.
*   **ML Model Guard (Phase 2)**
    Automatic reliability protection. If ML model decay (AUC/Correlation) is detected, the engine transparently falls back to rule-based scoring until a fresh model is promoted.
*   **Institutional ML Validation**
    Expanding-window walk-forward validation with a **21-calendar-day gap** to prevent near-term momentum leakage. Uses **predict_proba** for continuous meta-model scoring.
*   **Advanced Risk Controls (VaR & Drawdown)**
    Real-time **Value-at-Risk (VaR)** and a multi-tiered **Drawdown Guard** (Soft, Medium, Hard, and Circuit Breaker) protecting capital in `advanced_risk.py`.

---

## 📁 System Architecture

```text
Multibagger-Ultraversion/
├── app/                  # Streamlit UI (Industrial Precision) and APScheduler
├── data/                 # Data core and Feature Store:
...
├── engines/              # The core logic matrix:
│   ├── analysis/         # [NEW] NSE CycleDetector and sector rotation logic
│   ├── ml/               # ModelGuard, ModelRegistry, and SovereignEnsemble
│   ├── execution/        # Live trading via fyers_client.py
│   ├── monitoring/       # Prometheus metrics.py for Grafana observability
│   ├── backtest/         # Event-driven backtest with CircuitFilter and cost models
│   ├── regime/           # [UPGRADED] 8-state Tracker with ^NSEI live fetch
│   ├── score_engine/     # [REFINED] ScoreEngine with ModelGuard and proba scoring
│   ├── pipeline.py       # Main E2E Orchestrator (Data -> Score -> Signal)
│   └── portfolio/        # MVO Optimizer, Kelly Sizer (Post-MVO Overlay)
├── runtime/              # Active DBs (.db), model artifacts (.pkl), and backups
├── sovereign-cli.py      # Operations CLI (Health, Scans, Backtest, Retrain, Metrics)
└── config.py             # Global constants and sector-aware PE templates
```

---

## 🧠 Institutional ML & Governance

### 1. Model Guarding & Fallback
The engine uses a **ModelGuard** singleton to monitor the health of the active XGBoost ensemble. If the live performance (AUC) drops > 5% below the out-of-sample baseline, the system automatically activates a rule-based fallback mode to protect the portfolio.

### 2. Honest Validation & Probabilistic Scoring
The engine enforces a **21-day calendar gap** between training and validation blocks to eliminate "leakage" from mid-cap momentum. The meta-model produces **continuous value scores (0-100)** via `predict_proba`, allowing for finer granularity in position sizing than binary classification.

---

## 📈 Advanced Quant & Risk Management

### 1. NSE Cycle awareness
The **CycleDetector** biases position sizing by mapping ticker sectors to the current market phase (Recovery, Expansion, Peak, Contraction).

### 2. Triple-Gate Risk Audit
1.  **Regime Multiplier**: Total exposure reduction based on the 8-state volatility tracker.
2.  **Drawdown Guard**: Tiered exposure cuts in `advanced_risk.py` (Hard Breach = exit all).
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

---

## 🧪 Verification & Maintenance
- `python v1_verify.py` (ML Guard & ScoreEngine integration)
- `python engines/analysis/cycle_detector.py` (Cycle Detector test run)
- `python engines/monitoring/metrics.py` (Exposes port 9090)
- **NSE Circuits**: Backtest engine uses `CircuitFilter` to simulate realistic fills.
