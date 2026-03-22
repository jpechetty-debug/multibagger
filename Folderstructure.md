sovereign-engine/
│
├── app/
│   ├── streamlit_app.py         ← 10-page UI (Phase 5)
│   └── scheduler.py             ← APScheduler (Phase 5)
│
├── engines/
│   ├── pipeline.py              ← PipelineOrchestrator (Phase 5)
│   │
│   ├── analysis/
│   │   ├── momentum.py          ← Phase 3
│   │   ├── fundamentals.py      ← Phase 3
│   │   ├── ownership.py         ← Phase 3
│   │   ├── sector_rank.py       ← Phase 3
│   │   ├── liquidity.py         ← Phase 3
│   │   ├── earnings_revision.py ← Phase 3
│   │   └── risk_metrics.py      ← Phase 3
│   │
│   ├── score_engine/
│   │   ├── weights.py           ← Phase 3
│   │   ├── features.py          ← Phase 3
│   │   ├── regime.py            ← Phase 3
│   │   └── model.py             ← Phase 3
│   │
│   ├── valuation_engine.py      ← Phase 4
│   │
│   ├── risk/
│   │   ├── vix_filter.py        ← Phase 4
│   │   ├── position_sizing.py   ← Phase 4
│   │   ├── correlation.py       ← Phase 4
│   │   └── portfolio_limits.py  ← Phase 4
│   │
│   ├── signal_engine.py         ← Phase 4
│   ├── portfolio_engine.py      ← Phase 4
│   ├── portfolio_simulator.py   ← Phase 4
│   └── alert_engine.py          ← Phase 5
│
├── ml/
│   ├── train.py                 ← Phase 4
│   ├── predict.py               ← Phase 4
│   ├── features.py              ← Phase 4
│   └── registry.py              ← Phase 4
│
├── data/
│   ├── fetcher.py               ← Phase 2
│   ├── cache.py                 ← Phase 2
│   └── db.py                    ← Phase 1
│
├── models/
│   └── schemas.py               ← Phase 2
│
├── migrations/
│   ├── 001_init.sql             ← Phase 1
│   └── 002_ops.sql              ← Phase 1
│
├── data/                        ← created at runtime
│   ├── stocks.db
│   ├── pit_store.db
│   ├── cache.db
│   └── ops.db
│
├── sovereign-cli.py             ← Phase 1 + grows each phase
├── ticker_list.py               ← Phase 1
├── config.py                    ← Phase 1
├── requirements.txt             ← Phase 1
└── .env                         ← Phase 1