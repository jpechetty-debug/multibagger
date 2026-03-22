"""Sovereign Engine — Full Streamlit UI. Redesigned build."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path
from urllib.parse import urlencode
import asyncio
import math
import re
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import config
from data.db import db
from engines.analysis._common import load_benchmark_history, load_price_history
from engines.pipeline import PipelineOrchestrator
from engines.portfolio_engine import PortfolioEngine
from engines.portfolio_simulator import PortfolioSimulator
from models.schemas import SignalResult
from ticker_list import get_universe


st.set_page_config(
    page_title="Sovereign Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)
pio.templates.default = "plotly_dark"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_ORDER = {"BUY": 0, "WATCH": 1, "WEAK": 2, "REJECT": 3}
ACTION_BADGE_CLASS = {
    "BUY": "badge-buy",
    "WATCH": "badge-watch",
    "WEAK": "badge-weak",
    "REJECT": "badge-weak",
}
ACTION_ALERT_CLASS = {"BUY": "buy", "WATCH": "fv", "WEAK": "risk", "REJECT": "risk"}
RUN_SUMMARY_PATTERN = re.compile(
    r"Processed\s+(?P<processed>\d+)/(?P<requested>\d+)\s+tickers;\s+"
    r"(?P<buy>\d+)\s+BUY,\s+(?P<watch>\d+)\s+WATCH,\s+"
    r"(?P<weak>\d+)\s+WEAK,\s+(?P<reject>\d+)\s+REJECT,\s+"
    r"(?P<skip>\d+)\s+SKIP,\s+(?P<error>\d+)\s+ERROR"
)
NAV_ITEMS = [
    ("Dashboard", "Dashboard"),
    ("Swing Trades", "Swing"),
    ("Positionals", "BUY Signals"),
    ("Multibagger Hunt", "Multibaggers"),
    ("Fair Value", "Fair Value"),
    ("Sector Rank", "Sector Rank"),
    ("Ownership", "Ownership"),
    ("Risk Monitor", "Risk"),
    ("Portfolio", "Portfolio"),
    ("Backtest", "Backtest"),
    ("Alerts Feed", "Alerts"),
    ("Logs", "Logs"),
]

REGIME_SCORE_THRESHOLD = {"BULL": 75, "QUALITY": 72, "SIDEWAYS": 65, "BEAR": 55}
REGIME_SENTIMENT = {
    "BULL": ("BULLISH", "tag-green"),
    "QUALITY": ("BULLISH", "tag-green"),
    "SIDEWAYS": ("NEUTRAL", "tag-amber"),
    "BEAR": ("BEARISH", "tag-red"),
}


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_portfolio_engine() -> PortfolioEngine:
    return PortfolioEngine()


@st.cache_resource
def get_portfolio_simulator() -> PortfolioSimulator:
    return PortfolioSimulator()


@st.cache_data(ttl=300)
def load_scheduler_jobs() -> pd.DataFrame:
    """Load scheduler job execution history."""
    with db.connection("ops") as conn:
        df = pd.read_sql("SELECT * FROM scheduler_jobs ORDER BY finished_at DESC", conn)
    if not df.empty:
        df["started_at"] = pd.to_datetime(df["started_at"], unit="s")
        df["finished_at"] = pd.to_datetime(df["finished_at"], unit="s")
    return df


@st.cache_data(ttl=300)
def load_backup_history() -> pd.DataFrame:
    """Load database backup history."""
    with db.connection("ops") as conn:
        df = pd.read_sql("SELECT * FROM backups ORDER BY timestamp DESC", conn)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


@st.cache_resource
def get_pipeline_orchestrator() -> PipelineOrchestrator:
    return PipelineOrchestrator()


@st.cache_data(ttl=600)
def load_core_dataset() -> pd.DataFrame:
    try:
        fundamentals = pd.DataFrame([record.model_dump() for record in db.list_fundamentals(effective=True)])
        scores = pd.DataFrame(db.list_scores())
        signals = pd.DataFrame(db.list_signals())
        valuations = pd.DataFrame(db.list_valuations())
    except Exception:
        return pd.DataFrame()
    frame = fundamentals.copy()
    for extra in (scores, signals, valuations):
        if not extra.empty:
            frame = frame.merge(extra, on="ticker", how="left", suffixes=("", "_dup"))
            dup_cols = [c for c in frame.columns if c.endswith("_dup")]
            if dup_cols:
                frame = frame.drop(columns=dup_cols)
    if frame.empty:
        return pd.DataFrame(columns=[
            "ticker", "company_name", "sector", "price", "market_cap",
            "action", "confidence_score", "total_score", "reason_code",
            "fair_value", "dcf_value", "eps_value", "graham_value", "peg_value",
            "margin_of_safety_pct", "undervalued", "upside_pct", "generated_at"
        ])
    for column in (
        "action", "confidence_score", "total_score", "reason_code",
        "fair_value", "dcf_value", "eps_value", "graham_value", "peg_value",
        "margin_of_safety_pct", "undervalued",
    ):
        if column not in frame.columns:
            frame[column] = None
    if "fair_value" in frame.columns and "price" in frame.columns:
        valid = frame["fair_value"].notna() & frame["price"].notna() & (frame["price"] > 0)
        frame["upside_pct"] = ((frame["fair_value"] / frame["price"]) - 1.0).where(valid)
    return frame.sort_values(
        by=["action", "confidence_score"], ascending=[True, False], na_position="last"
    )


@st.cache_data(ttl=600)
def load_analysis_dataset(analysis_type: str) -> pd.DataFrame:
    try:
        rows = db.list_analysis_snapshots(analysis_type)
    except Exception:
        return pd.DataFrame()
    flattened: list[dict[str, object]] = []
    for row in rows:
        payload = dict(row["payload"])
        payload["ticker"] = row["ticker"]
        payload["as_of"] = row["as_of"]
        flattened.append(payload)
    return pd.DataFrame(flattened)


@st.cache_data(ttl=600)
def load_market_snapshot(snapshot_type: str) -> dict | None:
    try:
        snapshot = db.get_latest_market_snapshot(snapshot_type)
    except Exception:
        return None
    return snapshot if snapshot is None else dict(snapshot)


@st.cache_data(ttl=600)
def load_alert_logs() -> pd.DataFrame:
    try:
        return pd.DataFrame(db.list_logs(limit=50, component_prefix="engines.alert_engine"))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_run_history() -> pd.DataFrame:
    try:
        return pd.DataFrame(db.list_run_history(limit=50))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_model_versions() -> pd.DataFrame:
    try:
        return pd.DataFrame(db.list_model_versions(limit=50))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_error_logs() -> pd.DataFrame:
    try:
        return pd.DataFrame(db.list_logs(limit=100, level="ERROR"))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_latest_audit_rows() -> pd.DataFrame:
    try:
        return pd.DataFrame(db.list_latest_audit_rows())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_universe_audit_runs() -> pd.DataFrame:
    try:
        return pd.DataFrame(db.list_universe_audit_runs(limit=20))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_swing_dataset() -> pd.DataFrame:
    try:
        return pd.DataFrame(db.list_swing_signals())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_multibagger_dataset() -> pd.DataFrame:
    try:
        return pd.DataFrame(db.list_multibagger_candidates())
    except Exception:
        return pd.DataFrame()


def load_portfolio_transactions() -> pd.DataFrame:
    try:
        return pd.DataFrame(db.list_portfolio_transactions(limit=100))
    except Exception:
        return pd.DataFrame()


def invalidate_cached_views() -> None:
    load_core_dataset.clear()
    load_analysis_dataset.clear()
    load_market_snapshot.clear()
    load_swing_dataset.clear()
    load_multibagger_dataset.clear()
    load_alert_logs.clear()
    load_run_history.clear()
    load_model_versions.clear()
    load_error_logs.clear()
    load_latest_audit_rows.clear()
    load_universe_audit_runs.clear()
    load_portfolio_transactions.clear()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalize_page_name(page_name: object) -> str:
    if isinstance(page_name, list):
        page_name = page_name[0] if page_name else "Dashboard"
    if not isinstance(page_name, str):
        return "Dashboard"
    valid_pages = {page for page, _ in NAV_ITEMS}
    return page_name if page_name in valid_pages else "Dashboard"


def query_param_value(name: str, default: str = "") -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value)


def build_href(page_name: str, run_scan: bool = False) -> str:
    params = {"page": page_name}
    if run_scan:
        params["run_scan"] = "1"
    return f"?{urlencode(params)}"


def clamp(value: float | int | None, lower: float, upper: float) -> float:
    if value is None:
        return lower
    return max(lower, min(float(value), upper))


def normalize_percent_value(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if abs(number) <= 1.0:
        return number * 100.0
    return number


def format_timestamp(ts: object, with_date: bool = True) -> str:
    if ts is None or pd.isna(ts):
        return "n/a"
    moment = datetime.fromtimestamp(int(float(ts)))
    return moment.strftime("%d %b %Y, %I:%M %p" if with_date else "%I:%M %p")


def format_currency(value: object, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    number = float(value)
    if number <= 0:
        return "n/a"
    if decimals <= 0:
        return f"₹{number:,.0f}"
    return f"₹{number:,.{decimals}f}"


def format_decimal(value: object, decimals: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.{decimals}f}{suffix}"


def format_count(value: object) -> str:
    if value is None or pd.isna(value):
        return "0"
    return f"{int(float(value)):,}"


def format_ratio_percent(value: object, decimals: int = 1, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    number = float(value) * 100.0
    if not math.isfinite(number):
        return "n/a"
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number:.{decimals}f}%"


def format_percent_like(value: object, decimals: int = 1, signed: bool = False) -> str:
    normalized = normalize_percent_value(value)
    if normalized is None:
        return "n/a"
    prefix = "+" if signed and normalized > 0 else ""
    return f"{prefix}{normalized:.{decimals}f}%"


def html_escape(value: object) -> str:
    return escape("" if value is None else str(value))


def short_sector_name(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    name = str(value)
    replacements = {
        "Financial Services": "Financials",
        "Fast Moving Consumer Goods": "FMCG",
        "Oil Gas and Consumable Fuels": "Energy",
        "Telecommunication": "Telecom",
        "Consumer Durables": "Consumer",
        "Information Technology": "IT",
    }
    return replacements.get(name, name)


def action_sort_value(action: object) -> int:
    return ACTION_ORDER.get(str(action), 9)


def badge_class_for_action(action: object) -> str:
    return ACTION_BADGE_CLASS.get(str(action), "badge-weak")


def alert_class_for_action(action: object) -> str:
    return ACTION_ALERT_CLASS.get(str(action), "risk")


def score_fill_color(score: object, threshold: int = 75) -> str:
    if score is None or pd.isna(score):
        return "var(--c-muted)"
    number = float(score)
    if number >= threshold:
        return "var(--c-emerald)"
    if number >= threshold - 10:
        return "var(--c-amber)"
    return "var(--c-rose)"


def regime_display_color(regime: str | None) -> str:
    palette = {
        "BULL": "var(--c-emerald)",
        "QUALITY": "var(--c-sky)",
        "SIDEWAYS": "var(--c-amber)",
        "BEAR": "var(--c-rose)",
    }
    return palette.get((regime or "").upper(), "var(--c-violet)")


def vix_display_class(vix_value: object) -> str:
    if vix_value is None or pd.isna(vix_value):
        return "vix-neutral"
    value = float(vix_value)
    if value < 15:
        return "vix-safe"
    if value < 25:
        return "vix-caution"
    return "vix-risk"


def position_sizing_label(vix_state: str | None) -> str:
    mapping = {
        "NORMAL": "Normal (1.0×)",
        "HALF": "Half size (0.5×)",
        "HALT": "Kill switch active",
    }
    return mapping.get((vix_state or "").upper(), "Adaptive")


def parse_run_summary(summary: object) -> dict[str, int] | None:
    if not isinstance(summary, str):
        return None
    match = RUN_SUMMARY_PATTERN.search(summary)
    if not match:
        return None
    return {key: int(value) for key, value in match.groupdict().items()}


def scan_history_frame(run_history: pd.DataFrame) -> pd.DataFrame:
    if run_history.empty:
        return pd.DataFrame()
    frame = run_history[run_history["command_name"].isin(["scan", "pipeline.run"])].copy()
    if frame.empty:
        return frame
    return frame.sort_values(
        by=["finished_at", "started_at"], ascending=False, na_position="last"
    )


def latest_scan_snapshot(
    run_history: pd.DataFrame,
) -> tuple[pd.Series | None, dict[str, int] | None, dict[str, int] | None]:
    scan_frame = scan_history_frame(run_history)
    if scan_frame.empty:
        return None, None, None
    latest_row = scan_frame.iloc[0]
    latest_counts = parse_run_summary(latest_row.get("summary"))
    previous_counts = None
    for _, row in scan_frame.iloc[1:].iterrows():
        previous_counts = parse_run_summary(row.get("summary"))
        if previous_counts:
            break
    return latest_row, latest_counts, previous_counts


def build_dashboard_frame(core_frame: pd.DataFrame) -> pd.DataFrame:
    if core_frame.empty:
        return core_frame.copy()
    dashboard_frame = core_frame.copy()
    for analysis_type, columns in {
        "momentum": ["price_return_3m", "relative_strength_3m", "price_vs_50dma_pct", "above_50dma"],
        "sector_rank": ["sector_rank", "rank_percentile", "top_3"],
        "ownership": ["promoter_pct", "pledge_pct", "fii_delta", "dii_delta", "ownership_clean"],
    }.items():
        extra = load_analysis_dataset(analysis_type)
        if extra.empty:
            continue
        keep_columns = ["ticker"] + [column for column in columns if column in extra.columns]
        dashboard_frame = dashboard_frame.merge(
            extra[keep_columns], on="ticker", how="left", suffixes=("", f"_{analysis_type}")
        )
    if "company_name" not in dashboard_frame.columns:
        dashboard_frame["company_name"] = dashboard_frame["ticker"]
    dashboard_frame["company_name"] = dashboard_frame["company_name"].fillna(dashboard_frame["ticker"])
    dashboard_frame["sector"] = dashboard_frame["sector"].fillna("Unknown")
    dashboard_frame["action_rank"] = dashboard_frame["action"].apply(action_sort_value)
    return dashboard_frame


def filter_dashboard_signals(
    frame: pd.DataFrame, search_term: str, selected_filter: str | None
) -> pd.DataFrame:
    filtered = frame.copy()
    if search_term.strip():
        needle = search_term.strip().lower()
        searchable = (
            filtered["ticker"].fillna("") + " "
            + filtered["company_name"].fillna("") + " "
            + filtered["sector"].fillna("")
        ).str.lower()
        filtered = filtered[searchable.str.contains(needle, regex=False)]
    if selected_filter and selected_filter != "All":
        if selected_filter == "Score >80":
            filtered = filtered[filtered["total_score"].fillna(0.0) >= 80.0]
        else:
            filtered = filtered[filtered["sector"].fillna("").eq(selected_filter)]
    return filtered.sort_values(
        by=["action_rank", "confidence_score", "total_score", "market_cap"],
        ascending=[True, False, False, False],
        na_position="last",
    )


def get_current_regime(regime_snapshot: dict | None) -> str:
    if not regime_snapshot:
        return "UNKNOWN"
    return str(regime_snapshot.get("payload", {}).get("regime", "UNKNOWN")).upper()


def regime_score_threshold(regime: str) -> int:
    return REGIME_SCORE_THRESHOLD.get(regime, 75)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def render_top_navigation(regime_snapshot: dict | None, vix_snapshot: dict | None) -> None:
    regime = get_current_regime(regime_snapshot)
    vix_p = (vix_snapshot or {}).get("payload", {})
    vix_v = vix_p.get("vix_value", 0.0)
    vix_s = str(vix_p.get("state", "NORMAL")).upper()
    vix_c = "vd-safe" if vix_v < 15 else "vd-warn" if vix_v < 25 else "vd-risk"

    tabs = ["Dashboard", "Swing", "BUY Signals", "WATCH List", "Sector Scan", "Alerts", "Logs", "Performance"]
    active = query_param_value("page", "Dashboard")

    q = {"page": "Dashboard"}
    nav_html = []
    for t in tabs:
        cls = "nav-tab active" if active == t else "nav-tab"
        q["page"] = t
        nav_html.append(f'<a class="{cls}" href="?{urlencode(q)}" target="_self">{t}</a>')

    st.markdown(f"""
<div class="sov-nav">
  <div class="nav-brand">
    <div class="nav-sigma">Σ</div>
    <div class="nav-wordmark">SOVEREIGN</div>
    <div class="nav-sub">AI Engine</div>
    <div class="nav-regime nr-{regime}">{regime}</div>
  </div>
  <div class="nav-links">{"".join(nav_html)}</div>
  <div class="nav-right">
    <div class="vix-pill">
      <div class="vix-dot {vix_c}"></div>
      <span class="vix-cap">VIX:</span><span class="vix-num">{vix_v:.1f}</span>
    </div>
    <a href="#" class="scan-cta" id="run_scan_trigger">SCAN MARKET</a>
  </div>
</div>
<div class="ticker-tape">
  <div class="ticker-inner">
    <div class="ticker-item"><span class="t-sym">NIFTY</span><span class="t-px">22,450.2</span><span class="t-up">+1.2%</span></div>
    <div class="ticker-item"><span class="t-sym">RELIANCE</span><span class="t-px">2,904.5</span><span class="t-up">+0.8%</span></div>
    <div class="ticker-item"><span class="t-sym">HDFCBANK</span><span class="t-px">1,452.1</span><span class="t-dn">-0.3%</span></div>
    <div class="ticker-item"><span class="t-sym">TCS</span><span class="t-px">4,120.9</span><span class="t-up">+2.1%</span></div>
    <div class="ticker-item"><span class="t-sym">INFY</span><span class="t-px">1,620.4</span><span class="t-dn">-1.5%</span></div>
    <div class="ticker-item"><span class="t-sym">NIFTY</span><span class="t-px">22,450.2</span><span class="t-up">+1.2%</span></div>
    <div class="ticker-item"><span class="t-sym">RELIANCE</span><span class="t-px">2,904.5</span><span class="t-up">+0.8%</span></div>
    <div class="ticker-item"><span class="t-sym">HDFCBANK</span><span class="t-px">1,452.1</span><span class="t-dn">-0.3%</span></div>
    <div class="ticker-item"><span class="t-sym">TCS</span><span class="t-px">4,120.9</span><span class="t-up">+2.1%</span></div>
    <div class="ticker-item"><span class="t-sym">INFY</span><span class="t-px">1,620.4</span><span class="t-dn">-1.5%</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------

def sparkline(ticker: str, positive: bool = True) -> str:
    c = "#00ffa3" if positive else "#ff3060"
    o = "0.15" if positive else "0.1"
    vals = [20, 45, 30, 60, 40, 75, 50, 90] if positive else [80, 60, 70, 40, 50, 25, 35, 15]
    pts = " ".join([f"{i*20},{100-v}" for i, v in enumerate(vals)])
    return f'<svg width="100" height="30" viewBox="0 0 140 100" preserveAspectRatio="none"><polyline points="{pts}" fill="none" stroke="{c}" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/><path d="M0 100 L{pts} L140 100 Z" fill="{c}" opacity="{o}"/></svg>'


# ---------------------------------------------------------------------------
# Sector heatmap
# ---------------------------------------------------------------------------

def render_sector_heatmap(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    sector_data = (
        frame.groupby("sector", dropna=False)
        .agg({"ticker": "count", "total_score": "mean"})
        .reset_index()
    )
    sector_data["sector"] = sector_data["sector"].fillna("Unknown")
    sector_data = sector_data.sort_values("ticker", ascending=False)
    fig = px.treemap(
        sector_data,
        path=[px.Constant("Market"), "sector"],
        values="ticker",
        color="total_score",
        color_continuous_scale=["#f43f5e", "#f59e0b", "#10b981"],
    )
    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'IBM Plex Mono', monospace", size=11, color="#94a3b8"),
        height=180,
        coloraxis_showscale=False,
    )
    fig.update_traces(
        textfont=dict(family="'IBM Plex Mono', monospace", size=11),
        marker_line_width=1,
        marker_line_color="rgba(0,0,0,0.3)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Valuation gauge
# ---------------------------------------------------------------------------

def render_valuation_gauge(price: float, fair_value: float, confidence: float | None = None) -> str:
    if not fair_value or fair_value <= 0 or not price or price <= 0:
        return ""
    ratio = price / fair_value
    pos = clamp(ratio * 50, 4, 96)
    color = "#10b981" if pos < 45 else "#f59e0b" if pos < 60 else "#f43f5e"
    
    conf_html = ""
    if confidence is not None:
        conf_color = "#10b981" if confidence >= 70 else "#f59e0b" if confidence >= 40 else "#f43f5e"
        conf_html = f'<div style="text-align: center; font-size: 11px; margin-top: 12px; color: {conf_color}; font-weight: 700;">VALUATION CONFIDENCE: {confidence:.0f}%</div>'

    return (
        f'<div class="val-gauge">'
        f'<div class="val-gauge-labels">'
        f'<span>Undervalued</span><span>Fair</span><span>Overvalued</span>'
        f'</div>'
        f'<div class="val-gauge-track">'
        f'<div class="val-gauge-fill" style="width:{pos:.0f}%"></div>'
        f'<div class="val-gauge-thumb" style="left:{pos:.0f}%;background:{color}"></div>'
        f'</div>'
        f'{conf_html}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Wealth checklist
# ---------------------------------------------------------------------------

def render_wealth_checklist(
    ticker: str,
    score: float,
    price: float,
    fair_value: float,
    regime: str = "UNKNOWN",
    valuation_confidence: float | None = None,
) -> None:
    quality = "Exceptional" if score > 80 else "High" if score > 65 else "Moderate"
    q_cls = "tag-emerald" if quality == "Exceptional" else "tag-sky" if quality == "High" else "tag-amber"

    if price > 0 and fair_value > 0:
        valuation = "Attractive" if price < fair_value * 0.85 else "Fair" if price < fair_value * 1.15 else "Stretched"
        v_cls = "tag-emerald" if valuation == "Attractive" else "tag-amber" if valuation == "Fair" else "tag-rose"
    else:
        valuation, v_cls = "Insufficient data", "tag-muted"

    growth = "High" if score > 65 else "Stable" if score > 40 else "Low"
    g_cls = "tag-emerald" if growth == "High" else "tag-sky"

    sentiment_label, s_cls = REGIME_SENTIMENT.get(regime, ("NEUTRAL", "tag-amber"))
    gauge_html = render_valuation_gauge(price, fair_value, valuation_confidence)

    st.markdown(
        f'<div class="focus-card">'
        f'<div class="focus-header">'
        f'<span class="focus-ticker">{html_escape(ticker)}</span>'
        f'<span class="focus-label">Investment Checklist</span>'
        f'</div>'
        f'{gauge_html}'
        f'<div class="checklist-grid">'
        f'<div class="checklist-item"><span class="ci-label">Quality Score</span><span class="tag {q_cls}">{quality}</span></div>'
        f'<div class="checklist-item"><span class="ci-label">Intrinsic Value</span><span class="tag {v_cls}">{valuation}</span></div>'
        f'<div class="checklist-item"><span class="ci-label">Growth</span><span class="tag {g_cls}">{growth}</span></div>'
        f'<div class="checklist-item"><span class="ci-label">Regime Sentiment</span><span class="tag {s_cls}">{sentiment_label}</span></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Stat cards
# ---------------------------------------------------------------------------

def render_stat_cards(
    frame: pd.DataFrame,
    swing_df: pd.DataFrame,
    mb_df: pd.DataFrame,
    regime_snapshot: dict | None,
    run_history: pd.DataFrame
) -> None:
    regime = get_current_regime(regime_snapshot)
    latest_row, latest_cts, _ = latest_scan_snapshot(run_history)
    scan_size = int(latest_cts.get("requested", 0)) if latest_cts else len(frame)
    avg_score = frame["total_score"].mean() if not frame.empty else 0.0

    cards = [
        {"acc":"var(--acid)", "lbl":"Market Regime", "val":regime, "sub":"Systemic bias", "cls":"sv-acid"},
        {"acc":"var(--acid)", "lbl":"Positional BUYs", "val":len(frame[frame["action"]=="BUY"]), "sub":"Core long setups", "cls":"sv-acid"},
        {"acc":"var(--sky)", "lbl":"Swing ENTRYs", "val":len(swing_df[swing_df["action"]=="ENTRY"]), "sub":"T+5 momentum", "cls":"sv-sky"},
        {"acc":"var(--gold)", "lbl":"Multibaggers", "val":len(mb_df), "sub":"High conviction", "cls":"sv-gold"},
        {"acc":"var(--rose)", "lbl":"Weak / Reject", "val":len(frame[frame["action"]=="WEAK"]), "sub":f"of {scan_size} scanned", "cls":"sv-rose"},
        {"acc":"var(--violet)", "lbl":"Strategy Score", "val":f"{avg_score:.1f}", "sub":"Avg across universe", "cls":"sv-violet"},
    ]

    cols = st.columns(6, gap="small")
    for i, (col, c) in enumerate(zip(cols, cards)):
        with col:
            st.markdown(f"""
<div class="stat-card" style="--accent:{c['acc']}">
  <div class="stat-label">{c['lbl']}</div>
  <div class="stat-val {c['cls']}">{c['val']}</div>
  <div class="stat-foot">
    <div class="stat-sub">{c['sub']}</div>
  </div>
</div>
""", unsafe_allow_html=True)


def render_notice_strip(frame: pd.DataFrame, run_history: pd.DataFrame) -> None:
    latest_row, latest_cts, _ = latest_scan_snapshot(run_history)
    buy_count = len(frame[frame["action"]=="BUY"]) if not frame.empty else 0
    if latest_row is None:
        msg = "TERMINAL_STARTUP_PHASE: No active scan history. Showing cached snapshots."
    elif buy_count == 0:
        msg = "REGIME_FILTER_ACTIVE: No high-conviction BUY signals detected for current regime."
    else:
        last = latest_row.get("finished_at", 0)
        hrs = (datetime.now() - datetime.fromtimestamp(int(float(last)))).total_seconds()/3600.0
        if hrs < 12: return
        msg = f"DATA_DECAY_WARNING: Current signal snapshot is {hrs:.1f}h old. Consider re-scanning."

    st.markdown(f'<div class="notice-strip"><span class="sb-dot"></span>{msg}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Signal table
# ---------------------------------------------------------------------------

def render_signal_table(frame: pd.DataFrame, regime: str = "UNKNOWN") -> None:
    if frame.empty:
        st.info("No signals match filters.")
        return

    rows_html = []
    for rank, (_, r) in enumerate(frame.head(25).iterrows(), 1):
        sym = r["ticker"]
        px = f"{r['price']:.2f}"
        up_val = r.get("upside_pct", 0)
        up_px = f"{up_val*100:+.1f}%"
        up_cls = "pos" if up_val > 0 else "neg"
        
        score = r.get("total_score", 0)
        sc_c = "#00ffa3" if score > 70 else "#ffb700" if score > 40 else "#ff3060"
        
        tags = []
        if score > 80: tags.append('<span class="rtag hq">HQ</span>')
        if r.get("volume_surge", 0) > 2: tags.append('<span class="rtag mo">MO</span>')

        rows_html.append(f"""
<div class="trow">
  <div class="rank-n">{rank}</div>
  <div>
    <a href="?ticker={sym}" target="_self" class="ticker-link"><span class="sym-lg">{sym}</span></a>
    <div class="tag-row">{"".join(tags)}</div>
  </div>
  <div class="px-cell">{px}</div>
  <div class="c">{sparkline(sym, up_val>0)}</div>
  <div class="up-cell {up_cls}">{up_px}</div>
  <div class="sc-wrap">
    <div class="sc-bar"><div class="sc-fill" style="width:{score}%;background:{sc_c}"></div></div>
    <div class="sc-num" style="color:{sc_c}">{int(score)}</div>
  </div>
  <div class="sec-cell">{short_sector_name(r['sector'])}</div>
  <div class="r"><span class="badge badge-{r['action'].lower()}">{r['action']}</span></div>
</div>
""")

    st.markdown(f"""
<div class="signal-tbl">
  <div class="thead">
    <div>#</div><div>Ticker</div><div class="r">Price</div><div class="c">Trend</div>
    <div class="r">Upside</div><div class="r">Score</div><div class="r">Sector</div><div class="r">Action</div>
  </div>
  {"".join(rows_html)}
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Regime card
# ---------------------------------------------------------------------------

def render_regime_card(regime_snapshot: dict | None, vix_snapshot: dict | None) -> None:
    regime = get_current_regime(regime_snapshot) or "SIDEWAYS"
    vix_p = (vix_snapshot or {}).get("payload", {})
    vix_v = vix_p.get("vix_value", 0.0)

    st.markdown(f"""
<div class="regime-card">
  <div class="rc-label">Market Regime</div>
  <div class="regime-word rw-{regime}">{regime}</div>
  <div class="regime-desc">Institutional bias is currently positioned for {regime.lower()} market conditions.</div>
  <div class="rule-list">
    <div class="rule-row">
      <span class="rule-lbl">India VIX</span>
      <span class="rv-{"ok" if vix_v < 18 else "warn"}">{vix_v:.2f}</span>
    </div>
    <div class="rule-row">
      <span class="rule-lbl">Position Sizing</span>
      <span class="rv-ok">100% Core</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


def render_sector_distribution(frame: pd.DataFrame) -> None:
    actionable = frame[frame["action"].isin(["BUY", "WATCH"])].copy()
    if actionable.empty: actionable = frame.copy()
    sector_counts = actionable.groupby("sector").size().reset_index(name="count").sort_values("count", ascending=False).head(5)
    
    rows = []
    for _, row in sector_counts.iterrows():
        rows.append(f"""
<div class="sector-row">
  <div class="sec-meta">
    <span class="sec-name">{short_sector_name(row['sector'])}</span>
    <span class="sec-count">{int(row['count'])}</span>
  </div>
  <div class="sec-bar-bg"><div class="sec-bar-fill" style="width:{row['count']/sector_counts['count'].max()*100}%"></div></div>
</div>
""")

    st.markdown(f"""
<div class="sector-scan">
  <div class="sec-hdr">Sector Exposure</div>
  {"".join(rows)}
</div>
""", unsafe_allow_html=True)


def build_ownership_flags(frame: pd.DataFrame) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    if frame.empty:
        return flags

    low_promoter = frame.dropna(subset=["promoter_pct"]).copy()
    if not low_promoter.empty:
        low_promoter["promoter_norm"] = low_promoter["promoter_pct"].apply(normalize_percent_value)
        for _, row in low_promoter.sort_values(by="promoter_norm", ascending=True, na_position="last").head(2).iterrows():
            if row.get("promoter_norm", 100.0) < 45.0:
                flags.append({
                    "kind": "warn", "ticker": html_escape(row["ticker"]),
                    "desc": "Low promoter holding", "value": format_percent_like(row.get("promoter_norm"))
                })

    pledged = frame.dropna(subset=["pledge_pct"]).copy()
    if not pledged.empty:
        pledged["pledge_norm"] = pledged["pledge_pct"].apply(normalize_percent_value)
        for _, row in pledged[pledged["pledge_norm"].fillna(0.0) > 10.0].sort_values(
            by="pledge_norm", ascending=False, na_position="last"
        ).head(1).iterrows():
            flags.append({
                "kind": "warn", "ticker": html_escape(row["ticker"]),
                "desc": "Pledge elevated", "value": format_percent_like(row.get("pledge_norm"))
            })

    fii_positive = frame.dropna(subset=["fii_delta"]).copy()
    if not fii_positive.empty:
        fii_positive["fii_norm"] = fii_positive["fii_delta"].apply(normalize_percent_value)
        for _, row in fii_positive.sort_values(by="fii_norm", ascending=False, na_position="last").head(1).iterrows():
            if row.get("fii_norm", 0.0) > 0.5:
                flags.append({
                    "kind": "good", "ticker": html_escape(row["ticker"]),
                    "desc": "FII accumulation", "value": format_percent_like(row.get("fii_norm"), signed=True)
                })

    if not flags:
        clean = frame.dropna(subset=["ownership_clean"]).copy()
        if not clean.empty:
            for _, row in clean.sort_values(by="total_score", ascending=False, na_position="last").head(2).iterrows():
                flags.append({
                    "kind": "good", "ticker": html_escape(row["ticker"]),
                    "desc": "Ownership clean", "value": format_decimal(row.get("total_score"), 0)
                })
    return flags[:3]


def render_ownership_flags(frame: pd.DataFrame) -> None:
    flags = build_ownership_flags(frame)
    if not flags:
        st.info("Ownership snapshots unavailable.")
        return
    warn_count = sum(flag["kind"] == "warn" for flag in flags)
    rows = []
    for flag in flags:
        cls = "flag-warn" if flag["kind"] == "warn" else "flag-good"
        rows.append(
            f'<div class="flag-row {cls}">'
            f'<div class="flag-main">'
            f'<div class="flag-ticker">{flag["ticker"]}</div>'
            f'<div class="flag-desc">{html_escape(flag["desc"])}</div>'
            f'</div>'
            f'<div class="flag-val">{html_escape(flag["value"])}</div>'
            f'</div>'
        )
    pill = (
        f'<span class="pill tag-rose">{warn_count} warnings</span>'
        if warn_count else '<span class="pill tag-emerald">clean tape</span>'
    )
    st.markdown(
        f'<div class="panel-card">'
        f'<div class="panel-hdr"><span class="panel-title">Ownership Signals</span>{pill}</div>'
        f'<div class="flags-list">{"".join(rows)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_vix_gauge(vix_snapshot: dict | None) -> None:
    payload = vix_snapshot["payload"] if vix_snapshot else {}
    value = payload.get("vix_value") or 0.0
    state = str(payload.get("state", "NORMAL")).upper()
    reason = payload.get("reason") or "Snapshot unavailable"

    gauge_min, gauge_max = 10.0, 40.0
    normalized = clamp((float(value) - gauge_min) / (gauge_max - gauge_min), 0.0, 1.0)
    angle = math.pi - normalized * math.pi
    px = 90.0 + 56.0 * math.cos(angle)
    py = 90.0 - 56.0 * math.sin(angle)

    if value < 15:
        color, sub = "#10b981", "Safe zone - normal risk"
    elif value < 25:
        color, sub = "#f59e0b", "Caution - review sizing"
    else:
        color, sub = "#f43f5e", "Stress - reduce exposure"

    st.markdown(
        f'<div class="panel-card">'
        f'<div class="panel-hdr">'
        f'<span class="panel-title">India VIX</span>'
        f'<span class="tag tag-sky">{html_escape(state.title())}</span>'
        f'</div>'
        f'<div class="gauge-center">'
        f'<svg width="180" height="100" viewBox="0 0 180 100">'
        f'<path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="12" stroke-linecap="round"/>'
        f'<path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="url(#vix-grad)" stroke-width="12" stroke-linecap="round" stroke-dasharray="220" stroke-dashoffset="{220 - 220*normalized}"/>'
        f'<defs><linearGradient id="vix-grad" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#10b981"/><stop offset="60%" stop-color="#f59e0b"/><stop offset="100%" stop-color="#f43f5e"/></linearGradient></defs>'
        f'<line x1="90" y1="90" x2="{px:.1f}" y2="{py:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
        f'<circle cx="90" cy="90" r="6" fill="{color}"/>'
        f'</svg>'
        f'<div class="gauge-val" style="color:{color}">{float(value):.1f}</div>'
        f'<div class="gauge-sub">{html_escape(sub)}</div>'
        f'<div class="gauge-reason">{html_escape(reason)}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_top_scores(frame: pd.DataFrame, regime: str = "UNKNOWN") -> None:
    if frame.empty:
        return
    threshold = regime_score_threshold(regime)
    top = frame.sort_values(by=["total_score", "confidence_score"], ascending=False).head(5)
    rows = []
    for _, row in top.iterrows():
        action = str(row.get("action") or "UNKNOWN")
        color = score_fill_color(row.get("total_score"), threshold)
        rows.append(
            f'<div class="score-line">'
            f'<div class="sl-points" style="color:{color}">{html_escape(format_decimal(row.get("total_score"), 0))}</div>'
            f'<div class="sl-info"><div class="sl-sym">{html_escape(row.get("ticker"))}</div>'
            f'<div class="sl-sector">{html_escape(short_sector_name(row.get("sector")))}</div></div>'
            f'<span class="badge {badge_class_for_action(action)}">{html_escape(action)}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div class="panel-card">'
        f'<div class="panel-hdr"><span class="panel-title">Top Scores Today</span></div>'
        f'<div class="scores-list">{"".join(rows)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def build_alert_items(alert_logs: pd.DataFrame, frame: pd.DataFrame, run_history: pd.DataFrame) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if not alert_logs.empty:
        for _, row in alert_logs.sort_values(by="created_at", ascending=False).head(5).iterrows():
            msg = str(row.get("message") or "Alert")
            kind = "buy" if "BUY" in msg.upper() else "risk" if "ERROR" in msg.upper() else "fv"
            items.append({"kind": kind, "title": msg, "time": format_timestamp(row.get("created_at"), False)})
    if len(items) < 5 and not frame.empty:
        supp = frame.sort_values(by=["generated_at"], ascending=False)
        for _, row in supp.iterrows():
            if str(row.get("action")) not in {"BUY", "WATCH"}: continue
            items.append({
                "kind": alert_class_for_action(row["action"]),
                "title": f"{row['action']} signal: {row['ticker']} (Score {format_decimal(row.get('total_score'), 0)})",
                "time": format_timestamp(row.get("generated_at"), False),
            })
            if len(items) >= 5: break
    return items[:5]


def render_recent_alerts(alert_logs: pd.DataFrame, frame: pd.DataFrame, run_history: pd.DataFrame) -> None:
    alerts = build_alert_items(alert_logs, frame, run_history)
    if not alerts:
        st.info("No activity logs.")
        return
    rows = []
    for a in alerts:
        rows.append(
            f'<div class="alert-box {a["kind"]}">'
            f'<div class="alert-text">{html_escape(a["title"])}</div>'
            f'<div class="alert-meta">{html_escape(a["time"])}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="panel-card">'
        f'<div class="panel-hdr"><span class="panel-title">Terminal Activity</span></div>'
        f'<div class="alerts-list">{"".join(rows)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_peer_comparison(ticker: str, sector: str, frame: pd.DataFrame) -> None:
    peers = frame[frame["sector"] == sector].sort_values("total_score", ascending=False).head(4)
    rows = []
    for _, row in peers.iterrows():
        active_cls = "active" if row["ticker"] == ticker else ""
        rows.append(
            f'<div class="peer-row {active_cls}">'
            f'<span>{html_escape(row["ticker"])}</span>'
            f'<span class="mono">{format_decimal(row.get("total_score"), 0)}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div class="panel-card">'
        f'<div class="panel-hdr"><span class="panel-title">Sector Comparison</span></div>'
        f'<div class="peers-list">{"".join(rows)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Global styles
# ---------------------------------------------------------------------------

def inject_global_styles() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Geist+Mono:wght@300;400;500;600&family=Geist:wght@300;400;500;600&display=swap');
:root{
  --bg0:#04060d;--bg1:#080c17;--bg2:#0c1120;--bg3:#111928;--bg4:#161f30;
  --wire:rgba(255,255,255,0.055);--wire2:rgba(255,255,255,0.10);--wire3:rgba(255,255,255,0.18);
  --t0:#ffffff;--t1:#c9d4f0;--t2:#6d7fa8;--t3:#3a445e;
  --acid:#00ffa3;--acid-d:rgba(0,255,163,0.10);--acid-g:0 0 20px rgba(0,255,163,0.22);
  --rose:#ff3060;--rose-d:rgba(255,48,96,0.10);
  --gold:#ffb700;--gold-d:rgba(255,183,0,0.10);
  --sky:#3d9eff;--sky-d:rgba(61,158,255,0.10);
  --violet:#9b6dff;--violet-d:rgba(155,109,255,0.10);
  --mono:'Geist Mono',monospace;--sans:'Geist',sans-serif;--display:'Syne',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
.stApp{background:var(--bg0)!important;color:var(--t1);font-family:var(--sans)}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid var(--wire)!important}
[data-testid="stSidebarNav"]{display:none}
.stMarkdown p{margin-bottom:0!important}
.stApp::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent 0,transparent 1px,rgba(255,255,255,.012) 1px,rgba(255,255,255,.012) 2px);
  background-size:100% 2px;opacity:.7}
.stApp::after{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:radial-gradient(ellipse 60% 30% at 5% 0%,rgba(0,255,163,.04) 0%,transparent 70%),
             radial-gradient(ellipse 40% 40% at 95% 5%,rgba(61,158,255,.05) 0%,transparent 70%)}
/* SEARCH BAR */
.search-wrap{position:relative;width:100%}
.search-wrap .search-icon{position:absolute;left:12px;top:50%;transform:translateY(-50%);z-index:2;pointer-events:none;color:var(--t3);transition:color .2s}
.search-wrap:focus-within .search-icon{color:var(--acid)}
.search-wrap .search-shortcut{position:absolute;right:12px;top:50%;transform:translateY(-50%);z-index:2;pointer-events:none;
  font-family:var(--mono);font-size:9px;font-weight:600;color:var(--t3);background:var(--bg1);border:1px solid var(--wire2);
  padding:2px 7px;border-radius:4px;letter-spacing:.3px;transition:opacity .2s}
.search-wrap:focus-within .search-shortcut{opacity:0}
.search-wrap .stTextInput>div>div>input{background:var(--bg2)!important;border:1px solid var(--wire)!important;
  color:var(--t0)!important;border-radius:10px!important;font-family:var(--mono)!important;font-size:12px!important;
  padding:10px 44px 10px 38px!important;height:40px!important;transition:all .2s ease!important;
  letter-spacing:.3px!important}
.search-wrap .stTextInput>div>div>input::placeholder{color:var(--t3)!important;font-size:11px!important;letter-spacing:.5px!important}
.search-wrap .stTextInput>div>div>input:focus{border-color:var(--acid)!important;
  box-shadow:0 0 0 3px var(--acid-d),0 0 20px rgba(0,255,163,.08)!important;background:var(--bg1)!important}
.search-wrap .stTextInput label{display:none!important}
.search-wrap .stTextInput>div{background:transparent!important}
.search-wrap .stTextInput>div>div{background:transparent!important}
.search-wrap::after{content:'';position:absolute;bottom:-1px;left:20%;right:20%;height:1px;
  background:linear-gradient(90deg,transparent,var(--acid),transparent);opacity:0;transition:opacity .3s}
.search-wrap:focus-within::after{opacity:.6}
.stTextInput>div>div>input{background:var(--bg3)!important;border:1px solid var(--wire2)!important;
  color:var(--t1)!important;border-radius:8px!important;font-family:var(--mono)!important;font-size:11px!important}
.stTextInput>div>div>input:focus{border-color:var(--acid)!important;box-shadow:0 0 0 3px var(--acid-d)!important}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(1.3)}}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

/* NAV */
.sov-nav{display:flex;align-items:center;padding:0 40px;height:52px;
  background:rgba(4,6,13,.92);backdrop-filter:blur(24px) saturate(180%);
  border-bottom:1px solid var(--wire);position:sticky;top:0;z-index:1000;
  margin:-5.5rem -4rem 1.5rem -4rem}
.nav-brand{display:flex;align-items:center;gap:11px;padding-right:18px;margin-right:4px;border-right:1px solid var(--wire);flex-shrink:0}
.nav-sigma{width:30px;height:30px;border-radius:8px;
  background:conic-gradient(from 135deg,var(--acid),var(--sky),var(--acid));
  display:flex;align-items:center;justify-content:center;
  font-family:var(--display);font-weight:800;font-size:15px;color:var(--bg0);box-shadow:var(--acid-g)}
.nav-wordmark{font-family:var(--display);font-size:15px;font-weight:800;letter-spacing:2px;color:var(--t0);line-height:1}
.nav-sub{font-family:var(--mono);font-size:8px;color:var(--t3);letter-spacing:1px;text-transform:uppercase}
.nav-regime{font-family:var(--mono);font-size:10px;font-weight:600;padding:3px 10px;border-radius:4px;margin-left:8px;letter-spacing:.5px}
.nr-SIDEWAYS{background:var(--gold-d);color:var(--gold)}
.nr-BULL{background:var(--acid-d);color:var(--acid)}
.nr-BEAR{background:var(--rose-d);color:var(--rose)}
.nr-QUALITY{background:var(--sky-d);color:var(--sky)}
.nr-CRISIS{background:var(--rose-d);color:var(--rose)}
.nr-RISK_ON{background:var(--acid-d);color:var(--acid)}
.nr-RISK_OFF{background:var(--gold-d);color:var(--gold)}
.nav-links{display:flex;gap:1px;padding:0 12px;flex:1;overflow:hidden}
.nav-tab{padding:0 12px;height:52px;display:flex;align-items:center;font-family:var(--mono);font-size:11px;font-weight:500;color:var(--t3);text-decoration:none;border-bottom:2px solid transparent;transition:all .15s;white-space:nowrap}
.nav-tab:hover{color:var(--t1);border-bottom-color:var(--wire2)}
.nav-tab.active{color:var(--acid);border-bottom-color:var(--acid)}
.nav-right{display:flex;align-items:center;gap:10px;padding-left:16px;border-left:1px solid var(--wire);flex-shrink:0}
.vix-pill{display:flex;align-items:center;gap:7px;padding:5px 12px;border-radius:20px;background:var(--bg2);border:1px solid var(--wire);font-family:var(--mono);font-size:11px}
.vix-dot{width:6px;height:6px;border-radius:50%;animation:pulse 2.2s ease-in-out infinite}
.vd-safe{background:var(--acid);box-shadow:0 0 8px var(--acid)}
.vd-warn{background:var(--gold);box-shadow:0 0 8px var(--gold)}
.vd-risk{background:var(--rose);box-shadow:0 0 8px var(--rose)}
.vix-num{color:var(--gold);font-weight:600}
.vix-cap{color:var(--t3)}
.scan-cta{padding:7px 16px;border-radius:20px;background:linear-gradient(135deg,var(--acid) 0%,#00cc85 100%);color:var(--bg0);font-family:var(--mono);font-size:11px;font-weight:600;text-decoration:none;display:flex;align-items:center;gap:6px;box-shadow:0 4px 14px rgba(0,255,163,.22);transition:all .2s}
.scan-cta:hover{box-shadow:0 6px 22px rgba(0,255,163,.38);transform:translateY(-1px)}

/* TICKER */
.ticker-tape{height:31px;background:var(--bg1);border-bottom:1px solid var(--wire);display:flex;align-items:center;overflow:hidden;margin:0 -4rem}
.ticker-inner{display:flex;animation:tape 55s linear infinite;white-space:nowrap}
.ticker-item{display:flex;align-items:center;gap:9px;padding:0 22px;border-right:1px solid var(--wire);font-family:var(--mono);font-size:10.5px}
.t-sym{color:var(--t0);font-weight:600;letter-spacing:.3px}.t-px{color:var(--t2)}.t-up{color:var(--acid)}.t-dn{color:var(--rose)}
@keyframes tape{from{transform:translateX(0)}to{transform:translateX(-50%)}}

/* SCAN META + NOTICE */
.scan-meta{display:flex;align-items:center;gap:14px;padding:7px 0;font-family:var(--mono);font-size:10px;color:var(--t3);margin-bottom:4px}
.scan-dot{width:5px;height:5px;border-radius:50%;background:var(--acid);box-shadow:0 0 6px var(--acid);animation:pulse 2s infinite}
.notice-strip{display:flex;align-items:center;gap:10px;padding:10px 16px;margin-bottom:18px;background:rgba(255,183,0,.07);border:1px solid rgba(255,183,0,.18);border-left:3px solid var(--gold);border-radius:0 8px 8px 0;font-family:var(--mono);font-size:11px;color:var(--gold);line-height:1.5}

/* STAT CARDS */
.stat-card{background:var(--bg2);border:1px solid var(--wire);border-radius:12px;padding:16px 18px;position:relative;overflow:hidden;transition:border-color .2s,transform .2s;animation:rise .35s ease both}
.stat-card:hover{border-color:var(--wire2);transform:translateY(-1px)}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent,transparent)}
.stat-card::after{content:'';position:absolute;bottom:0;left:20%;right:20%;height:1px;background:linear-gradient(90deg,transparent,var(--accent,var(--wire)),transparent)}
.stat-card:nth-child(1){animation-delay:.04s}.stat-card:nth-child(2){animation-delay:.08s}.stat-card:nth-child(3){animation-delay:.12s}.stat-card:nth-child(4){animation-delay:.16s}.stat-card:nth-child(5){animation-delay:.20s}.stat-card:nth-child(6){animation-delay:.24s}
.stat-label{font-family:var(--mono);font-size:9px;font-weight:500;text-transform:uppercase;letter-spacing:1px;color:var(--t3);margin-bottom:9px}
.stat-val{font-family:var(--display);font-size:30px;font-weight:800;line-height:1;color:var(--t0);margin-bottom:5px;letter-spacing:-.5px}
.stat-val.serif{font-family:var(--display)}
.sv-acid{color:var(--acid)}.sv-rose{color:var(--rose)}.sv-gold{color:var(--gold)}.sv-sky{color:var(--sky)}.sv-violet{color:var(--violet)}
.stat-foot{display:flex;justify-content:space-between;align-items:center}
.stat-sub{font-family:var(--mono);font-size:9px;color:var(--t3)}
.d-up{font-family:var(--mono);font-size:9px;font-weight:700;padding:1px 6px;border-radius:2px;background:var(--acid-d);color:var(--acid)}
.d-dn{font-family:var(--mono);font-size:9px;font-weight:700;padding:1px 6px;border-radius:2px;background:var(--rose-d);color:var(--rose)}

/* PANEL */
.panel-card{background:var(--bg2);border:1px solid var(--wire);border-radius:16px;overflow:hidden;margin-bottom:14px}
.panel-hdr{display:flex;align-items:center;justify-content:space-between;padding:12px 18px;border-bottom:1px solid var(--wire)}
.panel-title{font-family:var(--mono);font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--t2);display:flex;align-items:center;gap:8px}
.panel-title::before{content:'';width:10px;height:1px;background:var(--accent-line,var(--acid))}
.ptag{font-family:var(--mono);font-size:9px;font-weight:700;padding:2px 8px;border-radius:20px}
.ptag-warn{background:var(--rose-d);color:var(--rose)}.ptag-ok{background:var(--acid-d);color:var(--acid)}.ptag-info{background:var(--sky-d);color:var(--sky)}

/* SIGNAL TABLE */
.signal-tbl{background:var(--bg2);border:1px solid var(--wire);border-radius:16px;overflow:hidden;margin-top:10px}
.thead{display:grid;grid-template-columns:30px 1.5fr .9fr 68px .75fr 1fr .9fr 90px;padding:9px 18px;background:rgba(255,255,255,.02);border-bottom:1px solid var(--wire);font-family:var(--mono);font-size:9px;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:.9px}
.trow{display:grid;grid-template-columns:30px 1.5fr .9fr 68px .75fr 1fr .9fr 90px;padding:11px 18px;border-bottom:1px solid rgba(255,255,255,.025);align-items:center;transition:background .12s;cursor:pointer}
.trow:last-child{border-bottom:none}
.trow:hover{background:rgba(255,255,255,.02)}
.trow.sel{background:rgba(0,255,163,.06)}
.r{text-align:right}.c{text-align:center}
.rank-n{font-family:var(--mono);font-size:10px;color:var(--t3)}
.sym-lg{font-family:var(--display);font-size:13.5px;font-weight:700;color:var(--t0);letter-spacing:.3px;display:block}
.ticker-link{text-decoration:none}
.co-sm{font-family:var(--mono);font-size:9px;color:var(--t3);margin-top:1px}
.tag-row{display:flex;gap:3px;margin-top:3px}
.rtag{font-family:var(--mono);font-size:8px;font-weight:700;padding:1px 5px;border-radius:2px;text-transform:uppercase;letter-spacing:.3px}
.rtag.hq{background:var(--acid-d);color:var(--acid)}.rtag.dv{background:var(--sky-d);color:var(--sky)}.rtag.mo{background:var(--gold-d);color:var(--gold)}
.px-cell{font-family:var(--mono);font-size:12px;font-weight:500;color:var(--t1);text-align:right}
.up-cell{font-family:var(--mono);font-size:12px;font-weight:600;text-align:right}
.up-cell.pos{color:var(--acid)}.up-cell.neg{color:var(--rose)}
.sc-wrap{display:flex;align-items:center;gap:7px;justify-content:flex-end}
.sc-bar{width:42px;height:3px;background:var(--bg0);border-radius:1px;overflow:hidden}
.sc-fill{height:100%;border-radius:1px}
.sc-num{font-family:var(--mono);font-size:11px;font-weight:600;min-width:22px;text-align:right}
.sec-cell{font-family:var(--mono);font-size:10px;color:var(--t2);text-align:right}
.badge{font-family:var(--mono);font-size:9px;font-weight:700;padding:3px 9px;border-radius:4px;text-transform:uppercase;letter-spacing:.4px;display:inline-block}
.badge-buy{background:var(--acid-d);color:var(--acid);border:1px solid rgba(0,255,163,.2)}
.badge-watch{background:var(--sky-d);color:var(--sky);border:1px solid rgba(61,158,255,.2)}
.badge-weak{background:var(--bg3);color:var(--t3);border:1px solid var(--wire)}

/* FOCUS CARD */
.focus-card{background:var(--bg2);border:1px solid var(--wire2);border-radius:16px;overflow:hidden;margin-bottom:14px}
.focus-hero{padding:18px 20px;border-bottom:1px solid var(--wire);background:linear-gradient(135deg,var(--bg3) 0%,var(--bg2) 100%);position:relative;overflow:hidden}
.focus-hero::after{content:attr(data-sym);pointer-events:none;position:absolute;right:-4px;top:-16px;font-family:var(--display);font-size:72px;font-weight:800;color:rgba(255,255,255,.025);letter-spacing:-2px;white-space:nowrap}
.focus-sym{font-family:var(--display);font-size:24px;font-weight:800;color:var(--t0);letter-spacing:.5px}
.focus-co{font-family:var(--mono);font-size:10px;color:var(--t3);margin:2px 0 13px}
.focus-price{font-family:var(--mono);font-size:20px;font-weight:600;color:var(--t0)}
.focus-chg{font-family:var(--mono);font-size:12px;font-weight:600;margin-left:10px}
.focus-chg.up{color:var(--acid)}.focus-chg.dn{color:var(--rose)}
.vg-wrap{padding:10px 20px 0}
.vg-labels{display:flex;justify-content:space-between;font-family:var(--mono);font-size:8px;color:var(--t3);margin-bottom:5px;text-transform:uppercase;letter-spacing:.5px}
.vg-track{height:5px;background:var(--bg0);border-radius:2px;position:relative}
.vg-fill{height:100%;width:100%;border-radius:2px;background:linear-gradient(90deg,var(--acid),var(--gold),var(--rose))}
.vg-thumb{position:absolute;top:-5px;width:3px;height:15px;background:white;border-radius:1px;transform:translateX(-50%);box-shadow:0 0 10px rgba(255,255,255,.6)}
.checklist{padding:14px 20px}
.cl-title{font-family:var(--mono);font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--t3);margin-bottom:10px}
.cl-item{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.cl-item:last-child{border-bottom:none}
.cl-key{font-family:var(--mono);font-size:11px;color:var(--t2)}
.cl-val{font-family:var(--mono);font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px}
.cv-great{background:var(--acid-d);color:var(--acid)}.cv-good{background:var(--sky-d);color:var(--sky)}.cv-mid{background:var(--gold-d);color:var(--gold)}.cv-poor{background:var(--rose-d);color:var(--rose)}

/* REGIME */
.regime-card{background:var(--bg2);border:1px solid var(--wire);border-radius:16px;padding:18px;margin-bottom:14px}
.rc-label{font-family:var(--mono);font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--t3);display:flex;align-items:center;gap:8px;margin-bottom:10px}
.rc-label::before{content:'';width:10px;height:1px;background:var(--gold)}
.regime-word{font-family:var(--display);font-size:40px;font-weight:800;letter-spacing:-1px;line-height:.95;margin-bottom:5px}
.rw-SIDEWAYS{color:var(--gold)}.rw-BULL{color:var(--acid)}.rw-BEAR{color:var(--rose)}.rw-QUALITY{color:var(--sky)}.rw-CRISIS{color:var(--rose)}.rw-RISK_ON{color:var(--acid)}.rw-RISK_OFF{color:var(--gold)}
.regime-desc{font-family:var(--mono);font-size:10px;color:var(--t3);margin-bottom:16px}
.rule-list{border-top:1px solid var(--wire);padding-top:12px}
.rule-row{display:flex;align-items:center;justify-content:space-between;padding:6px 0;font-family:var(--mono);font-size:11px}
.rule-lbl{color:var(--t2)}.rv-ok{color:var(--acid);font-weight:600}.rv-warn{color:var(--gold);font-weight:600}.rv-bad{color:var(--rose);font-weight:600}

/* SECTOR BARS */
.sector-card{background:var(--bg2);border:1px solid var(--wire);border-radius:16px;padding:16px 18px;margin-bottom:14px}
.sc-label{font-family:var(--mono);font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--t3);display:flex;align-items:center;gap:8px;margin-bottom:12px}
.sc-label::before{content:'';width:10px;height:1px;background:var(--sky)}
.sr{margin-bottom:11px}
.sr-meta{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;margin-bottom:4px}
.sr-name{color:var(--t1);font-weight:500}.sr-cnt{color:var(--t3)}
.sr-track{height:3px;background:var(--bg0);border-radius:1px;overflow:hidden}
.sr-fill{height:100%;border-radius:1px}

/* OWNERSHIP FLAGS */
.flags-card{background:var(--bg2);border:1px solid var(--wire);border-radius:16px;padding:16px 18px;margin-bottom:14px}
.fl-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.fl-title{font-family:var(--mono);font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--t3);display:flex;align-items:center;gap:8px}
.fl-title::before{content:'';width:10px;height:1px;background:var(--rose)}
.fl-badge{font-family:var(--mono);font-size:9px;font-weight:700;padding:2px 8px;border-radius:20px;background:var(--rose-d);color:var(--rose)}
.fl-ok-badge{font-family:var(--mono);font-size:9px;font-weight:700;padding:2px 8px;border-radius:20px;background:var(--acid-d);color:var(--acid)}
.flag-row{display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.flag-row:last-child{border-bottom:none}
.flag-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.flag-info{flex:1}
.flag-sym{font-family:var(--display);font-size:13px;font-weight:700;color:var(--t0)}
.flag-desc{font-family:var(--mono);font-size:9px;color:var(--t3);margin-top:1px}
.flag-val{font-family:var(--mono);font-size:11px;font-weight:600}

/* VIX GAUGE */
.vix-card{background:var(--bg2);border:1px solid var(--wire);border-radius:16px;padding:16px 18px;margin-bottom:14px}
.vix-hd{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.vix-hd-t{font-family:var(--mono);font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--t3);display:flex;align-items:center;gap:8px}
.vix-hd-t::before{content:'';width:10px;height:1px;background:var(--gold)}
.vix-state-p{font-family:var(--mono);font-size:9px;font-weight:700;padding:2px 8px;border-radius:20px;background:var(--gold-d);color:var(--gold)}
.vix-svg-c{display:flex;justify-content:center;padding:4px 0}
.vix-big{font-family:var(--display);font-size:42px;font-weight:800;text-align:center;line-height:1}
.vix-sub{font-family:var(--mono);font-size:10px;color:var(--t3);text-align:center;margin-top:4px}

/* HEATMAP */
.heatmap-wrap{background:var(--bg2);border:1px solid var(--wire);border-radius:16px;overflow:hidden;margin-bottom:14px}
.hm-head{display:flex;justify-content:space-between;align-items:center;padding:12px 18px;border-bottom:1px solid var(--wire)}
.hm-title{font-family:var(--mono);font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--t2);display:flex;align-items:center;gap:8px}
.hm-title::before{content:'';width:10px;height:1px;background:var(--acid)}
.hm-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:3px;padding:10px}
.hm-cell{border-radius:4px;padding:9px 8px;cursor:pointer;transition:opacity .15s,transform .15s}
.hm-cell:hover{opacity:.8;transform:scale(1.04);outline:1px solid rgba(255,255,255,.15)}
.hm-name{font-family:var(--mono);font-size:9px;font-weight:600;color:rgba(255,255,255,.85);margin-bottom:4px}
.hm-count{font-family:var(--display);font-size:18px;color:#fff;line-height:1}

/* ALERTS */
.alerts-card{background:var(--bg2);border:1px solid var(--wire);border-radius:16px;padding:16px 18px;margin-bottom:14px}
.alert-item{padding:11px 12px;border-radius:8px;margin-bottom:8px;border-left:2px solid transparent}
.alert-item:last-child{margin-bottom:0}
.alert-item.buy{background:rgba(0,255,163,.05);border-color:var(--acid)}
.alert-item.fv{background:rgba(61,158,255,.05);border-color:var(--sky)}
.alert-item.risk{background:rgba(255,48,96,.05);border-color:var(--rose)}
.alert-title{font-family:var(--mono);font-size:11px;font-weight:600;color:var(--t1);line-height:1.4}
.alert-time{font-family:var(--mono);font-size:9px;color:var(--t3);margin-top:3px}

/* PEERS */
.peers-card{background:var(--bg2);border:1px solid var(--wire);border-radius:16px;padding:16px 18px;margin-bottom:14px}
.peer-row{display:flex;justify-content:space-between;align-items:center;padding:8px 10px;border-radius:8px;font-family:var(--mono);font-size:12px}
.peer-row.active{background:var(--acid-d);color:var(--acid);font-weight:700}

/* STATUS BAR */
.status-bar{display:flex;align-items:center;justify-content:space-between;padding:6px 40px;background:var(--bg1);border-top:1px solid var(--wire);font-family:var(--mono);font-size:9px;color:var(--t3);margin:0 -4rem -4rem -4rem}
.sb-left{display:flex;align-items:center;gap:16px}
.sb-dot{display:inline-block;width:4px;height:4px;border-radius:50%;background:var(--acid);box-shadow:0 0 5px var(--acid);margin-right:5px;animation:pulse 2s infinite}

::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg0)}
::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--bg3)}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def render_dashboard(
    frame: pd.DataFrame,
    swing_df: pd.DataFrame,
    mb_df: pd.DataFrame,
    regime_snapshot: dict | None,
    vix_snapshot: dict | None,
    alert_logs: pd.DataFrame,
    run_history: pd.DataFrame,
) -> None:
    c_panel, s_panel = st.columns([0.72, 0.28], gap="medium")

    with c_panel:
        render_stat_cards(frame, swing_df, mb_df, regime_snapshot, run_history)
        render_notice_strip(frame, run_history)

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        h_col, a_col = st.columns([0.65, 0.35], gap="small")
        with h_col:
            st.markdown('<div class="panel-hdr"><span class="panel-title">Core Strategy Heatmap</span></div>', unsafe_allow_html=True)
            render_sector_heatmap(frame)
        with a_col:
            render_ownership_flags(frame)

        st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)

        # --- Signal table header with premium search ---
        t_col, f_col = st.columns([0.55, 0.45])
        with t_col:
            st.markdown('<div class="panel-hdr" style="margin-bottom:0"><span class="panel-title">Institutional BUY/WATCH Signals</span></div>', unsafe_allow_html=True)
        with f_col:
            st.markdown("""
<div class="search-wrap">
  <div class="search-icon">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>
  </div>
  <span class="search-shortcut">/</span>
""", unsafe_allow_html=True)
            st_search = st.text_input("Ticker search", placeholder="Filter by ticker, sector, action...", label_visibility="collapsed", key="dash_search")
            st.markdown('</div>', unsafe_allow_html=True)

        regime = get_current_regime(regime_snapshot)
        filtered = filter_dashboard_signals(frame, st_search, None)
        render_signal_table(filtered, regime)

    with s_panel:
        selected_ticker = query_param_value("ticker", "")
        if not selected_ticker and not frame.empty:
            top_buy = frame[frame["action"] == "BUY"]
            if not top_buy.empty:
                selected_ticker = str(top_buy.iloc[0]["ticker"])

        if selected_ticker:
            row = frame[frame["ticker"] == selected_ticker]
            if not row.empty:
                r = row.iloc[0]
                render_wealth_checklist(selected_ticker, r["total_score"], r["price"], r["fair_value"], regime, r.get("valuation_confidence", None))
                render_peer_comparison(selected_ticker, r["sector"], frame)
                st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        render_regime_card(regime_snapshot, vix_snapshot)
        render_vix_gauge(vix_snapshot)
        render_sector_distribution(frame)
        render_recent_alerts(alert_logs, frame, run_history)


def build_price_figure(ticker: str, period: str = "6mo") -> go.Figure | None:
    history = load_price_history(ticker, period=period, interval="1d")
    if history.empty or "Close" not in history.columns:
        return None
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=history.index,
                open=history["Open"],
                high=history["High"],
                low=history["Low"],
                close=history["Close"],
                name=ticker,
            )
        ]
    )
    fig.update_layout(
        margin=dict(t=10, l=10, r=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        height=360,
    )
    return fig


def render_swing_page(swing_df: pd.DataFrame, frame: pd.DataFrame) -> None:
    st.header("Swing Trades")
    if swing_df.empty:
        st.info("Swing dataset is empty. Run the swing pipeline first.")
        return

    working = swing_df.copy()
    if not frame.empty:
        working = working.merge(frame[["ticker", "sector", "company_name"]], on="ticker", how="left")
    status_filter = st.radio("Status", ["Active", "Exited", "All"], horizontal=True)
    sector_options = ["All"] + sorted(value for value in working.get("sector", pd.Series(dtype=str)).dropna().unique())
    selected_sector = st.selectbox("Sector", sector_options, index=0)

    if status_filter == "Active":
        working = working[working["action"].isin(["ENTRY", "HOLD"])]
    elif status_filter == "Exited":
        working = working[working["action"].eq("EXIT")]
    if selected_sector != "All":
        working = working[working["sector"].eq(selected_sector)]

    display_cols = ["ticker", "action", "price", "target", "stop_loss", "rsi", "trend", "volume_surge", "generated_at"]
    available_cols = [column for column in display_cols if column in working.columns]
    st.dataframe(
        working[available_cols].rename(
            columns={
                "price": "Entry Price",
                "target": "Target",
                "stop_loss": "Stop",
                "rsi": "RSI",
                "trend": "Trend",
                "volume_surge": "Volume Surge",
                "generated_at": "Generated At",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    tickers = working["ticker"].dropna().tolist()
    if not tickers:
        return
    selected_ticker = st.selectbox("Swing Chart", tickers, key="swing_chart_ticker")
    figure = build_price_figure(selected_ticker, period="3mo")
    if figure is not None:
        row = working[working["ticker"] == selected_ticker].iloc[0]
        if "price" in row and pd.notna(row["price"]):
            figure.add_scatter(
                x=[pd.Timestamp.now()],
                y=[float(row["price"])],
                mode="markers",
                marker=dict(size=10, color="#38bdf8"),
                name=str(row.get("action", "Signal")),
            )
        if "target" in row and pd.notna(row["target"]):
            figure.add_hline(y=float(row["target"]), line_dash="dot", line_color="#10b981")
        if "stop_loss" in row and pd.notna(row["stop_loss"]):
            figure.add_hline(y=float(row["stop_loss"]), line_dash="dot", line_color="#f43f5e")
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def render_positionals_page(frame: pd.DataFrame, regime_snapshot: dict | None) -> None:
    st.header("Positionals")
    signals = frame[frame["action"].isin(["BUY", "WATCH"])].copy()
    if signals.empty:
        st.info("No positional BUY/WATCH signals are available.")
        return

    render_signal_table(signals, get_current_regime(regime_snapshot))
    selected_ticker = st.selectbox("Ticker Analysis", signals["ticker"].tolist(), key="positionals_primary")
    score_snapshot = db.get_latest_score(selected_ticker)
    detail_row = signals[signals["ticker"] == selected_ticker].iloc[0]

    left, right = st.columns([0.45, 0.55], gap="large")
    with left:
        st.subheader(f"{selected_ticker} Snapshot")
        st.metric("Price", format_currency(detail_row.get("price")))
        st.metric("Fair Value", format_currency(detail_row.get("fair_value")))
        st.metric("Score", format_decimal(detail_row.get("total_score"), 1))
        st.metric("Confidence", format_decimal(detail_row.get("confidence_score"), 1))
        figure = build_price_figure(selected_ticker, period="6mo")
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.subheader("Factor Breakdown")
        factors = pd.DataFrame(score_snapshot.get("factor_scores", []) if score_snapshot else [])
        if factors.empty:
            st.info("Factor scores are unavailable for the selected ticker.")
        else:
            st.dataframe(
                factors.rename(columns={"factor": "Factor", "normalized_score": "Score", "weight": "Weight"}),
                use_container_width=True,
                hide_index=True,
            )

    comparison_options = signals["ticker"].tolist()
    default_compare = comparison_options[1] if len(comparison_options) > 1 else comparison_options[0]
    compare_ticker = st.selectbox("Compare With", comparison_options, index=comparison_options.index(default_compare), key="positionals_compare")
    comparison = signals[signals["ticker"].isin([selected_ticker, compare_ticker])][
        ["ticker", "price", "fair_value", "total_score", "confidence_score", "sector", "promoter_pct", "pledge_pct"]
    ].set_index("ticker")
    if not comparison.empty:
        st.subheader("Comparison View")
        st.dataframe(comparison.T, use_container_width=True)


def render_multibagger_page(mb_df: pd.DataFrame, frame: pd.DataFrame) -> None:
    st.header("Multibagger Hunt")
    leaderboard = mb_df.copy()
    if leaderboard.empty:
        fallback = frame[frame["action"].isin(["BUY", "WATCH"])].copy()
        if fallback.empty:
            st.info("Multibagger candidates are unavailable.")
            return
        leaderboard = fallback[["ticker", "price", "total_score", "roe_5y", "sales_growth_5y", "debt_equity", "piotroski_score"]].copy()
        leaderboard["conviction_score"] = leaderboard["total_score"]
        leaderboard["quality_score"] = leaderboard["total_score"]
        leaderboard["early_signal_score"] = leaderboard["sales_growth_5y"].fillna(0.0) * 100
        leaderboard["tam_score"] = leaderboard["roe_5y"].fillna(0.0) * 100
    st.dataframe(
        leaderboard[["ticker", "conviction_score", "quality_score", "early_signal_score", "tam_score", "price"]]
        .sort_values("conviction_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    ticker_options = leaderboard["ticker"].dropna().tolist()
    selected_ticker = st.selectbox("Candidate Detail", ticker_options, key="multibagger_detail")
    detail = frame[frame["ticker"] == selected_ticker]
    if not detail.empty:
        row = detail.iloc[0]
        cols = st.columns(4)
        cols[0].metric("ROE 5Y", format_percent_like(row.get("roe_5y")))
        cols[1].metric("Sales CAGR", format_percent_like(row.get("sales_growth_5y")))
        cols[2].metric("Debt/Equity", format_decimal(row.get("debt_equity"), 2))
        cols[3].metric("Piotroski", format_decimal(row.get("piotroski_score"), 0))

    capital = st.number_input("Capital for Tranche Calculator", min_value=10000.0, value=float(config.DEFAULT_PORTFOLIO_CAPITAL), step=10000.0)
    tranche_size = capital / max(1, config.MB_TRANCHE_COUNT)
    st.caption(
        f"{config.MB_TRANCHE_COUNT} tranches x {format_currency(tranche_size)} "
        f"at {format_percent_like(config.MB_TRANCHE_RISK_PCT, 2)} risk per tranche."
    )


def render_fair_value_page(frame: pd.DataFrame) -> None:
    st.header("Fair Value")
    valuations = frame[frame["fair_value"].notna() & frame["price"].notna()].copy()
    if valuations.empty:
        st.info("No valuation snapshots are available.")
        return

    st.dataframe(
        valuations[["ticker", "price", "dcf_value", "graham_value", "eps_value", "peg_value", "fair_value", "margin_of_safety_pct", "upside_pct"]]
        .sort_values("upside_pct", ascending=False, na_position="last"),
        use_container_width=True,
        hide_index=True,
    )

    scatter = px.scatter(
        valuations,
        x="price",
        y="fair_value",
        color="upside_pct",
        hover_name="ticker",
        color_continuous_scale=["#f43f5e", "#f59e0b", "#10b981"],
    )
    scatter.add_shape(type="line", x0=valuations["price"].min(), y0=valuations["price"].min(), x1=valuations["price"].max(), y1=valuations["price"].max(), line=dict(color="#94a3b8", dash="dash"))
    scatter.update_layout(margin=dict(t=20, l=10, r=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(scatter, use_container_width=True, config={"displayModeBar": False})

    selected_ticker = st.selectbox("Margin of Safety Detail", valuations["ticker"].tolist(), key="fair_value_detail")
    row = valuations[valuations["ticker"] == selected_ticker].iloc[0]
    st.markdown(render_valuation_gauge(float(row["price"]), float(row["fair_value"]), row.get("valuation_confidence")), unsafe_allow_html=True)


def render_sector_rank_page(frame: pd.DataFrame) -> None:
    st.header("Sector Rank")
    if frame.empty:
        st.info("No sector ranking data is available.")
        return
    sector_stats = frame.groupby("sector", dropna=False).agg(
        ticker_count=("ticker", "count"),
        avg_score=("total_score", "mean"),
        avg_upside=("upside_pct", "mean"),
        avg_momentum=("price_return_3m", "mean") if "price_return_3m" in frame.columns else ("total_score", "mean"),
    ).reset_index()
    sector_stats["sector"] = sector_stats["sector"].fillna("Unknown")
    heatmap = px.bar(
        sector_stats.sort_values("avg_score", ascending=False),
        x="sector",
        y="avg_score",
        color="avg_upside",
        color_continuous_scale=["#f43f5e", "#f59e0b", "#10b981"],
    )
    heatmap.update_layout(margin=dict(t=20, l=10, r=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(heatmap, use_container_width=True, config={"displayModeBar": False})

    top_by_sector = (
        frame.sort_values(["sector", "total_score"], ascending=[True, False])
        .groupby("sector", dropna=False)
        .head(3)[["sector", "ticker", "total_score", "upside_pct"]]
    )
    st.dataframe(top_by_sector, use_container_width=True, hide_index=True)


def render_ownership_page(frame: pd.DataFrame) -> None:
    st.header("Ownership")
    ownership = load_analysis_dataset("ownership")
    if ownership.empty:
        ownership = frame[["ticker", "sector", "promoter_pct", "pledge_pct", "fii_delta", "dii_delta"]].copy()
    else:
        ownership = ownership.merge(frame[["ticker", "sector"]], on="ticker", how="left")
    if ownership.empty:
        st.info("Ownership snapshots are unavailable.")
        return

    st.dataframe(
        ownership[["ticker", "sector", "promoter_pct", "pledge_pct", "fii_delta", "dii_delta"]].sort_values("promoter_pct", ascending=False, na_position="last"),
        use_container_width=True,
        hide_index=True,
    )
    selected_ticker = st.selectbox("Ownership Detail", ownership["ticker"].dropna().tolist(), key="ownership_detail")
    row = ownership[ownership["ticker"] == selected_ticker].iloc[0]
    chart_df = pd.DataFrame(
        [
            {"metric": "Promoter %", "value": row.get("promoter_pct")},
            {"metric": "Pledge %", "value": row.get("pledge_pct")},
            {"metric": "FII Delta", "value": row.get("fii_delta")},
            {"metric": "DII Delta", "value": row.get("dii_delta")},
        ]
    ).dropna()
    if not chart_df.empty:
        fig = px.bar(chart_df, x="metric", y="value", color="value", color_continuous_scale=["#f43f5e", "#f59e0b", "#10b981"])
        fig.update_layout(margin=dict(t=20, l=10, r=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_risk_page(frame: pd.DataFrame, vix_snapshot: dict | None) -> None:
    st.header("Risk Monitor")
    left, right = st.columns([0.35, 0.65], gap="large")
    with left:
        render_vix_gauge(vix_snapshot)
        state = str((vix_snapshot or {}).get("payload", {}).get("state", "NORMAL")).upper()
        st.metric("Kill Switch", "ACTIVE" if state == "HALT" else "INACTIVE")
    with right:
        risk_df = load_analysis_dataset("risk_metrics")
        if risk_df.empty:
            st.info("Risk metric snapshots are unavailable.")
        else:
            st.dataframe(
                risk_df[["ticker", "volatility_6m", "beta_vs_nifty", "max_drawdown_6m", "score"]].sort_values("score"),
                use_container_width=True,
                hide_index=True,
            )

    candidates = [position.ticker for position in db.list_portfolio_positions()]
    if not candidates and not frame.empty:
        candidates = frame[frame["action"].isin(["BUY", "WATCH"])]["ticker"].head(6).tolist()
    if len(candidates) >= 2:
        returns: dict[str, pd.Series] = {}
        for ticker in candidates[:6]:
            history = load_price_history(ticker, period="6mo", interval="1d")
            if not history.empty and "Close" in history.columns:
                returns[ticker] = history["Close"].pct_change().dropna()
        corr_frame = pd.DataFrame(returns).dropna()
        if not corr_frame.empty and corr_frame.shape[1] >= 2:
            corr = corr_frame.corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn_r")
            fig.update_layout(margin=dict(t=20, l=10, r=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.subheader("Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_portfolio_page(frame: pd.DataFrame) -> None:
    st.header("Portfolio")
    snapshot = get_portfolio_engine().snapshot()
    metrics = st.columns(3)
    metrics[0].metric("Total Value", format_currency(snapshot.total_value))
    metrics[1].metric("Cash", format_currency(snapshot.cash))
    metrics[2].metric("Equity", format_currency(snapshot.equity_value))

    if snapshot.positions:
        positions_df = pd.DataFrame(
            [
                {
                    "ticker": position.ticker,
                    "sector": position.sector,
                    "quantity": position.quantity,
                    "avg_cost": position.avg_cost,
                    "last_price": position.last_price,
                    "market_value": position.market_value,
                    "pnl_pct": (position.last_price / position.avg_cost - 1.0) if position.avg_cost else None,
                    "stop_loss": position.stop_loss,
                }
                for position in snapshot.positions
            ]
        )
        st.dataframe(positions_df, use_container_width=True, hide_index=True)
        pie = px.pie(
            names=list(snapshot.sector_exposure.keys()),
            values=list(snapshot.sector_exposure.values()),
            hole=0.5,
        )
        pie.update_layout(margin=dict(t=10, l=10, r=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No open positions. Use CLI portfolio actions to seed holdings.")

    transactions = load_portfolio_transactions()
    if not transactions.empty:
        st.subheader("Transaction Log")
        st.dataframe(transactions, use_container_width=True, hide_index=True)

    recommendations = frame[frame["action"] == "BUY"][["ticker", "price", "confidence_score", "sector"]].head(10)
    if not recommendations.empty:
        st.subheader("Position Sizing Watchlist")
        st.dataframe(recommendations, use_container_width=True, hide_index=True)


def render_backtest_page() -> None:
    st.header("Backtest")
    col1, col2, col3 = st.columns(3)
    default_end = date.today()
    default_start = default_end - timedelta(days=180)
    start_date = col1.date_input("Start", value=default_start)
    end_date = col2.date_input("End", value=default_end)
    signal_filter = col3.selectbox("Signals", ["BUY", "BUY + WATCH"], index=1)

    if st.button("Run Backtest", type="primary"):
        signal_rows = db.list_signals()
        if signal_filter == "BUY":
            signal_rows = [row for row in signal_rows if row["action"] == "BUY"]
        signal_models = [
            SignalResult(
                ticker=row["ticker"],
                action=row["action"],
                confidence_score=row["confidence_score"],
                reason_code=row["reason_code"],
                satisfied_conditions=row["satisfied_conditions"],
                failed_conditions=row["failed_conditions"],
                data_warnings=row["data_warnings"],
                generated_at=row["generated_at"],
            )
            for row in signal_rows
            if row["action"] in {"BUY", "WATCH"}
        ]
        result = get_portfolio_simulator().run(signal_models, str(start_date), str(end_date))
        metrics = st.columns(4)
        metrics[0].metric("Total P&L", format_percent_like(result.total_pnl))
        metrics[1].metric("Win Rate", format_percent_like(result.win_rate))
        metrics[2].metric("Best", result.best_ticker or "n/a")
        metrics[3].metric("Worst", result.worst_ticker or "n/a")
        if result.equity_curve:
            equity = pd.DataFrame(result.equity_curve)
            equity["date"] = pd.to_datetime(equity["date"])
            st.plotly_chart(px.line(equity, x="date", y="equity"), use_container_width=True, config={"displayModeBar": False})
        if result.trade_outcomes:
            st.dataframe(pd.DataFrame(result.trade_outcomes), use_container_width=True, hide_index=True)


def render_alerts_page(alert_logs: pd.DataFrame) -> None:
    st.header("Alerts Feed")
    if alert_logs.empty:
        st.info("No alert logs are available.")
        return
    alert_type = st.selectbox("Filter", ["All", "BUY", "Fair Value", "Risk", "System"], index=0)
    working = alert_logs.copy()
    if alert_type != "All":
        keyword_map = {
            "BUY": "BUY",
            "Fair Value": "fair",
            "Risk": "risk",
            "System": "system",
        }
        working = working[working["message"].str.contains(keyword_map[alert_type], case=False, na=False)]
    st.dataframe(working[["created_at", "level", "message", "context"]], use_container_width=True, hide_index=True)


def render_logs_page() -> None:
    st.header("Logs & Data Audit")
    run_history = load_run_history()
    error_logs = load_error_logs()
    model_versions = load_model_versions()
    latest_audits = load_latest_audit_rows()
    universe_audits = load_universe_audit_runs()
    scheduler_jobs = load_scheduler_jobs()
    backups = load_backup_history()

    tabs = st.tabs(["Ops Health", "Model Performance", "Run History", "Errors", "Models", "Audit"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Scheduler Jobs")
            if scheduler_jobs.empty:
                st.info("No scheduler jobs recorded.")
            else:
                st.dataframe(scheduler_jobs, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Backups")
            if backups.empty:
                st.info("No backups recorded.")
            else:
                st.dataframe(backups, use_container_width=True, hide_index=True)

    with tabs[1]:
        render_model_performance_dashboard(model_versions)

    with tabs[2]:
        st.dataframe(run_history, use_container_width=True, hide_index=True)
    with tabs[3]:
        st.dataframe(error_logs, use_container_width=True, hide_index=True)
    with tabs[4]:
        st.dataframe(model_versions, use_container_width=True, hide_index=True)
    with tabs[5]:
        if not universe_audits.empty:
            st.dataframe(
                universe_audits[["audit_time", "triggered_by", "tickers_audited", "pass_count", "warn_count", "fail_count", "average_score", "median_score"]],
                use_container_width=True,
                hide_index=True,
            )
        st.subheader("Latest Ticker Audits")
        st.dataframe(latest_audits, use_container_width=True, hide_index=True)


def render_model_performance_dashboard(versions: pd.DataFrame) -> None:
    st.subheader("Model Performance & Observability")
    if versions.empty:
        st.info("No model versions found in registry.")
        return

    # Filter to active model or latest ensemble
    active_mask = (versions["active"] == True) | (versions["model_name"].str.contains("ensemble"))
    eligible = versions[active_mask]
    if eligible.empty:
        eligible = versions.head(1)
        
    selected_version = st.selectbox(
        "Select Model Version",
        options=eligible["version"].tolist(),
        format_func=lambda v: f"{v} ({eligible[eligible['version']==v]['model_name'].iloc[0]})"
    )
    
    version_row = eligible[eligible["version"] == selected_version].iloc[0]
    meta = version_row.get("metadata", {})
    
    # 1. High Level Metrics
    m_cols = st.columns(4)
    metrics = meta.get("metrics", {})
    m_cols[0].metric("WF AUC", format_decimal(metrics.get("wf_auc"), 3))
    m_cols[1].metric("Test AUC", format_decimal(metrics.get("test_auc"), 3))
    m_cols[2].metric("Brier Score", format_decimal(metrics.get("brier_score"), 4))
    m_cols[3].metric("Samples", format_count(meta.get("train_samples", 0)))

    c1, c2 = st.columns(2)
    
    # 2. SHAP Feature Importance
    with c1:
        st.write("#### Feature Importance (SHAP)")
        importance_data = meta.get("feature_importance", [])
        if importance_data:
            imp_df = pd.DataFrame(importance_data).sort_values("shap_mean", ascending=True)
            fig = px.bar(imp_df, x="shap_mean", y="feature", orientation="h",
                         color="shap_mean", color_continuous_scale="Viridis")
            fig.update_layout(height=400, margin=dict(t=0, b=0, l=0, r=0), showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SHAP data not available for this version.")

    # 3. Calibration Curve
    with c2:
        st.write("#### Reliability Curve")
        cal = meta.get("calibration", {})
        if cal and "mean_predicted" in cal:
            fig = go.Figure()
            # Perfectly calibrated line
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", 
                                     line=dict(color="white", dash="dash")))
            # Model calibration
            fig.add_trace(go.Scatter(x=cal["mean_predicted"], y=cal["fraction_pos"], 
                                     mode="lines+markers", name="Model",
                                     marker=dict(size=8, color="#38bdf8")))
            fig.update_layout(height=400, margin=dict(t=0, b=0, l=0, r=0),
                              xaxis_title="Average Probability", yaxis_title="Actual Outcome share",
                              hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Calibration data not available.")


def maybe_run_scan(page_name: str) -> None:
    if query_param_value("run_scan", "0").lower() == "1":
        try:
            with st.spinner(f"Running {config.DEFAULT_SCAN_UNIVERSE} scan..."):
                orchestrator = get_pipeline_orchestrator()
                result = asyncio.run(orchestrator.run(get_universe(config.DEFAULT_SCAN_UNIVERSE)))
                st.success(f"Scan complete: {result.summary}")
                invalidate_cached_views()
                st.query_params.update({"page": page_name})
                st.rerun()
        except Exception as e:
            st.error(f"Scan failed: {e}")


def main() -> None:
    inject_global_styles()
    page_name = normalize_page_name(query_param_value("page", "Dashboard"))
    regime_snapshot = load_market_snapshot("regime")
    vix_snapshot = load_market_snapshot("india_vix")
    render_top_navigation(regime_snapshot, vix_snapshot)
    maybe_run_scan(page_name)

    frame = load_core_dataset()
    if frame.empty and page_name == "Dashboard":
        st.warning("No core datasets found. Please run a 'scan' from the CLI or UI. Some features may not render.")

    dashboard_frame = build_dashboard_frame(frame)
    swing_df = load_swing_dataset()
    mb_df = load_multibagger_dataset()
    run_history = load_run_history()
    alert_logs = load_alert_logs()

    if page_name == "Dashboard":
        render_dashboard(dashboard_frame, swing_df, mb_df, regime_snapshot, vix_snapshot, alert_logs, run_history)
    elif page_name == "Swing Trades":
        render_swing_page(swing_df, dashboard_frame)
    elif page_name == "Positionals":
        render_positionals_page(dashboard_frame, regime_snapshot)
    elif page_name == "Multibagger Hunt":
        render_multibagger_page(mb_df, dashboard_frame)
    elif page_name == "Fair Value":
        render_fair_value_page(dashboard_frame)
    elif page_name == "Sector Rank":
        render_sector_rank_page(dashboard_frame)
    elif page_name == "Ownership":
        render_ownership_page(dashboard_frame)
    elif page_name == "Risk Monitor":
        render_risk_page(dashboard_frame, vix_snapshot)
    elif page_name == "Portfolio":
        render_portfolio_page(dashboard_frame)
    elif page_name == "Backtest":
        render_backtest_page()
    elif page_name == "Alerts Feed":
        render_alerts_page(alert_logs)
    elif page_name == "Logs":
        render_logs_page()
    else:
        st.header(f"{page_name} module")
        st.info("Page routing is available but this module has no renderer yet.")


if __name__ == "__main__":
    main()
