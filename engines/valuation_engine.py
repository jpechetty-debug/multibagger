"""Valuation engine for intrinsic value estimates."""

from __future__ import annotations

import math
import statistics
import pandas as pd

import config
from data.db import db
from data.fetcher import DataFetcher
from engines.analysis._common import current_timestamp, load_financial_statements
from models.schemas import FundamentalData, ValuationResult


def _latest_value(frame: pd.DataFrame, names: list[str]) -> float | None:
    """Return the latest numeric value for a matching row."""

    for name in names:
        if name in frame.index:
            series = frame.loc[name]
            if isinstance(series, pd.Series):
                series = series.dropna().sort_index(ascending=False)
                if not series.empty:
                    return float(series.iloc[0])
    return None


class ValuationEngine:
    """Computes multiple fair-value estimates and stores the result."""

    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        """Initialize the valuation engine."""

        self.fetcher = fetcher or DataFetcher()

    def _sector_median_pe(self, data: FundamentalData) -> float | None:
        """Return the sector median PE from stored fundamentals."""

        if not data.sector:
            return None
        peer_pes = [
            record.pe_ratio
            for record in db.list_fundamentals(effective=True)
            if record.ticker != data.ticker
            and record.sector == data.sector
            and record.pe_ratio is not None
            and record.pe_ratio > 0
        ]
        return float(statistics.median(peer_pes)) if peer_pes else None

    def _baseline_pe(self, data: FundamentalData, sector_median_pe: float | None, template: dict | None = None) -> float:
        """Return the PE multiple to use for fallback valuations."""

        template = template or config.valuation_template_for_sector(data.sector)
        pe = data.pe_ratio or sector_median_pe or config.EPS_VALUATION_DEFAULT_PE
        return float(min(max(pe, template.get("pe_floor", 8.0)), template.get("pe_ceiling", 35.0)))

    def _derived_eps(
        self,
        data: FundamentalData,
        statements: dict[str, pd.DataFrame],
        sector_median_pe: float | None,
        template: dict | None = None,
    ) -> float | None:
        """Derive EPS from statements, then fall back to implied EPS when needed."""

        financials = statements["financials"]
        balance_sheet = statements["balance_sheet"]
        net_income = _latest_value(
            financials,
            ["Net Income", "Net Income From Continuing Operation Net Minority Interest"],
        )
        shares = _latest_value(balance_sheet, ["Ordinary Shares Number", "Share Issued"])

        eps = None
        if net_income is not None and shares not in (None, 0):
            eps = net_income / shares

        fallback_pe = self._baseline_pe(data, sector_median_pe, template)
        implied_eps = None
        if data.price not in (None, 0):
            implied_eps = data.price / fallback_pe

        if implied_eps is not None and (eps is None or eps < implied_eps * 0.1):
            eps = implied_eps
        return eps

    def _peer_reference_value(
        self,
        data: FundamentalData,
        statements: dict[str, pd.DataFrame],
        sector_median_pe: float | None,
        template: dict | None = None,
    ) -> float | None:
        """Return a peer-based reference valuation for DCF cross-checking."""

        eps = self._derived_eps(data, statements, sector_median_pe, template)
        if eps is None or eps <= 0:
            return data.price
        return eps * self._baseline_pe(data, sector_median_pe, template)

    def DCF(
        self,
        data: FundamentalData,
        statements: dict[str, pd.DataFrame],
        sector_median_pe: float | None = None,
        template: dict | None = None,
    ) -> float | None:
        """Estimate fair value from discounted free cash flow per share."""

        template = template or config.valuation_template_for_sector(data.sector)
        cashflow = statements["cashflow"]
        balance_sheet = statements["balance_sheet"]
        financials = statements["financials"]
        free_cash_flow = _latest_value(cashflow, ["Free Cash Flow"])
        shares = _latest_value(balance_sheet, ["Ordinary Shares Number", "Share Issued"])
        net_income = _latest_value(financials, ["Net Income", "Net Income From Continuing Operation Net Minority Interest"])
        
        if free_cash_flow is None and net_income is not None and data.cfo_to_pat is not None:
            free_cash_flow = net_income * max(0.25, min(3.0, data.cfo_to_pat))

        if free_cash_flow is None or shares in (None, 0):
            return None

        fcf_per_share = free_cash_flow / shares

        implied_eps = self._derived_eps(data, statements, sector_median_pe, template)
        if implied_eps is not None and fcf_per_share < implied_eps * 0.1:
            # yfinance share-count units can drift; anchor FCF/share to earnings when off by 10x.
            fcf_per_share = implied_eps * 0.8

        growth_floor = template.get("growth_floor", 0.01)
        growth_ceiling = template.get("growth_ceiling", 0.18)
        growth = max(growth_floor, min(growth_ceiling, data.sales_growth_5y or data.eps_growth_ttm or 0.08))
        present_value = 0.0
        current_cash = fcf_per_share
        
        discount_rate = template.get("discount_rate", config.DCF_DISCOUNT_RATE)
        terminal_growth = template.get("terminal_growth", config.DCF_TERMINAL_GROWTH)
        projection_years = template.get("projection_years", config.DCF_PROJECTION_YEARS)

        for year in range(1, projection_years + 1):
            current_cash *= 1 + growth
            present_value += current_cash / ((1 + discount_rate) ** year)
        terminal_cash = current_cash * (1 + terminal_growth)
        terminal_value = terminal_cash / max(0.01, discount_rate - terminal_growth)
        present_value += terminal_value / ((1 + discount_rate) ** projection_years)
        return float(max(0.0, present_value))

    def eps_valuation(
        self,
        data: FundamentalData,
        statements: dict[str, pd.DataFrame],
        sector_median_pe: float | None = None,
        template: dict | None = None,
    ) -> float | None:
        """Estimate fair value from earnings per share and a justified PE."""

        template = template or config.valuation_template_for_sector(data.sector)
        eps = self._derived_eps(data, statements, sector_median_pe, template)
        if eps is None:
            return None

        growth_floor = template.get("growth_floor", 0.0)
        growth_ceiling = template.get("growth_ceiling", 0.20)
        growth = max(growth_floor, min(growth_ceiling, data.eps_growth_ttm or data.sales_growth_5y or 0.06))
        forward_eps = eps * (1 + growth)
        justified_pe = self._baseline_pe(data, sector_median_pe, template)
        return float(max(0.0, forward_eps * justified_pe))

    def graham_number(
        self,
        data: FundamentalData,
        statements: dict[str, pd.DataFrame],
        sector_median_pe: float | None = None,
        template: dict | None = None,
    ) -> float | None:
        """Estimate fair value using the Graham number."""

        balance_sheet = statements["balance_sheet"]
        equity = _latest_value(balance_sheet, ["Common Stock Equity"])
        shares = _latest_value(balance_sheet, ["Ordinary Shares Number", "Share Issued"])
        eps = self._derived_eps(data, statements, sector_median_pe, template)

        if eps is None or eps <= 0:
            return None

        if equity is None or shares in (None, 0):
            return None
        bvps = equity / shares
        if bvps <= 0:
            return None

        return float(math.sqrt(config.GRAHAM_NUMBER_FACTOR * eps * bvps))

    def peg_valuation(
        self,
        data: FundamentalData,
        statements: dict[str, pd.DataFrame],
        sector_median_pe: float | None = None,
        template: dict | None = None,
    ) -> float | None:
        """Estimate fair value from PEG-implied PE multiple."""

        template = template or config.valuation_template_for_sector(data.sector)
        growth = data.eps_growth_ttm
        eps = self._derived_eps(data, statements, sector_median_pe, template)

        if eps is None or growth is None or growth <= 0:
            return None

        pe_floor = template.get("pe_floor", 6.0)
        pe_ceiling = template.get("pe_ceiling", 35.0)
        implied_pe = max(pe_floor, min(pe_ceiling, growth * 100 * config.PEG_TARGET))
        return float(max(0.0, eps * implied_pe))

    def value_ticker(self, ticker: str, data: FundamentalData | None = None) -> ValuationResult:
        """Compute, persist, and return valuation estimates for a ticker."""

        normalized_ticker = ticker.strip().upper()
        data = data or db.get_fundamental(normalized_ticker, effective=True) or self.fetcher.fetch(normalized_ticker)
        statements = load_financial_statements(normalized_ticker)
        sector_median_pe = self._sector_median_pe(data)
        template = config.valuation_template_for_sector(data.sector)

        # Compute models with safety
        models = [
            ("DCF", self.DCF),
            ("EPS", self.eps_valuation),
            ("Graham", self.graham_number),
            ("PEG", self.peg_valuation),
        ]

        results = {}
        for name, func in models:
            try:
                results[name] = func(data, statements, sector_median_pe, template)
            except Exception as e:
                db.log_engine_event("WARN", "engines.valuation_engine", f"{name} model failed", {"ticker": normalized_ticker, "error": str(e)})
                results[name] = None

        dcf_value = results["DCF"]
        eps_value = results["EPS"]
        graham_value = results["Graham"]
        peg_value = results["PEG"]

        peer_reference = self._peer_reference_value(data, statements, sector_median_pe)
        if (
            dcf_value is not None
            and dcf_value > 0
            and peer_reference not in (None, 0)
            and (dcf_value > peer_reference * 3 or dcf_value < peer_reference / 3)
        ):
            db.log_engine_event(
                "WARN",
                "engines.valuation_engine",
                "dcf deviates materially from peer reference",
                {
                    "ticker": normalized_ticker,
                    "dcf_value": dcf_value,
                    "peer_reference": peer_reference,
                    "sector_median_pe": sector_median_pe,
                },
            )

        candidate_values = [
            ("DCF", dcf_value),
            ("EPS", eps_value),
            ("Graham", graham_value),
            ("PEG", peg_value),
        ]
        fair_candidates = [
            value
            for name, value in candidate_values
            if value is not None
            and value > 1.0
            and (
                name != "DCF"
                or peer_reference in (None, 0)
                or peer_reference / 3 <= value <= peer_reference * 3
            )
        ]
        fair_value = min(fair_candidates) if fair_candidates else None

        valid_values = [v for _, v in candidate_values if v is not None and v > 1.0]
        valuation_confidence = 0.0
        if not valid_values:
            valuation_confidence = 0.0
        elif len(valid_values) == 1:
            valuation_confidence = 30.0
        else:
            spread_ratio = max(valid_values) / min(valid_values)
            if spread_ratio > config.VALUATION_MODEL_SPREAD_QUARANTINE:
                valuation_confidence = 10.0
            elif spread_ratio > config.VALUATION_MODEL_SPREAD_WARN:
                valuation_confidence = 45.0
            else:
                valuation_confidence = max(0.0, min(100.0, 100.0 - (spread_ratio - 1.0) * 50.0))

        if fair_value and data.price and fair_value < (data.price * 0.05):
            fair_value = data.price
            valuation_confidence = 0.0
            db.log_engine_event(
                "WARN",
                "engines.valuation_engine",
                "fair value suspiciously low, using price fallback",
                {"ticker": normalized_ticker, "fair_value": fair_value},
            )

        if (fair_value is None or fair_value <= 0) and data.price is not None and data.price > 0:
            pe = self._baseline_pe(data, sector_median_pe, template)
            growth = max(0.02, data.eps_growth_ttm or data.sales_growth_5y or 0.06)
            implied_eps = data.price / pe
            forward_eps = implied_eps * (1 + growth)
            fair_value = forward_eps * min(pe * 1.1, 30.0)
            valuation_confidence = 20.0  # low confidence if purely fallback
            db.log_engine_event(
                "INFO",
                "engines.valuation_engine",
                "fair value derived from fallback EPS multiple",
                {
                    "ticker": normalized_ticker,
                    "fair_value": fair_value,
                    "pe_used": pe,
                    "sector_median_pe": sector_median_pe,
                },
            )

        margin_of_safety_pct = None
        undervalued = False
        if fair_value is not None and data.price not in (None, 0):
            margin_of_safety_pct = fair_value / data.price - 1
            undervalued = data.price < config.MARGIN_OF_SAFETY_FACTOR * fair_value
            
        result = ValuationResult(
            ticker=normalized_ticker,
            dcf_value=dcf_value,
            eps_value=eps_value,
            graham_value=graham_value,
            peg_value=peg_value,
            fair_value=fair_value,
            margin_of_safety_pct=margin_of_safety_pct,
            undervalued=undervalued,
            valuation_confidence=valuation_confidence,
            generated_at=current_timestamp(),
        )
        db.save_valuation_result(result)
        db.log_engine_event("INFO", "engines.valuation_engine", "valuation complete", result.model_dump())
        return result


if __name__ == "__main__":
    print(ValuationEngine().value_ticker("RELIANCE").model_dump())
