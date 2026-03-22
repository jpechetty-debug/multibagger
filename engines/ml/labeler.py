"""Label generation for ML training."""

import json
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from data.db import db
from ticker_list import to_yfinance

class PointInTimeLabeler:
    """Generates forward return labels for point-in-time fundamentals."""

    def __init__(self) -> None:
        self.price_cache: dict[str, pd.DataFrame | None] = {}

    def _get_price(self, ticker: str, target_date: datetime) -> float | None:
        """Fetch the closest closing price on or after the target date."""
        if ticker not in self.price_cache:
            try:
                # Fetch 2 years of history to allow forward labels
                hist = yf.Ticker(to_yfinance(ticker)).history(period="2y")
                if not hist.empty:
                    self.price_cache[ticker] = hist
                else:
                    self.price_cache[ticker] = None
            except Exception:
                self.price_cache[ticker] = None

        hist = self.price_cache[ticker]
        if hist is None or hist.empty:
            return None

        if hist.index.tz is not None:
            # ensure target_date has the same timezone
            if target_date.tzinfo is None:
                import pytz
                target_date = target_date.replace(tzinfo=pytz.UTC)
            
        target_date = pd.to_datetime(target_date).normalize()
        if hist.index.tz is not None and target_date.tzinfo is not None:
            target_date = target_date.tz_convert(hist.index.tz)
            
        future_prices = hist[hist.index >= target_date]
        if future_prices.empty:
            return None
            
        return float(future_prices.iloc[0]["Close"])

    def generate_labeled_dataset(self) -> pd.DataFrame:
        """Extract all PIT records and attach forward returns."""
        db.initialize()
        records = []
        with db.connection("pit") as conn:
            rows = conn.execute("SELECT ticker, captured_at, fundamentals_json FROM fundamentals_pit ORDER BY captured_at ASC").fetchall()
            for row in rows:
                ticker = row["ticker"]
                captured_at = row["captured_at"]
                
                try:
                    fundamentals = json.loads(row["fundamentals_json"])
                except Exception:
                    continue
                    
                capture_date = datetime.fromtimestamp(captured_at)
                
                base_price = self._get_price(ticker, capture_date)
                if not base_price:
                    continue
                    
                price_1m = self._get_price(ticker, capture_date + timedelta(days=30))
                price_3m = self._get_price(ticker, capture_date + timedelta(days=90))
                price_6m = self._get_price(ticker, capture_date + timedelta(days=180))
                
                ret_1m = (price_1m / base_price - 1.0) if price_1m else None
                ret_3m = (price_3m / base_price - 1.0) if price_3m else None
                ret_6m = (price_6m / base_price - 1.0) if price_6m else None
                
                record = {
                    "ticker": ticker,
                    "captured_at": captured_at,
                    "date": capture_date.strftime("%Y-%m-%d"),
                    "forward_1m_ret": ret_1m,
                    "forward_3m_ret": ret_3m,
                    "forward_6m_ret": ret_6m,
                }
                
                numeric_keys = [
                    "market_cap", "avg_volume", "roe_5y", "roe_ttm", "sales_growth_5y", 
                    "eps_growth_ttm", "cfo_to_pat", "debt_equity", "peg_ratio", "pe_ratio", 
                    "piotroski_score", "promoter_pct", "pledge_pct", "fii_delta", "dii_delta"
                ]
                for key in numeric_keys:
                    val = fundamentals.get(key)
                    record[key] = float(val) if val is not None else None
                    
                records.append(record)
                
        return pd.DataFrame(records)
