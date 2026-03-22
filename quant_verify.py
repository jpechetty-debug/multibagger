import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from engines.quant_orchestrator import QuantOrchestrator

def test_quant_orchestrator():
    print("Testing Quant Orchestrator...")
    
    # 1. Mock Market Data
    market_data = {
        "nifty_close": 22000,
        "nifty_sma50": 21500,
        "nifty_sma200": 20000,
        "nifty_sma50_slope": 0.1,
        "nifty_sma200_slope": 0.05,
        "pct_above_200dma": 65,
        "advance_count": 300,
        "decline_count": 200,
        "india_vix": 13.4,
        "net_fii_20d_cr": 2000,
        "ad_ratio_10d_avg": 1.2,
        "midcap_vs_nifty_30d": 0.02,
        "gsec_10y": 6.85,
        "repo_rate": 6.5,
        "pct_rsi_above_50": 60,
    }
    
    # 2. Mock Signals
    signals = [
        {"ticker": "HDFCBANK", "sector": "Financials", "total_score": 85, "price": 1450, "action": "BUY"},
        {"ticker": "ICICIBANK", "sector": "Financials", "total_score": 82, "price": 1050, "action": "BUY"},
        {"ticker": "RELIANCE", "sector": "Energy", "total_score": 78, "price": 2900, "action": "BUY"},
        {"ticker": "TCS", "sector": "IT", "total_score": 76, "price": 4000, "action": "BUY"},
        {"ticker": "INFY", "sector": "IT", "total_score": 74, "price": 1600, "action": "WATCH"},
        {"ticker": "SBIN", "sector": "Financials", "total_score": 88, "price": 750, "action": "BUY"},
        {"ticker": "AXISBANK", "sector": "Financials", "total_score": 80, "price": 1100, "action": "BUY"},
        {"ticker": "BHARTIARTL", "sector": "Telecommunication", "total_score": 75, "price": 1200, "action": "BUY"},
        {"ticker": "LT", "sector": "Capital Goods", "total_score": 83, "price": 3500, "action": "BUY"},
        {"ticker": "MARUTI", "sector": "Auto", "total_score": 77, "price": 11000, "action": "BUY"},
    ]
    
    # 3. Mock Returns (20 stocks x 100 days)
    tickers = [s["ticker"] for s in signals]
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
    data = np.random.normal(0.0005, 0.015, (100, len(tickers)))
    returns_df = pd.DataFrame(data, index=dates, columns=tickers)
    
    # 4. Mock Universe Fundamentals
    universe_df = pd.DataFrame({
        "pe_ratio": [25.0] * 10,
        "roic_current": [15.0] * 10,
        "market_cap": [100000] * 10,
        "price_return_3m": [0.1] * 10,
    }, index=tickers)
    
    orchestrator = QuantOrchestrator(total_capital=10000000) # 1 Cr
    result = orchestrator.run(
        signals=signals,
        market_data=market_data,
        returns_df=returns_df,
        universe_df=universe_df,
    )
    
    print(f"\nSummary: {result.summary()}")
    print(f"Regime: {result.regime.regime} (Score: {result.regime.composite_score:.1f})")
    print(f"Confidence: {result.regime.confidence:.2f}")
    
    print("\nPortfolio Allocation:")
    res_dict = result.to_dict()
    for pos in res_dict["portfolio"]["positions"]:
        print(f"  {pos['ticker']} ({pos['sector']}): {pos['position_pct']:.2f}% | ₹{pos['capital_allocated']:,.0f}")
        
    print("\nSector Weights:")
    for sector, weight in res_dict["portfolio"]["sector_weights"].items():
        print(f"  {sector}: {weight*100:.1f}%")

    print("\nRisk Status:")
    print(f"  Action Required: {result.risk.action_required}")
    print(f"  Risk Summary: {result.risk.risk_summary}")
    
    # Verification assertions
    assert result.regime.regime in ["BULL", "QUALITY", "SIDEWAYS", "BEAR"]
    assert len(result.portfolio.positions) > 0
    assert sum(pos.weight for pos in result.portfolio.positions) <= 1.0001
    
    # Check sector constraint (Financials capped at 30%)
    fin_weight = res_dict["portfolio"]["sector_weights"].get("Financials", 0)
    print(f"\nFinancials weight: {fin_weight*100:.1f}%")
    assert fin_weight <= 0.3001
    
    print("\nQUANT ORCHESTRATOR VERIFIED SUCCESSFULLY")

if __name__ == "__main__":
    test_quant_orchestrator()
