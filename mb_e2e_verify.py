import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from data.fetcher import DataFetcher
from engines.multibagger.conviction_scorer import ConvictionScorer
from models.schemas import FundamentalData

def test_e2e_scoring():
    print("Testing End-to-End Multibagger Flow...")
    
    # 1. Mock a FundamentalData object with history in source_metadata
    # This simulates what DataFetcher.fetch() would now return
    data = FundamentalData(
        ticker="RELIANCE",
        company_name="Reliance Industries",
        sector="Energy",
        price=2900.0,
        debt_equity=0.4,
        promoter_pct=50.6,
        roe_ttm=12.5,
        sales_growth_5y=15.0,
        eps_growth_ttm=18.0,
        source_metadata={
            "roic_history": [10.5, 11.2, 11.8, 12.5],
            "eps_history": [45.0, 52.0, 60.0, 72.0],
            "de_history": [0.45, 0.42, 0.40],
            "price_vs_200dma_pct": 8.5,
            "valuation_percentile": 60.0,
        }
    )
    
    scorer = ConvictionScorer()
    # We pass the mock data directly to score_ticker
    candidate = scorer.score_ticker("RELIANCE", data=data)
    
    print(f"Ticker: {candidate.ticker}")
    print(f"Action: {candidate.action}")
    print(f"Conviction Score: {candidate.conviction_score}")
    print(f"Reasoning Sample: {candidate.reasoning[0]}")
    
    assert candidate.conviction_score > 0
    assert len(candidate.reasoning) > 0
    print("\nE2E FLOW VERIFIED SUCCESSFULLY")

if __name__ == "__main__":
    try:
        test_e2e_scoring()
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
