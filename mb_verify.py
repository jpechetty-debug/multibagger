import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from engines.multibagger import MultibaggerScorer

def test_multibagger_scorer():
    print("Testing Multibagger Scorer...")
    
    # HDFCBANK mock data (roughly)
    data = {
        "ticker": "HDFCBANK",
        "sector": "Financials",
        "roic_current": 16.5,
        "roic_history": [15.0, 15.5, 16.0, 16.5],
        "eps_history": [50.0, 60.0, 75.0, 92.0],
        "debt_to_equity": 0.1,  # standalone bank entity de is low usually
        "fcf_yield": 0.045,
        "revenue_cagr_3y": 0.18,
        "eps_cagr_3y": 0.22,
        "fwd_eps_growth_pct": 20.0,
        "tam_runway_score": 85.0,
        "valuation_percentile": 25.0,
        "margin_of_safety_pct": 30.0,
        "promoter_pct": 25.5, # Boundary case
        "pledge_pct": 0.0,
        "fii_delta": 0.015,
        "dii_delta": 0.005,
        "insider_buys_90d": 2,
        "price_vs_200dma_pct": 12.0,
        "relative_strength_3m": 0.06,
        "rank_percentile": 85.0,
        "interest_coverage": 15.0,
    }
    
    scorer = MultibaggerScorer()
    result = scorer.score("HDFCBANK", data)
    
    res_dict = result.to_dict()
    print(f"Ticker: {res_dict['ticker']}")
    print(f"Tier: {res_dict['tier']}")
    print(f"Composite: {res_dict['composite']}")
    print(f"Narrative: {res_dict['narrative']}")
    
    assert res_dict["tier"] in ("ELITE", "STRONG", "WATCH", "REJECT")
    assert result.gates_passed is True
    
    # Test a failure case (High pledge)
    print("\nTesting Failure Case (High Pledge)...")
    data["pledge_pct"] = 20.0
    result_fail = scorer.score("HDFCBANK", data)
    print(f"Tier: {result_fail.tier}")
    print(f"Gate Failures: {result_fail.gate_failures}")
    assert result_fail.tier == "REJECT"
    assert "Pledge" in result_fail.gate_failures[0]

    print("\nMULTIBAGGER SCORER VERIFIED SUCCESSFULLY")

if __name__ == "__main__":
    try:
        test_multibagger_scorer()
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
