import pandas as pd
import json

try:
    file_path = r"d:\Tradeidesa\Multibagger-Ultraversion\RelativeStrength (1).xlsx"
    xl = pd.ExcelFile(file_path)
    df = xl.parse("Stocks")
    
    # Let's see the ranking columns
    rank_cols = [c for c in df.columns if 'RANK' in c.upper()]
    print(f"Ranking columns: {rank_cols}")
    
    # I'll use the first ranking column found
    if rank_cols:
        target_rank = rank_cols[0]
        # Sort by rank (ascending, assuming 1 is best)
        # But wait, sometimes rank is a score. Let's check values.
        print(f"Sample values for {target_rank}: {df[target_rank].head().tolist()}")
        
        # If it's 1, 2, 3... sort ascending. If it's scores like 99, 98... sort descending.
        avg_val = df[target_rank].mean()
        if avg_val < 500: # Typical for rank 1-500
             df_sorted = df.sort_values(by=target_rank, ascending=True)
        else:
             df_sorted = df.sort_values(by=target_rank, ascending=False)
    else:
        df_sorted = df

    # Extract symbols
    symbols = df_sorted['Symbol'].dropna().head(100).astype(str).tolist()
    
    # Process symbols
    final_symbols = []
    for s in symbols:
        s = s.strip().upper()
        # Remove any unwanted suffixes or characters if they exist
        # Just ensure they are clean tickers
        final_symbols.append(s)

    print("---TOP_100_START---")
    print(json.dumps(final_symbols))
    print("---TOP_100_END---")

except Exception as e:
    print(f"Error: {e}")
