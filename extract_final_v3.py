import pandas as pd
import json
import datetime

try:
    file_path = r"d:\Tradeidesa\Multibagger-Ultraversion\RelativeStrength (1).xlsx"
    xl = pd.ExcelFile(file_path)
    df = xl.parse("Stocks")
    
    # Ranking column
    rank_cols = [c for c in df.columns if 'RANK' in c.upper()]
    if rank_cols:
        target_rank = rank_cols[0]
        df_sorted = df.sort_values(by=target_rank, ascending=True)
    else:
        df_sorted = df

    def clean_symbol(s):
        if pd.isna(s): return None
        # Convert to string first
        s_str = str(s).strip().upper()
        # Common Excel date mangling for tickers like 'IRFC' -> '2024-03-01'
        # If it looks like a date, we might need the original if possible,
        # but usually, these files are exports where mangling happened at source.
        # I'll just skip anything that doesn't look like a ticker (all digits, or contains spaces/dashes)
        # unless it's a known format.
        if "00:00:00" in s_str: # It's a timestamp string
            return None
        return s_str

    symbols = []
    for s in df_sorted['Symbol'].dropna():
        cleaned = clean_symbol(s)
        if cleaned and cleaned not in symbols:
            symbols.append(cleaned)
        if len(symbols) >= 100:
            break
    
    # Final check: remove anything that's clearly not a ticker (e.g. '1/1/2024')
    # Valid tickers are usually alphanumeric.
    import re
    valid_tickers = [s for s in symbols if re.match(r'^[A-Z0-9&\-]+$', s)]

    print("---TOP_100_START---")
    print(json.dumps(valid_tickers))
    print("---TOP_100_END---")

except Exception as e:
    print(f"Error: {e}")
