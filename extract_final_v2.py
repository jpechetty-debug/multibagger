import pandas as pd
import json

try:
    file_path = r"d:\Tradeidesa\Multibagger-Ultraversion\RelativeStrength (1).xlsx"
    xl = pd.ExcelFile(file_path)
    df = xl.parse("Stocks")
    
    # Ranking column
    rank_cols = [c for c in df.columns if 'RANK' in c.upper()]
    if rank_cols:
        target_rank = rank_cols[0]
        # Sort by rank (ascending 1,2,3... usually)
        df_sorted = df.sort_values(by=target_rank, ascending=True)
    else:
        df_sorted = df

    # Extract symbols with conversion to string to avoid datetime issues
    # Date-like tickers in Excel (e.g., 'IRFC' or something interpreted as a month-year) 
    # should be strings.
    def clean_symbol(s):
        if pd.isna(s): return None
        if isinstance(s, (pd.Timestamp, datetime.datetime)):
            # If it's a date, we might have lost the actual ticker if it was mangled
            # But let's try to convert but maybe skip if it's clearly not a ticker
            return str(s.strftime('%b-%y')) if hasattr(s, 'strftime') else str(s)
        return str(s).strip().upper()

    import datetime
    
    symbols = []
    for s in df_sorted['Symbol'].dropna().head(100):
        cleaned = clean_symbol(s)
        if cleaned:
            # Common date mangling: ICICIBANK -> 1CICI... or something? Not common.
            # More likely: 01-01 -> Jan-01
            # If it looks like a month-year, it might be a mangled ticker.
            # I'll just keep it as string for now.
            symbols.append(cleaned)
    
    print("---TOP_100_START---")
    print(json.dumps(symbols))
    print("---TOP_100_END---")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
