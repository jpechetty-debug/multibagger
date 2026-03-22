import pandas as pd
import json

try:
    file_path = r"d:\Tradeidesa\Multibagger-Ultraversion\RelativeStrength (1).xlsx"
    xl = pd.ExcelFile(file_path)
    df = xl.parse("Stocks")
    
    # Ranking column
    rank_cols = [c for c in df.columns if 'RANK' in str(c).upper()]
    if rank_cols:
        target_col = rank_cols[0]
        df_sorted = df.sort_values(by=target_col, ascending=True)
    else:
        df_sorted = df

    # Extract symbols manually to be safe
    symbols = []
    symbol_list_raw = df_sorted['Symbol'].dropna().tolist()
    
    for s in symbol_list_raw:
        # Convert to string and handle potential date mangling
        s_str = str(s).strip().upper()
        
        # Filter out obvious non-tickers (like dates: '2024-03-01' or 'MAR-24')
        if "00:00:00" in s_str or len(s_str) < 2:
            continue
        
        # Heuristic: Tickers don't usually contain months
        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        if any(m in s_str for m in months) and ("-" in s_str or "/" in s_str):
            continue
            
        if s_str not in symbols:
            symbols.append(s_str)
            
        if len(symbols) >= 100:
            break
            
    print("---TOP_100_START---")
    print(json.dumps(symbols))
    print("---TOP_100_END---")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
