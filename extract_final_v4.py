import pandas as pd
import json

try:
    file_path = r"d:\Tradeidesa\Multibagger-Ultraversion\RelativeStrength (1).xlsx"
    xl = pd.ExcelFile(file_path)
    df = xl.parse("Stocks")
    
    # Sorting column
    rank_cols = [c for c in df.columns if 'RANK' in c.upper()]
    if rank_cols:
        target_col = rank_cols[0]
        df_sorted = df.sort_values(by=target_col, ascending=True)
    else:
        df_sorted = df

    # Convert Symbol column to string safely using pandas
    df_sorted['Symbol_Str'] = df_sorted['Symbol'].astype(str).str.strip().str.upper()
    
    # Extract top 100 unique symbols that look like tickers
    # Tickers are typically alphanumeric, 3-15 chars, no spaces
    symbols = []
    for s in df_sorted['Symbol_Str']:
        if pd.isna(s) or s == 'NAN' or s == 'NONE' or '00:00:00' in s:
            continue
            
        # Basic ticker validation (at least 2 chars, no common date patterns)
        if len(s) >= 2 and not any(month in s for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']):
             if s not in symbols:
                 symbols.append(s)
        if len(symbols) >= 100:
            break
            
    print("---TOP_100_START---")
    print(json.dumps(symbols))
    print("---TOP_100_END---")

except Exception as e:
    print(f"Error: {e}")
