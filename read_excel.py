import pandas as pd
import json
import sys
import os

try:
    file_path = r"d:\Tradeidesa\Multibagger-Ultraversion\RelativeStrength (1).xlsx"
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
        
    df = pd.read_excel(file_path)
    
    # Identify symbol column
    symbol_col = None
    for col in df.columns:
        c_lower = str(col).lower()
        if any(kw in c_lower for kw in ['symbol', 'ticker', 'stock']):
            symbol_col = col
            break
            
    if not symbol_col:
        # Check first column
        symbol_col = df.columns[0]
        
    # Identify RS/Score column
    rs_col = None
    rs_keywords = ['rs', 'relative', 'strength', 'score', 'rank', 'alpha']
    for col in df.columns:
        c_lower = str(col).lower()
        if any(kw in c_lower for kw in rs_keywords):
            rs_col = col
            break
            
    if rs_col:
        # Sort by RS score descending (assuming higher is better)
        df_sorted = df.sort_values(by=rs_col, ascending=False)
    else:
        # Just take the first 100 if no score found
        df_sorted = df

    top_100 = df_sorted[symbol_col].dropna().head(100).astype(str).tolist()
    
    # Process symbols (strip, uppercase, add .NS suffix if missing)
    processed_symbols = []
    for s in top_100:
        s = s.strip().upper()
        if not s.endswith(".NS"):
            # Check if it looks like an Indian ticker
            processed_symbols.append(f"{s}.NS")
        else:
            processed_symbols.append(s)
            
    print("---TOP_100_START---")
    print(json.dumps(processed_symbols))
    print("---TOP_100_END---")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
