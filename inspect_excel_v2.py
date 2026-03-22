import pandas as pd
import json

try:
    file_path = r"d:\Tradeidesa\Multibagger-Ultraversion\RelativeStrength (1).xlsx"
    xl = pd.ExcelFile(file_path)
    print(f"Sheet names: {xl.sheet_names}")
    
    # Check the first few rows of the first sheet properly
    df = xl.parse(xl.sheet_names[0])
    print(f"\nColumns ({xl.sheet_names[0]}): {df.columns.tolist()}")
    print("\nFirst 10 rows:")
    print(df.head(10).to_dict(orient='records'))
    
except Exception as e:
    print(f"Error: {e}")
