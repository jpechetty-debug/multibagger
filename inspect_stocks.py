import pandas as pd
import json

try:
    file_path = r"d:\Tradeidesa\Multibagger-Ultraversion\RelativeStrength (1).xlsx"
    xl = pd.ExcelFile(file_path)
    df = xl.parse("Stocks")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 10 rows:")
    print(df.head(10).to_dict(orient='records'))
    
except Exception as e:
    print(f"Error: {e}")
