import pandas as pd
import json

try:
    file_path = r"d:\Tradeidesa\Multibagger-Ultraversion\RelativeStrength (1).xlsx"
    xl = pd.ExcelFile(file_path)
    # Most interesting sheets are likely the periodic ones
    sheet = 'D-o-D'
    df = xl.parse(sheet)
    print(f"Sheet: {sheet}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 10 rows:")
    print(df.head(10).to_dict(orient='records'))
    
except Exception as e:
    print(f"Error: {e}")
