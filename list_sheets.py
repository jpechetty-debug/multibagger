import pandas as pd
import json

try:
    file_path = r"d:\Tradeidesa\Multibagger-Ultraversion\RelativeStrength (1).xlsx"
    xl = pd.ExcelFile(file_path)
    sheets = xl.sheet_names
    print(f"---SHEETS_START---")
    print(json.dumps(sheets))
    print(f"---SHEETS_END---")
    
    # Also print the first sheet name's content to be sure
    df = xl.parse(sheets[0])
    print(f"\nSheet {sheets[0]} columns: {df.columns.tolist()}")
    
except Exception as e:
    print(f"Error: {e}")
