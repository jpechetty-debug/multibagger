import pandas as pd
import os

try:
    file_path = r"d:\Tradeidesa\Multibagger-Ultraversion\RelativeStrength (1).xlsx"
    df = pd.read_excel(file_path)
    
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 20 rows:")
    print(df.head(20).to_string())
    
except Exception as e:
    print(f"Error: {e}")
