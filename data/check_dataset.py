import pandas as pd

path = "data/dataset.xlsx"  # we will copy file here later
df = pd.read_excel(path)

print("Columns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())
