import pandas as pd
import numpy as np

# Simulate the data loading scenario
data = {
    "gemrate_data.universal_gemrate_id": [np.nan, "univ_123", np.nan],
    "gemrate_data.gemrate_id": ["orig_1", "orig_2", "orig_3"]
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("-" * 20)

# Simulate the problematic code snippet from main.py
print("Running problematic code snippet...")
# Line 401
df["universal_gemrate_id"] = df["gemrate_data.universal_gemrate_id"].astype(str)
print("After astype(str):")
print(df["universal_gemrate_id"].tolist())

# Line 402
df["gemrate_id"] = df["gemrate_data.gemrate_id"]

# Line 403
# Issue hypothesis: 'nan' string is considered notna(), so it overwrites valid gemrate_id
mask = df['universal_gemrate_id'].notna()
print(f"Mask (notna): {mask.tolist()}")

df.loc[df['universal_gemrate_id'].notna(), "gemrate_id"] = df.loc[df['universal_gemrate_id'].notna(), "universal_gemrate_id"]

# Line 404
df = df.dropna(subset=["gemrate_id"])

print("-" * 20)
print("Final DataFrame:")
print(df)

# Check if we lost orig_1 and orig_3 and replaced them with "nan"
print("-" * 20)
print("gemrate_id values:")
print(df["gemrate_id"].tolist())
