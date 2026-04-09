import pandas as pd

df = pd.read_csv(
    r"E:\Ava\New folder\annotation\ava_val_filtered.csv",
    header=None
)

ids = sorted(df[6].unique())
print(f"Unique action IDs: {len(ids)}")
print(f"Min: {min(ids)}, Max: {max(ids)}")

missing = set(range(1, 81)) - set(ids)
if missing:
    print(f"Missing IDs in val: {sorted(missing)}")
else:
    print("All 80 action IDs present in val. Clean.")