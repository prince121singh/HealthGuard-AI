import pandas as pd

RAW_FILE = "data/heart_raw.csv"
OUTPUT_FILE = "data/heart.csv"

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

# Load raw UCI dataset
df = pd.read_csv(
    RAW_FILE,
    header=None,
    names=columns,
    na_values="?"
)

# Convert columns to numeric
for column in columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows containing missing values
df = df.dropna().reset_index(drop=True)

# UCI target:
# 0 = no heart disease
# 1-4 = presence of heart disease
df["target"] = (df["target"] > 0).astype(int)

# Save clean dataset
df.to_csv(OUTPUT_FILE, index=False)

print("===================================")
print("HEART DATASET CREATED SUCCESSFULLY")
print("===================================")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["target"].value_counts())

print("\nFirst 5 rows:")
print(df.head())

print(f"\nSaved to: {OUTPUT_FILE}")