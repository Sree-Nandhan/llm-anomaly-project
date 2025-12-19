import numpy as np
import pandas as pd
import json


# -----------------------------
# 1. Load your existing realistic dataset
# -----------------------------
realistic_df = pd.read_csv("data/transactions_realistic.csv")


# -----------------------------
# 2. Create correction dataset for NORMAL mid-range values ($50–$200)
# -----------------------------
normal_correction_amounts = np.random.uniform(50, 200, size=300)

normal_correction_df = pd.DataFrame({
    "user_id": np.random.randint(1000, 9999, size=300),
    "amount": np.round(normal_correction_amounts, 2),
    "location": np.random.choice(["StoreA", "StoreB", "StoreC"], size=300),
})

normal_correction_df["label"] = "NORMAL"
normal_correction_df["explanation"] = normal_correction_df["amount"].apply(
    lambda x: f"The amount ${x} is typical for normal spending and does not indicate suspicious behavior."
)


# -----------------------------
# 3. Add a few extra anomaly reinforcement examples (optional but improves model)
# -----------------------------
extra_anomaly_amounts = [2500, 3000, 4500, 6000, 7500]

extra_anomaly_df = pd.DataFrame({
    "user_id": np.random.randint(1000, 9999, size=len(extra_anomaly_amounts)),
    "amount": extra_anomaly_amounts,
    "location": ["StoreA", "StoreB", "StoreC", "StoreA", "StoreB"]
})

extra_anomaly_df["label"] = "ANOMALY"
extra_anomaly_df["explanation"] = extra_anomaly_df["amount"].apply(
    lambda x: f"The amount ${x} is extremely high and indicates abnormal spending behavior."
)


# -----------------------------
# 4. MERGE EVERYTHING
# -----------------------------
combined_df = pd.concat([
    realistic_df,
    normal_correction_df,
    extra_anomaly_df
], ignore_index=True)

# Shuffle dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save merged CSV
combined_df.to_csv("data/transactions_combined.csv", index=False)

print("Combined dataset created:")
print(combined_df.head())
print(f"Total rows: {len(combined_df)}")
