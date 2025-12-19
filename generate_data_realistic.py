import numpy as np
import pandas as pd
import random

def generate_realistic_transactions(n=6000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # -----------------------------
    # NORMAL TRANSACTIONS
    # -----------------------------
    normal_amounts = np.random.normal(loc=200, scale=80, size=int(n * 0.75))
    normal_amounts = np.abs(normal_amounts)

    normal_df = pd.DataFrame({
        "user_id": np.random.randint(1000, 9999, size=len(normal_amounts)),
        "amount": np.round(normal_amounts, 2),
        "location": np.random.choice(["StoreA", "StoreB", "StoreC"], size=len(normal_amounts)),
    })
    normal_df["label"] = "NORMAL"
    normal_df["explanation"] = normal_df["amount"].apply(
        lambda x: f"The amount ${x} is typical for general spending behavior."
    )

    # -----------------------------
    # ANOMALY TYPE 1 — HIGH AMOUNTS
    # -----------------------------
    high_amounts = np.random.uniform(2000, 8000, size=int(n * 0.15))
    high_df = pd.DataFrame({
        "user_id": np.random.randint(1000, 9999, size=len(high_amounts)),
        "amount": np.round(high_amounts, 2),
        "location": np.random.choice(["StoreA", "StoreB", "StoreC"], size=len(high_amounts)),
    })
    high_df["label"] = "ANOMALY"
    high_df["explanation"] = high_df["amount"].apply(
        lambda x: f"The amount ${x} is unusually high compared to typical spending."
    )

    # -----------------------------
    # ANOMALY TYPE 2 — VERY LOW AMOUNTS (refund fraud)
    # -----------------------------
    low_amounts = np.random.uniform(0.5, 5.0, size=int(n * 0.05))
    low_df = pd.DataFrame({
        "user_id": np.random.randint(1000, 9999, size=len(low_amounts)),
        "amount": np.round(low_amounts, 2),
        "location": np.random.choice(["StoreA", "StoreB", "StoreC"], size=len(low_amounts)),
    })
    low_df["label"] = "ANOMALY"
    low_df["explanation"] = low_df["amount"].apply(
        lambda x: f"The amount ${x} is extremely small and may indicate refund fraud or abnormal activity."
    )

    # -----------------------------
    # ANOMALY TYPE 3 — LOCATION-BASED ANOMALIES
    # (rare store or unusual location)
    # -----------------------------
    location_df = pd.DataFrame({
        "user_id": np.random.randint(1000, 9999, size=int(n * 0.05)),
        "amount": np.random.normal(200, 80, size=int(n * 0.05)),
        "location": np.random.choice(["RareStoreX", "OutlierMallY"], size=int(n * 0.05)),
    })
    location_df["amount"] = np.round(np.abs(location_df["amount"]), 2)
    location_df["label"] = "ANOMALY"
    location_df["explanation"] = location_df.apply(
        lambda row: f"The location '{row['location']}' is unusual compared to common transaction locations.",
        axis=1
    )

    # -----------------------------
    # COMBINE ALL PARTS
    # -----------------------------
    df = pd.concat([normal_df, high_df, low_df, location_df], ignore_index=True)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Final cleanup
    df["amount"] = df["amount"].round(2)

    return df


if __name__ == "__main__":
    df = generate_realistic_transactions()
    df.to_csv("data/transactions_realistic.csv", index=False)
    print("Realistic dataset saved to data/transactions_realistic.csv")
    print(df.head())
