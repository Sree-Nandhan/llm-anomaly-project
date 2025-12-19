import numpy as np
import pandas as pd

def generate_final_clean_dataset(n=6000, seed=42):
    np.random.seed(seed)

    # -----------------------------------------------
    # NORMAL TRANSACTIONS (majority)
    # -----------------------------------------------
    normal_amounts = np.random.normal(loc=200, scale=60, size=int(n * 0.65))
    normal_amounts = np.clip(normal_amounts, 20, 400)

    df_normal = pd.DataFrame({
        "user_id": np.random.randint(1000, 9999, size=len(normal_amounts)),
        "amount": np.round(normal_amounts, 2),
        "location": np.random.choice(["StoreA", "StoreB", "StoreC"], size=len(normal_amounts)),
        "label": "NORMAL",
    })
    df_normal["explanation"] = df_normal["amount"].apply(
        lambda x: f"This transaction amount (${x}) is typical for normal spending behavior."
    )

    # -----------------------------------------------
    # MID-RANGE NORMAL REINFORCEMENT (50–200)
    # -----------------------------------------------
    mid_amounts = np.random.uniform(50, 200, size=1000)
    df_mid = pd.DataFrame({
        "user_id": np.random.randint(1000, 9999, size=1000),
        "amount": np.round(mid_amounts, 2),
        "location": np.random.choice(["StoreA", "StoreB", "StoreC"], size=1000),
        "label": "NORMAL",
    })
    df_mid["explanation"] = df_mid["amount"].apply(
        lambda x: f"This transaction amount (${x}) falls within the typical range of spending."
    )

    # -----------------------------------------------
    # HIGH AMOUNT ANOMALIES (CORRECTED EXPLANATION)
    # -----------------------------------------------
    high_amounts = np.random.uniform(2500, 8000, size=int(n * 0.15))
    df_high = pd.DataFrame({
        "user_id": np.random.randint(1000, 9999, size=len(high_amounts)),
        "amount": np.round(high_amounts, 2),
        "location": np.random.choice(["StoreA", "StoreB", "StoreC"], size=len(high_amounts)),
        "label": "ANOMALY",
    })
    df_high["explanation"] = df_high["amount"].apply(
        lambda x: f"This transaction amount (${x}) is unusually high compared to normal spending levels."
    )

    # -----------------------------------------------
    # VERY LOW AMOUNT ANOMALIES (< $5)
    # -----------------------------------------------
    low_amounts = np.random.uniform(0.5, 5, size=int(n * 0.1))
    df_low = pd.DataFrame({
        "user_id": np.random.randint(1000, 9999, size=len(low_amounts)),
        "amount": np.round(low_amounts, 2),
        "location": np.random.choice(["StoreA", "StoreB", "StoreC"], size=len(low_amounts)),
        "label": "ANOMALY",
    })
    df_low["explanation"] = df_low["amount"].apply(
        lambda x: f"This transaction amount (${x}) is extremely low and may indicate refund fraud or abnormal behavior."
    )

    # -----------------------------------------------
    # RARE LOCATION ANOMALIES
    # -----------------------------------------------
    df_rare = pd.DataFrame({
        "user_id": np.random.randint(1000, 9999, size=int(n * 0.1)),
        "amount": np.round(np.random.normal(200, 60, size=int(n * 0.1)), 2),
        "location": np.random.choice(["RareStoreX", "OutlierMallY"], size=int(n * 0.1)),
        "label": "ANOMALY",
    })
    df_rare["explanation"] = df_rare.apply(
        lambda row: f"This transaction occurred at an unusual location ('{row['location']}'), which is not typical for normal spending.",
        axis=1
    )

    # -----------------------------------------------
    # COMBINE & SHUFFLE
    # -----------------------------------------------
    df = pd.concat([df_normal, df_mid, df_high, df_low, df_rare], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df.to_csv("data/final_clean_transactions.csv", index=False)
    print("Saved → data/final_clean_transactions.csv, rows:", len(df))


if __name__ == "__main__":
    generate_final_clean_dataset()
