import numpy as np
import pandas as pd

def generate_transaction_data(n=5000, random_seed=42):
    np.random.seed(random_seed)

    # Generate synthetic amounts from a normal-like distribution
    base_amounts = np.random.normal(loc=200, scale=100, size=n)  # typical purchases
    base_amounts = np.abs(base_amounts)  # ensure positive

    # Inject large anomalies
    anomaly_indices = np.random.choice(n, size=int(n * 0.05), replace=False)
    base_amounts[anomaly_indices] *= np.random.uniform(5, 15)  # inflate anomalies

    # Round amounts
    base_amounts = np.round(base_amounts, 2)

    # Create dataframe
    df = pd.DataFrame({
        "user_id": np.random.randint(1000, 9999, size=n),
        "amount": base_amounts,
        "location": np.random.choice(["StoreA", "StoreB", "StoreC"], size=n)
    })

    # Compute mean & std for z-score labeling
    mean = df["amount"].mean()
    std = df["amount"].std()

    def z_label(row):
        z = (row["amount"] - mean) / std
        if z > 3:
            return "ANOMALY", f"The amount ${row['amount']} is {z:.2f} standard deviations above the mean."
        else:
            return "NORMAL", f"The amount ${row['amount']} is within the normal spending range."

    labels = df.apply(lambda row: z_label(row), axis=1)
    df["label"] = [l[0] for l in labels]
    df["explanation"] = [l[1] for l in labels]

    return df, mean, std


if __name__ == "__main__":
    df, mean, std = generate_transaction_data()
    df.to_csv("data/transactions_zscore.csv", index=False)
    print("Saved dataset to data/transactions_zscore.csv")
    print(f"Mean: {mean:.2f}, STD: {std:.2f}")
