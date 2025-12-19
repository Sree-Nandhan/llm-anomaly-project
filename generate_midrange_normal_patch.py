import numpy as np
import pandas as pd

def generate_midrange_normal(n=1000, seed=42):
    np.random.seed(seed)

    amounts = np.random.uniform(50, 200, size=n)

    df = pd.DataFrame({
        "user_id": np.random.randint(1000, 9999, size=n),
        "amount": np.round(amounts, 2),
        "location": np.random.choice(["StoreA", "StoreB", "StoreC"], size=n),
    })

    df["label"] = "NORMAL"
    df["explanation"] = df["amount"].apply(
        lambda x: f"The amount ${x} is typical for normal spending behavior."
    )

    df.to_csv("data/midrange_normal_patch.csv", index=False)
    print("Generated mid-range NORMAL patch dataset:", len(df))

if __name__ == "__main__":
    generate_midrange_normal()
