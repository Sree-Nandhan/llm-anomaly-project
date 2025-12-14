import pandas as pd
import numpy as np

# Generate a simple transaction dataset
np.random.seed(42)

N = 2000

data = {
    "user_id": np.random.randint(1000, 2000, N),
    "amount": np.random.normal(80, 20, N).round(2),  # mostly normal payments
    "location": np.random.choice(["StoreA", "StoreB", "StoreC"], N),
    "time_of_day": np.random.choice(["morning", "afternoon", "evening"], N)
}

df = pd.DataFrame(data)

# Inject anomalies
# Extremely large amounts
df.loc[np.random.choice(N, 50, replace=False), "amount"] *= np.random.randint(20, 80)

df.to_csv("data/transactions.csv", index=False)

print("Saved data/transactions.csv")
