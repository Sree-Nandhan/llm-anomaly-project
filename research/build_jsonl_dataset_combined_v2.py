import pandas as pd
import json

realistic = pd.read_csv("data/transactions_realistic.csv")
midrange_patch = pd.read_csv("data/midrange_normal_patch.csv")
extra_anomaly = pd.DataFrame({
    "user_id": [1001, 1002, 1003],
    "amount": [5000, 7000, 8000],
    "location": ["StoreA", "StoreB", "StoreC"],
    "label": ["ANOMALY", "ANOMALY", "ANOMALY"],
    "explanation": [
        "The amount is extremely high and indicates abnormal spending.",
        "The amount is extremely high and indicates abnormal spending.",
        "The amount is extremely high and indicates abnormal spending."
    ]
})

combined = pd.concat([realistic, midrange_patch, extra_anomaly], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

with open("data/train_combined_v2.jsonl", "w") as f:
    for _, row in combined.iterrows():
        input_text = (
            f"Table: user_id, amount, location\n"
            f"Row: {row['user_id']}, {row['amount']}, {row['location']}"
        )
        output_text = f"{row['label']} || {row['explanation']}"
        json.dump({"input": input_text, "output": output_text}, f)
        f.write("\n")

print("Created train_combined_v2.jsonl with total rows:", len(combined))
