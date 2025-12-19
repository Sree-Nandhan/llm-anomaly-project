import json

OUTPUT_JSONL = "data/train_polish.jsonl"

normal_examples = []
low_anomaly_examples = []

# NORMAL examples (amounts 50–200)
for amount in [50, 75, 90, 120, 150, 180, 200]:
    normal_examples.append({
        "input": f"Table: user_id, amount, location\nRow: 1111, {amount}, StoreA",
        "output": f"NORMAL || The amount ${amount} is typical for normal spending at this store."
    })

# EXTREMELY LOW anomaly examples (< 5)
for amount in [0.5, 1.2, 2.5, 3.0, 4.5]:
    low_anomaly_examples.append({
        "input": f"Table: user_id, amount, location\nRow: 2222, {amount}, StoreB",
        "output": f"ANOMALY || The amount ${amount} is extremely small and may indicate refund fraud or abnormal activity."
    })

data = normal_examples + low_anomaly_examples

# Save JSONL
with open(OUTPUT_JSONL, "w") as f:
    for row in data:
        json.dump(row, f)
        f.write("\n")

print(f"Polish dataset saved → {OUTPUT_JSONL}")
print(f"Total examples: {len(data)}")
