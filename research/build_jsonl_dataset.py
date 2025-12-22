import pandas as pd
import json

df = pd.read_csv("data/transactions.csv")

mean_amount = df["amount"].mean()
std_amount = df["amount"].std()

outputs = []

for _, row in df.iterrows():
    
    # Classification
    if row["amount"] > mean_amount + 4 * std_amount:
        label = "ANOMALY"
        explanation = f"The amount {row['amount']} is extremely higher than typical transactions (avg {mean_amount:.2f})."
    else:
        label = "NORMAL"
        explanation = "This transaction amount is within the expected normal range."

    # Build LLM input prompt fields
    input_text = (
        f"Table: user_id, amount, location, time_of_day\n"
        f"Row: {row['user_id']}, {row['amount']}, {row['location']}, {row['time_of_day']}\n"
        f"Context: Analyze whether this transaction is suspicious."
    )

    output_text = f"{label} || {explanation}"

    outputs.append({"input": input_text, "output": output_text})

# Save JSONL
with open("data/train.jsonl", "w") as f:
        for o in outputs:
            f.write(json.dumps(o) + "\n")

print("Saved data/train.jsonl")
