import pandas as pd
import json

INPUT_CSV = "data/transactions_realistic.csv"
OUTPUT_JSONL = "data/train_realistic.jsonl"

df = pd.read_csv(INPUT_CSV)

with open(OUTPUT_JSONL, "w") as f:
    for _, row in df.iterrows():
        input_text = (
            f"Table: user_id, amount, location\n"
            f"Row: {row['user_id']}, {row['amount']}, {row['location']}"
        )

        output_text = f"{row['label']} || {row['explanation']}"

        json.dump({"input": input_text, "output": output_text}, f)
        f.write("\n")

print(f"Saved JSONL → {OUTPUT_JSONL}")
