import pandas as pd
import json

df = pd.read_csv("data/final_clean_transactions.csv")

with open("data/train_final_clean.jsonl", "w") as f:
    for _, row in df.iterrows():
        input_text = (
            f"Table: user_id, amount, location\n"
            f"Row: {row['user_id']}, {row['amount']}, {row['location']}"
        )

        output_text = f"{row['label']} || {row['explanation']}"

        json.dump({"input": input_text, "output": output_text}, f)
        f.write("\n")

print("Saved → train_final_clean.jsonl")
