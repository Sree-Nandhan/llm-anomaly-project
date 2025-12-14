from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("json", data_files="data/train.jsonl")["train"]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Show first sample
sample = dataset[0]

print("\n=== SAMPLE INPUT ===")
print(sample["input"])

print("\n=== SAMPLE OUTPUT ===")
print(sample["output"])

# Tokenize
encoded = tokenizer(sample["input"] + " " + sample["output"],
                    truncation=True,
                    max_length=512)

print("\nFirst 30 token IDs:")
print(encoded["input_ids"][:30])

print("\nTotal token length:", len(encoded["input_ids"]))
