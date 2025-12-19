import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# Paths to your new model
# -----------------------------
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
lora_path = "/Users/sreenandhan/Desktop/llm-anomaly-project/lora-checkpoint-zscore"

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Load base model + LoRA adapter
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(model, lora_path)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

# -----------------------------
# TEST INPUT (You can modify)
# -----------------------------
test_input = """
You are an AI financial analyst.
Use the z-score anomaly rule:
- If z > 3 → ANOMALY
- Else → NORMAL

Format: <LABEL> || <EXPLANATION>

Table: user_id, amount, location
Row: 1500, 4500, StoreA

Answer:
"""

inputs = tokenizer(test_input, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False
    )

print("\n====================")
print("MODEL OUTPUT:")
print("====================\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
