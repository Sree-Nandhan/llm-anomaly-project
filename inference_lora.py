import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# === IMPORTANT ===
# Must match the SAME base model used during fine-tuning
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
lora_path = "lora-checkpoint"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16
)

# Load LoRA fine-tuned weights
model = PeftModel.from_pretrained(model, lora_path)

# Use MPS GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

# ---------------------------
# Test example
# ---------------------------
test_input = """
You are an AI financial analyst.
Table: user_id, amount, location
Row: 1500, 4500, StoreA
Context: Check anomaly.
Answer:
"""

inputs = tokenizer(test_input, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False
    )

print("\nModel Output:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
