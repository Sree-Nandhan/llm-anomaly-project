import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# === Correct model paths ===
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
lora_path = "lora-checkpoint-polished"      # <-- polished checkpoint

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16
)

# Load polished LoRA weights
model = PeftModel.from_pretrained(model, lora_path)

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

# ---------------------------
# Polished inference prompt
# ---------------------------
test_input = """
You are an AI financial analyst.
Analyze the following transaction and classify it as ANOMALY or NORMAL.
Then provide a short explanation.

Format your response EXACTLY as:
<LABEL> || <EXPLANATION>

Table: user_id, amount, location
Row: 1500, 4500, StoreA
Context: Check anomaly.

Answer:
"""

inputs = tokenizer(test_input, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

print("\nModel Output:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
