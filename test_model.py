from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "microsoft/phi-2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Move model to Apple GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
model.to(device)

prompt = "Explain anomaly detection in simple words."

inputs = tokenizer(prompt, return_tensors="pt").to(device)

output = model.generate(**inputs, max_new_tokens=40)
print("\nGenerated:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
