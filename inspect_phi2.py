from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

print("\n=== MODEL MODULE NAMES ===\n")
for name, module in model.named_modules():
    print(name)
