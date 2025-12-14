import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class LlamaLoraModel:
    def __init__(self, base_model, lora_path, device="mps"):
        self.device = device if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
        self.model = PeftModel.from_pretrained(base, lora_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, max_new_tokens=120):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
