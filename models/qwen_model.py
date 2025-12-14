import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class QwenModel:
    def __init__(self, model_id="Qwen/Qwen1.5-1.8B-Chat", device="mps"):
        self.device = device if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
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
