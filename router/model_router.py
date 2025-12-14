from models.llama_lora import LlamaLoraModel
from models.qwen_model import QwenModel
from utils.normalize_output import normalize_output

class ModelRouter:
    def __init__(self):
        self.llama = LlamaLoraModel(
            "meta-llama/Llama-3.2-1B-Instruct",
            "lora-checkpoint-polished"
        )
        self.qwen = QwenModel()

    def choose_model(self, strategy="llama"):
        if strategy == "llama":
            return self.llama
        elif strategy == "qwen":
            return self.qwen
        elif strategy == "ensemble":
            return "ensemble"
        else:
            return self.llama  # default

    def predict(self, prompt, strategy="llama"):
        model = self.choose_model(strategy)

        if strategy in ["llama", "qwen"]:
            raw = model.predict(prompt)
            return normalize_output(raw)


        elif strategy == "ensemble":
            raw_llama = normalize_output(self.llama.predict(prompt))
            raw_qwen = normalize_output(self.qwen.predict(prompt))

            if "ANOMALY" in raw_llama or "ANOMALY" in raw_qwen:
                label = "ANOMALY"
            else:
                label = "NORMAL"
            explanation = f"LLaMA: {raw_llama} | Qwen: {raw_qwen}"
            return f"{label} || {explanation}"
        else:
            return self.llama.predict(prompt)
