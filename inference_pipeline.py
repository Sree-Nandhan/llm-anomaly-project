import numpy as np
import pandas as pd
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# LOAD CLASSIFIER + METADATA
# ============================================================
clf = joblib.load("classifier_out/rf_pipeline.joblib")
meta = joblib.load("classifier_out/meta.joblib")

amount_mean = meta["amount_mean"]
amount_std = meta["amount_std"]
num_features = meta["num_features"]
cat_features = meta["cat_features"]

# ============================================================
# LOAD LLM (FOR EXPLANATION ONLY)
# ============================================================
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LORA_PATH = "lora-checkpoint-final-clean"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.to(DEVICE)
model.eval()

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def compute_features(amount, location):
    row = {
        "amount": amount,
        "log_amount": np.log1p(amount),
        "z_amount": (amount - amount_mean) / (amount_std if amount_std != 0 else 1.0),
        "is_small": int(amount < 5),
        "is_large": int(amount > 2000),
        "location": location
    }
    X_df = pd.DataFrame([row], columns=num_features + cat_features)
    return X_df, row

# ============================================================
# LLM EXPLANATION (NOW CONDITIONED ON CLASSIFIER LABEL)
# ============================================================
def get_llm_explanation(user_id, amount, location, final_label):
    prompt = f"""
You are an AI financial analyst.

The classifier has predicted: {final_label}

Explain THIS decision in 1–2 sentences.
Do NOT contradict the classifier.
Base your explanation on the amount and location only.

Transaction:
user_id={user_id}, amount={amount}, location={location}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in decoded:
        decoded = decoded.split("Answer:", 1)[1].strip()

    return decoded.split("\n")[0]

# ============================================================
# FINAL PREDICTION PIPELINE
# ============================================================
def predict(user_id, amount, location):
    X_df, feat = compute_features(amount, location)

    label = clf.predict(X_df)[0]
    prob = clf.predict_proba(X_df)[0].tolist()
    explanation = get_llm_explanation(user_id, amount, location, label)

    return {
        "label": label,
        "probabilities": prob,
        "explanation": explanation,
        "features": feat
    }

# ============================================================
# TEST CASES
# ============================================================
tests = [
    (1234, 75, "StoreA"),
    (5678, 4500, "StoreB"),
    (3456, 2.5, "StoreC"),
    (9999, 120, "RareStoreX"),
]

print("\n=== FINAL HYBRID SYSTEM TEST ===\n")
for uid, amt, loc in tests:
    out = predict(uid, amt, loc)
    print("------------------------------")
    print(f"INPUT → {amt} at {loc}")
    print("LABEL:", out["label"])
    print("PROBS:", out["probabilities"])
    print("EXPLANATION:", out["explanation"])
    print("------------------------------\n")
