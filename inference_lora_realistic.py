import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# -----------------------------
# LABEL NORMALIZATION FUNCTION
# -----------------------------
def normalize_label(label_raw: str) -> str:
    """
    Converts ANY generated label into either NORMAL or ANOMALY.
    """
    text = label_raw.lower().strip()

    anomaly_words = [
        "anomal", "anom", "extra", "extreme", "extremely",
        "rare", "unusual", "suspicious", "extraordinary", "very low"
    ]
    for w in anomaly_words:
        if text.startswith(w) or w in text:
            return "ANOMALY"

    # normal-like words
    normal_words = ["normal", "typical", "regular", "common"]
    for w in normal_words:
        if text.startswith(w) or w in text:
            return "NORMAL"

    return "ANOMALY"


# -----------------------------
# MODEL LOADING
# -----------------------------
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LORA_PATH = "lora-checkpoint-final-clean"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading tokenizer and base model ({BASE_MODEL})...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16
)

print("Loading LoRA adapter:", LORA_PATH)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.to(DEVICE)
model.eval()

print("Model loaded successfully on", DEVICE)


# -----------------------------
# INFERENCE FUNCTION
# -----------------------------
def test_transaction(user_id, amount, location, max_new_tokens=80):
    prompt = f"""
You are an AI financial analyst.
Classify the transaction as NORMAL or ANOMALY.
Respond ONLY in the format:
<LABEL> || <EXPLANATION>

Table: user_id, amount, location
Row: {user_id}, {amount}, {location}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract model output after "Answer:"
    if "Answer:" in decoded:
        decoded = decoded.split("Answer:", 1)[1].strip()

    # Split into label/explanation
    if "||" in decoded:
        raw_label, explanation = decoded.split("||", 1)
    else:
        return "ANOMALY || Model returned unstructured output."

    # Normalize the label
    final_label = normalize_label(raw_label)

    return f"{final_label} || {explanation.strip()}"


# -----------------------------
# RUN FINAL TEST CASES
# -----------------------------
tests = [
    (1234, 75, "StoreA"),       # EXPECT: NORMAL
    (5678, 4500, "StoreB"),     # EXPECT: ANOMALY
    (3456, 2.5, "StoreC"),      # EXPECT: ANOMALY
    (9999, 120, "RareStoreX"),  # EXPECT: ANOMALY
]

print("\n=== FINAL MODEL TEST ===\n")
for uid, amt, loc in tests:
    print("------------------------------")
    print(f"INPUT → {amt} at {loc}")
    print("OUTPUT:")
    print(test_transaction(uid, amt, loc))
    print("------------------------------\n")

print("Inference complete.")
