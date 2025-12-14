# evaluate_lora.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------- CONFIG --------
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"   # MUST match what you used for training
LORA_PATH = "lora-checkpoint"
DATA_FILE = "data/train.jsonl"                   # your dataset
MAX_SAMPLES = 30                                # reduce to speed up; change if desired
GEN_MAX_NEW_TOKENS = 60
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# ------------------------

print("Using device:", DEVICE)

ds = load_dataset("json", data_files=DATA_FILE)["train"]
n = min(MAX_SAMPLES, len(ds))
eval_ds = ds.select(range(n))
print(f"Loaded {len(eval_ds)} examples for eval")

print("Loading tokenizer and base model (this may take a minute)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.to(DEVICE)
model.eval()

def gen_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)

gts, preds, rows = [], [], []
for i, ex in enumerate(eval_ds):
    prompt = "You are an AI financial analyst.\n" + ex["input"] + "\nAnswer:\n"
    generated = gen_text(prompt)

    up = generated.upper()
    if "ANOMALY" in up:
        pred = "ANOMALY"
    elif "NORMAL" in up:
        pred = "NORMAL"
    elif "ANOMAL" in up:
        pred = "ANOMALY"
    else:
        # fallback heuristics
        pred = "ANOMALY" if any(w in up for w in ["UNUSUAL","HIGH","SUSPICIOUS","FRAUD"]) else "NORMAL"

    gt_text = ex.get("output","").upper()
    if "ANOMALY" in gt_text:
        gt = "ANOMALY"
    elif "NORMAL" in gt_text:
        gt = "NORMAL"
    elif "ANOMAL" in gt_text:
        gt = "ANOMALY"
    else:
        gt = "NORMAL"

    gts.append(gt)
    preds.append(pred)
    rows.append({"index": i, "input": ex["input"], "gt": gt, "pred": pred, "gen": generated})

# Metrics
acc = accuracy_score(gts, preds)
prec = precision_score(gts, preds, pos_label="ANOMALY", zero_division=0)
rec = recall_score(gts, preds, pos_label="ANOMALY", zero_division=0)
f1 = f1_score(gts, preds, pos_label="ANOMALY", zero_division=0)
cm = confusion_matrix(gts, preds, labels=["ANOMALY","NORMAL"])

print("\n=== EVALUATION METRICS ===")
print(f"Samples evaluated: {len(gts)}")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (ANOMALY): {prec:.4f}")
print(f"Recall (ANOMALY): {rec:.4f}")
print(f"F1 (ANOMALY): {f1:.4f}")
print("Confusion matrix (rows=GT [ANOMALY,NORMAL], cols=Pred [ANOMALY,NORMAL]):")
print(cm.tolist())

# Write sample outputs to CSV for quick inspection
import csv
with open("eval_samples_short.csv","w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["index","input","gt","pred","gen"])
    writer.writeheader()
    for r in rows[:100]:
        writer.writerow(r)

print("\nWrote first 100 generated examples to eval_samples_short.csv for inspection.")
print("Done.")
