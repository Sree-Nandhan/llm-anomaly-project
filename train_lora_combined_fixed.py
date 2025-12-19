# train_lora_combined_fixed.py
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# --- 1) Config
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "lora-output-combined-fixed"
LORA_OUT = "lora-checkpoint-combined-fixed"
COMBINED_JSONL = "data/train_combined.jsonl"

# --- 2) Load dataset
dataset = load_dataset("json", data_files=COMBINED_JSONL)["train"]
print("Loaded dataset length:", len(dataset))

# --- 3) Tokenizer & base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16  # base weights in float16 ok; training will use bf16 on MPS
)

# --- 4) Create new LoRA adapter (fresh training)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, lora_config)

# --- 5) Ensure LoRA params require grad (explicit)
model.train()
for name, p in model.named_parameters():
    if "lora" in name.lower():
        p.requires_grad = True
    else:
        p.requires_grad = False

# print trainable summary
total, trainable = 0, 0
for n, p in model.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()
print(f"Trainable params: {trainable:,} / {total:,}")

# --- 6) Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print("Using device:", device)

# --- 7) Prompt formatting with unique marker and robust masking
def format_prompt(example):
    # use a stable unique output marker
    prompt = (
        "You are an AI financial analyst.\n"
        "Classify the transaction using realistic anomaly rules.\n"
        "Respond ONLY in this exact format:\n"
        "<LABEL> || <EXPLANATION>\n\n"
        f"{example['input']}\n\n"
        "### OUTPUT:\n"
        f"{example['output']}"
    )

    tokenized = tokenizer(prompt, truncation=True, max_length=384, padding="max_length")
    labels = tokenized["input_ids"].copy()

    output_marker = "### OUTPUT:"
    marker_index = prompt.find(output_marker)
    if marker_index == -1:
        # fallback: find last newline before the output text
        marker_ids = tokenizer(prompt)["input_ids"]  # fallback to whole prompt (worst case)
        mask_length = len(marker_ids) - len(tokenizer(example['output'])["input_ids"])
    else:
        marker_ids = tokenizer(prompt[: marker_index + len(output_marker)])["input_ids"]
        mask_length = len(marker_ids)

    # Clip mask_length to sequence length
    mask_length = min(mask_length, len(labels))
    labels[:mask_length] = [-100] * mask_length
    tokenized["labels"] = labels
    return tokenized

# map dataset (with disable = False so we can inspect)
dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

# --- 8) Sanity check: make sure labels include some non -100 tokens
import numpy as np
sample_labels = [ex["labels"] for ex in dataset.select(range(min(20, len(dataset))))]
non_masked_counts = [int(np.sum(np.array(l) != -100)) for l in sample_labels]
print("Non-masked token counts for first examples:", non_masked_counts)
if all(c == 0 for c in non_masked_counts):
    raise SystemExit("Error: All labels masked for initial examples - check masking logic.")

# --- 9) Training args (safer for MPS)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=200,
    bf16=True,    # use bf16 on supported Apple Silicon MPS
    fp16=False,   # disable fp16 if on MPS
    optim="adamw_torch",
    report_to=[],
)

# small callback to print a bit more info at each logging step
from transformers import TrainerCallback, TrainingArguments

class DebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print("LOG:", logs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[DebugCallback()]
)

# --- 10) quick smoke backward test (single batch) to ensure gradients compute
def smoke_test():
    batch = trainer.get_train_dataloader().__iter__().__next__()
    batch = {k: v.to(device) for k, v in batch.items()}
    model.train()
    out = model(**batch)
    loss = out.loss if hasattr(out, "loss") and out.loss is not None else None
    print("Smoke loss attribute:", loss)
    if loss is None:
        raise SystemExit("No loss returned by model on a single batch. Check labels and model.")
    loss.backward()
    grad_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None and p.requires_grad:
            grad_norm += (p.grad.detach().float().norm().item())**2
    import math
    grad_norm = math.sqrt(grad_norm)
    print("Smoke backward grad_norm:", grad_norm)
    model.zero_grad()

# run smoke test
try:
    smoke_test()
except SystemExit as e:
    print("Smoke test failed:", e)
    raise

# --- 11) Start training
trainer.train()

# --- 12) Save final checkpoint
model.save_pretrained(LORA_OUT)
print("Training complete! Saved as", LORA_OUT)
