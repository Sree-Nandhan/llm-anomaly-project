import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# CONFIG
# ============================================================
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH = "data/train_final_clean.jsonl"      # FINAL CLEAN DATASET
OUTPUT_DIR = "lora-output-final-clean"
LORA_OUT = "lora-checkpoint-final-clean"

print("Using dataset:", DATA_PATH)

# ============================================================
# 1. LOAD DATASET
# ============================================================
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
print("Loaded dataset size:", len(dataset))

# ============================================================
# 2. LOAD TOKENIZER + BASE MODEL
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16   # safest on MPS
)

# ============================================================
# 3. LoRA CONFIGURATION
# ============================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, lora_config)
model.train()

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

print("Training on device:", device)

# ============================================================
# 4. FORMAT PROMPTS + MASK THE INSTRUCTION PART
# ============================================================
def format_prompt(example):
    prompt = (
        "You are an AI financial analyst.\n"
        "Classify the transaction as NORMAL or ANOMALY.\n"
        "Respond ONLY in the format:\n"
        "<LABEL> || <EXPLANATION>\n\n"
        f"{example['input']}\n\n"
        "### OUTPUT:\n"
        f"{example['output']}"
    )

    tokenized = tokenizer(prompt, truncation=True, max_length=384, padding="max_length")

    labels = tokenized["input_ids"].copy()

    # Find marker to start computing loss
    marker = "### OUTPUT:"
    marker_idx = prompt.find(marker)

    if marker_idx == -1:
        raise ValueError("Marker ### OUTPUT: not found in prompt")

    marker_ids = tokenizer(prompt[: marker_idx + len(marker)])["input_ids"]
    mask_len = len(marker_ids)

    # Mask everything before the output
    labels[:mask_len] = [-100] * mask_len

    tokenized["labels"] = labels
    return tokenized

dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

# ============================================================
# 5. SANITY CHECK (VERY IMPORTANT)
# ============================================================
labels0 = dataset[0]["labels"]
non_masked = sum(1 for t in labels0 if t != -100)

print("Non-masked tokens sample:", non_masked)
if non_masked == 0:
    raise SystemExit("FATAL ERROR: All labels masked → No learning will happen.")

# ============================================================
# 6. TRAINING ARGUMENTS
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-4,
    logging_steps=20,
    save_steps=200,
    bf16=True,         # safe on MPS
    fp16=False,        # avoid fp16 NaNs
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# ============================================================
# 7. TRAIN
# ============================================================
trainer.train()

# ============================================================
# 8. SAVE FINAL LoRA WEIGHTS
# ============================================================
model.save_pretrained(LORA_OUT)
print("\n=====================================================")
print("FINAL CLEAN LORA MODEL SAVED →", LORA_OUT)
print("=====================================================\n")
