import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# -----------------------------
# 1. Load combined dataset
# -----------------------------
dataset = load_dataset("json", data_files="data/train_combined.jsonl")["train"]

# -----------------------------
# 2. Load base model
# -----------------------------
model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# -----------------------------
# 3. Create NEW LoRA adapter (fresh retrain)
# -----------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, lora_config)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# -----------------------------
# 4. Format prompt
# -----------------------------
def format_prompt(example):
    prompt = (
        "You are an AI financial analyst.\n"
        "Classify the transaction using realistic anomaly rules.\n"
        "Respond ONLY in this format:\n"
        "<LABEL> || <EXPLANATION>\n\n"
        f"{example['input']}\n"
        "Answer: "
        f"{example['output']}"
    )

    tokenized = tokenizer(prompt, truncation=True, max_length=384, padding="max_length")
    labels = tokenized["input_ids"].copy()

    answer_start = prompt.find("Answer:")
    if answer_start == -1:
        answer_start = len(prompt)

    prompt_ids = tokenizer(prompt[:answer_start])["input_ids"]
    mask_length = len(prompt_ids)
    labels[:mask_length] = [-100] * mask_length

    tokenized["labels"] = labels
    return tokenized

dataset = dataset.map(format_prompt)

# -----------------------------
# 5. Training arguments (1 epoch)
# -----------------------------
args = TrainingArguments(
    output_dir="lora-output-combined",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("lora-checkpoint-combined")
print("Training complete! Saved as lora-checkpoint-combined")
