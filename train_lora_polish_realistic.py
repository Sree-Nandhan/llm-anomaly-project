import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

# -----------------------------
# 1. Load polish dataset
# -----------------------------
dataset = load_dataset("json", data_files="data/train_polish.jsonl")["train"]

# -----------------------------
# 2. Load base + previous realistic LoRA
# -----------------------------
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
lora_path = "lora-checkpoint-realistic"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, lora_path)

# Enable training on LoRA layers
model.train()
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# -----------------------------
# 3. Prompt Formatting
# -----------------------------
def format_prompt(example):
    prompt = (
        "You are an AI financial analyst.\n"
        "Fix classification mistakes using the following rule:\n"
        "- Amounts between $50–$200 are NORMAL.\n"
        "- Amounts below $10 are ANOMALY.\n\n"
        "Respond ONLY in this format:\n"
        "<LABEL> || <EXPLANATION>\n\n"
        f"{example['input']}\n"
        "Answer: "
        f"{example['output']}"
    )

    tokenized = tokenizer(prompt, truncation=True, max_length=384, padding="max_length")

    # Create labels and apply mask
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
# 4. Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="lora-output-polish-realistic",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=1,
    fp16=True,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("lora-checkpoint-realistic-final")
print("Polish LoRA training complete!")
