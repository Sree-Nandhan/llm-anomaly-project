import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

dataset = load_dataset("json", data_files="data/train_combined_v2.jsonl")["train"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)

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

def format_prompt(example):
    prompt = (
        "You are an AI financial analyst.\n"
        "Classify the transaction as NORMAL or ANOMALY.\n"
        "Respond in this exact format:\n"
        "<LABEL> || <EXPLANATION>\n\n"
        f"{example['input']}\n\n"
        "### OUTPUT:\n"
        f"{example['output']}"
    )

    tokenized = tokenizer(prompt, truncation=True, max_length=384, padding="max_length")
    labels = tokenized["input_ids"].copy()

    marker = "### OUTPUT:"
    marker_idx = prompt.find(marker)
    marker_ids = tokenizer(prompt[: marker_idx + len(marker)])["input_ids"]
    mask_len = len(marker_ids)

    labels[:mask_len] = [-100] * mask_len
    tokenized["labels"] = labels
    return tokenized

dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="lora-output-combined-final",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-4,
    bf16=True,
    logging_steps=20,
    save_steps=200,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained("lora-checkpoint-combined-final")
print("Final model saved → lora-checkpoint-combined-final")
