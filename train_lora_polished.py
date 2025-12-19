# import torch
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
# from peft import LoraConfig, get_peft_model, TaskType

# # -----------------------------
# # 1. Load dataset
# # -----------------------------
# dataset = load_dataset("json", data_files="data/train_zscore.jsonl")["train"]

# # -----------------------------
# # 2. Choose model
# # -----------------------------
# model_name = "meta-llama/Llama-3.2-1B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16
# )

# device = "mps" if torch.backends.mps.is_available() else "cpu"
# model.to(device)

# # -----------------------------
# # 3. LoRA Configuration
# # -----------------------------
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout=0.05,
#     task_type=TaskType.CAUSAL_LM,
# )

# model = get_peft_model(model, lora_config)

# # -----------------------------
# # 4. Tokenization + Prompt Formatting
# # -----------------------------
# def format_prompt(example):

#     prompt = (
#         "You are an AI financial analyst.\n"
#         "Use the z-score anomaly rule:\n"
#         "- Compute how many standard deviations the amount is above the mean.\n"
#         "- If z > 3, classify as ANOMALY.\n"
#         "- Otherwise, classify as NORMAL.\n\n"
#         "Follow this exact response format:\n"
#         "<LABEL> || <EXPLANATION>\n\n"
#         f"{example['input']}\n"
#         "Answer: "
#         f"{example['output']}"
#     )

#     tokenized = tokenizer(
#         prompt,
#         truncation=True,
#         max_length=384,
#         padding="max_length"
#     )

#     labels = tokenized["input_ids"].copy()
    
#     # Mask everything up to the answer part
#     answer_start = prompt.index("Answer:")
#     prompt_ids = tokenizer(prompt[:answer_start])["input_ids"]
#     mask_length = len(prompt_ids)

#     labels[:mask_length] = -100

#     tokenized["labels"] = tokenized["input_ids"].copy()
#     return tokenized

# tokenized_dataset = dataset.map(format_prompt)

# # -----------------------------
# # 5. Training Args (1 epoch polish)
# # -----------------------------
# training_args = TrainingArguments(
#     output_dir="lora-output-polish",
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=8,
#     num_train_epochs=2,        # polish pass = 1 epoch
#     learning_rate=2e-4,
#     logging_steps=20,
#     save_steps=200,
#     fp16=True,
#     optim="adamw_torch",
# )

# # -----------------------------
# # 6. Trainer
# # -----------------------------
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset
# )

# # -----------------------------
# # 7. Train
# # -----------------------------
# trainer.train()

# # -----------------------------
# # 8. Save LoRA Model
# # -----------------------------
# model.save_pretrained("lora-checkpoint-zcore")
# print("Polish training complete. LoRA checkpoint saved!")
