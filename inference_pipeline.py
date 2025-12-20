import numpy as np
import pandas as pd
import joblib
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# LangChain CORE only (Python 3.14 compatible)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda


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
# DEVICE SETUP
# ============================================================
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("Using device:", DEVICE)


# ============================================================
# LOAD TOKENIZER + BASE MODEL + LORA
# ============================================================
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LORA_PATH = "lora-checkpoint-final-clean"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE != "cpu" else None,
    device_map="auto" if DEVICE != "cpu" else None
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    device_map="auto" if DEVICE != "cpu" else None
)

if DEVICE == "cpu":
    model.to("cpu")


# ============================================================
# TRANSFORMERS TEXT-GENERATION PIPELINE
# (NO device= ARG — Accelerate handles placement)
# ============================================================
text_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=80,
    do_sample=False
)


# ============================================================
# LANGCHAIN PROMPT (CORE)
# ============================================================
PROMPT_TEMPLATE = """
You are an AI financial analyst.

The classifier has predicted: {label}

Explain THIS decision in one concise sentence.
Do NOT contradict the classifier.
Base the explanation ONLY on the amount and location.

Transaction:
user_id={user_id}
amount={amount}
location={location}

Format:
<LABEL> || <EXPLANATION>

Answer:
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["user_id", "amount", "location", "label"]
)


# ============================================================
# LANGCHAIN RUNNABLE (FIXED PromptValue HANDLING)
# ============================================================
def llm_call(prompt_value) -> str:
    """
    Converts LangChain PromptValue -> string
    then runs HuggingFace text-generation pipeline.
    """
    prompt_text = prompt_value.to_string()
    output = text_gen(prompt_text)[0]["generated_text"]
    return output.strip()

llm_runnable = RunnableLambda(llm_call)

# Full LangChain runnable pipeline
explanation_chain = prompt | llm_runnable


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
# LANGCHAIN EXPLANATION FUNCTION
# ============================================================
def get_llm_explanation(user_id, amount, location, label):
    out = explanation_chain.invoke({
        "user_id": user_id,
        "amount": amount,
        "location": location,
        "label": label
    })

    # Strip any prompt echo
    if "Answer:" in out:
        out = out.split("Answer:", 1)[1]

    return out.strip()


# ============================================================
# FINAL PREDICTION PIPELINE (USED BY GRADIO)
# ============================================================
def predict(user_id, amount, location):
    X_df, feat = compute_features(amount, location)

    label = clf.predict(X_df)[0]
    prob = clf.predict_proba(X_df)[0].tolist()

    explanation = get_llm_explanation(
        user_id=user_id,
        amount=amount,
        location=location,
        label=label
    )

    return {
        "label": label,
        "probabilities": prob,
        "explanation": explanation,
        "features": feat
    }


# ============================================================
# TEST (RUN ONLY WHEN EXECUTED DIRECTLY)
# ============================================================
if __name__ == "__main__":
    tests = [
        (1234, 75, "StoreA"),
        (5678, 4500, "StoreB"),
        (3456, 2.5, "StoreC"),
        (9999, 120, "RareStoreX"),
    ]

    print("\n=== LANGCHAIN CORE HYBRID TEST ===\n")
    for uid, amt, loc in tests:
        out = predict(uid, amt, loc)
        print("------------------------------")
        print(f"INPUT → {amt} at {loc}")
        print("LABEL:", out["label"])
        print("PROBS:", out["probabilities"])
        print("EXPLANATION:", out["explanation"])
        print("FEATURES:", out["features"])
        print("------------------------------\n")
