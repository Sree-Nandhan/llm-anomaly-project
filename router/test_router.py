from router.model_router import ModelRouter

router = ModelRouter()

prompt = """
You are an AI financial analyst.
Analyze this transaction:

Table: user_id, amount, location
Row: 1500, 3200, StoreB
Context: Check anomaly.

Format:
<LABEL> || <EXPLANATION>

Answer:
"""

print("\n=== LLaMA Model ===")
print(router.predict(prompt, strategy="llama"))

print("\n=== Qwen Model ===")
print(router.predict(prompt, strategy="qwen"))

print("\n=== Ensemble ===")
print(router.predict(prompt, strategy="ensemble"))
