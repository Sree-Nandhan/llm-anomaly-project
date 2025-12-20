**🚨 Hybrid AI Anomaly Detection System**

**(ML + LLM Fine-Tuning + LangChain)**

**📌 Overview**

This project implements a **production-style anomaly detection system** for transactional data using a **hybrid architecture** that combines:

**Classical Machine Learning** for reliable numeric anomaly classification

**Large Language Models (LLMs)** for human-readable explanations

**LoRA fine-tuning (PEFT)** to adapt a local LLaMA model

**LangChain Core (Runnable API)** to orchestrate explanation generation

The system is designed for **batch CSV analysis**, mirroring how anomaly and fraud detection pipelines operate in real-world enterprise systems.

**❓ Motivation**

Pure LLM-based anomaly detection systems often fail when:

numeric thresholds matter

deterministic decisions are required

hallucination or inconsistency is unacceptable

At the same time, traditional ML models lack **interpretability and analyst-friendly explanations**.

This project intentionally separates:

**decision-making (ML)**

**explanation generation (LLM)**

to achieve both correctness and explainability.

**🏗️ System Architecture**
**High-level pipeline:**

<img width="421" height="274" alt="Screenshot 2025-12-20 at 3 09 45 PM" src="https://github.com/user-attachments/assets/a9cc080f-7fb4-4306-9fc3-ff8da03ab4a9" />

**🔍 Core Components**
**1️⃣ Machine Learning — Decision Layer**

**Model:** RandomForestClassifier

**Role:** Deterministic anomaly classification

**Output:** NORMAL / ANOMALY + probability scores

**Features used:**

Transaction amount

Log-scaled amount

Z-score of amount

Flags for extremely small or large values

Transaction location (categorical)

The ML model is the single source of truth for anomaly decisions.

**2️⃣ Large Language Model — Explanation Layer**

**Model:** LLaMA (1B parameters)

**Fine-tuning:** LoRA (via PEFT)

**Role:** Generate concise, domain-specific explanations

Important design choice:

The LLM never decides the label.
It only explains the classifier’s decision.

This prevents hallucination and contradictory outputs.

**🔧 LoRA & PEFT (Why They Matter)**
**PEFT (Parameter-Efficient Fine-Tuning)**

PEFT enables fine-tuning large models by updating only a small subset of parameters, keeping the base model frozen.

**LoRA (Low-Rank Adaptation)**

LoRA injects lightweight trainable matrices into attention layers, allowing:

Efficient domain adaptation

Low memory usage

Local fine-tuning on consumer hardware

In this project, LoRA is used to:

Adapt LLaMA to the transaction anomaly domain

Improve explanation consistency and clarity

Avoid retraining billions of parameters

**🔗 LangChain Integration**

This project integrates **LangChain Core (Runnable API)** to orchestrate the LLM explanation step.

LangChain is used for:

Prompt templating

Runnable composition

Clean separation between logic layers

Future extensibility (RAG, routing, logging)

The system uses modern LangChain primitives rather than deprecated APIs.

**🖥️ User Interface**

A **Gradio-based UI** is provided for batch analysis:

Upload a CSV file

Analyze all transactions

View anomaly labels, probabilities, and explanations

Download enriched results as a CSV

This reflects enterprise-style batch workflows, not toy single-input demos.

**📄 Input Format**
user_id,amount,location
1001,75,StoreA
1002,4500,StoreB
1003,2.5,StoreC
1004,120,RareStoreX

**📤 Output Example**
user_id	amount	location	label	anomaly_prob	explanation
1001	75	StoreA	NORMAL	0.00	Typical spending behavior
1002	4500	StoreB	ANOMALY	1.00	Unusually high transaction
1003	2.5	StoreC	ANOMALY	1.00	Extremely low value
1004	120	RareStoreX	ANOMALY	0.99	Unusual transaction location

**🧠 Key Design Insights**

LLMs are not reliable numeric classifiers

Classical ML models excel at deterministic decisions

LLMs excel at explanation and communication

Hybrid ML + LLM architectures are standard in production

Separating decision logic from explanation logic improves robustness

**🚀 Future Extensions**

Real-time streaming ingestion

Retrieval-augmented explanations (RAG)

Model routing via LangChain

API deployment (FastAPI)

Analyst feedback loops

**🏁 Conclusion**

This project demonstrates a realistic, production-inspired approach to explainable anomaly detection by combining:

Classical ML for correctness

LoRA-fine-tuned LLMs for interpretability

LangChain for orchestration

Clean system design principles

**📬 Contact**

If you’d like to discuss this project or its design decisions, feel free to reach out.
