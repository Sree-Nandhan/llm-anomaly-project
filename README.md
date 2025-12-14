**🚀 Multi-Model LLM Anomaly Detection System**

Fine-Tuned LLaMA LoRA + Qwen + Ensemble Router + Streamlit UI

This project is a GenAI-powered anomaly detection system built using:

Fine-tuned LLaMA-3.2-1B-Instruct (LoRA)

Qwen1.5B-Chat (open-source)

Ensemble routing logic

Custom normalization pipeline

Streamlit User Interface

It supports CSV-based tabular anomaly detection and returns structured outputs:

LABEL || EXPLANATION


This project demonstrates LLM fine-tuning, model orchestration, data engineering, and UI deployment, making it suitable for Data Science, ML Engineer, and GenAI Engineer portfolio use.

**📌 Features**

 **1. Fine-Tuned LLaMA Model (LoRA)**

Trained on synthetic financial transaction data

Learns anomaly vs. normal patterns

Produces structured explanations

 **2. Multi-Model Routing (Qwen + LLaMA + Ensemble)**

LLaMA: fine-tuned LoRA weights

Qwen: open-source reasoning baseline

Ensemble: safety-oriented combined decision

 **3. Streamlit UI**

Upload CSV files

Select your model: LLaMA / Qwen / Ensemble

View anomaly classifications

Highlighted output table

Downloadable results

 **4. Output Normalization Engine**

Ensures every LLM output follows consistent format:

ANOMALY || This transaction amount is unusually high...

 **5. Modular Architecture**

models/ → model wrappers

router/ → model routing logic

utils/ → normalization helpers

app/ → Streamlit UI

**🧱 Project Architecture**

<img width="552" height="489" alt="Screenshot 2025-12-14 at 6 39 49 PM" src="https://github.com/user-attachments/assets/1fb96a61-4e9d-43f1-9d61-0f985b1cc675" />

**🧪 Usage Guide**

**1️ Install dependencies**

pip install -r requirements.txt


(Or your current environment requirements.)

**2️ Run the Streamlit App**

streamlit run app/streamlit_app.py


Upload a file shaped like:

user_id,amount,location
101,80,StoreA
102,120,StoreB
103,4500,StoreC
...

**3️ Choose your model**

llama → fine-tuned LoRA model

qwen → baseline open-source model

ensemble → hybrid safer classifier

**4️ Output Format**

Each classifier returns:

LABEL || EXPLANATION


Example:

ANOMALY || The amount 4500 is significantly higher than typical transactions.

**🧠 Model Details**

Fine-Tuned LLaMA (LoRA)

Base model: meta-llama/Llama-3.2-1B-Instruct

Method: QLoRA-style adapter training

Dataset: 2,000 synthetic financial rows

Eval accuracy: 100% on validation subset

Polished using instructional prompt-formatting pass

Qwen1.5B-Chat Model

Serves as a general reasoning companion

Produces richer explanations

Ensemble Logic
If LLaMA or Qwen flags ANOMALY → label = ANOMALY
Else label = NORMAL

**🏎️ Performance Notes**

First UI load may take a few seconds due to model loading

Cached model loading ensures fast switching

Best run using Apple Silicon with MPS acceleration (your setup)

**🎯 Future Improvements**

Add improved anomaly logic (statistical or ML-based)

Tune explanation generation consistency

Add GPT-4/Gemini as an additional model option

Add a confidence score + calibration

Add database integration (Snowflake / BigQuery)

Deploy Streamlit app online (Streamlit Cloud, HuggingFace Spaces)

**📄 License**

You can choose any open-source license.
MIT License is recommended and standard.

**🙌 Acknowledgements**

This project integrates:

HuggingFace Transformers

LangChain Model Routing

PEFT (LoRA Adapters)

Streamlit UI Framework

Qwen & LLaMA community models

**🚀 Final Notes**

This project is highly impressive in a portfolio because it demonstrates:

Full LLM lifecycle

Fine-tuning with LoRA

Multi-model engineering

Routing systems

UI + MLOps mindset

Evaluation and dataset engineering
