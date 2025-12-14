import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
from router.model_router import ModelRouter
from utils.normalize_output import normalize_output



# -------------------------
# Load router (multi-model)
# -------------------------
@st.cache_resource
def load_router():
    return ModelRouter()

router = load_router()


# -------------------------
# Streamlit UI Layout
# -------------------------
st.set_page_config(page_title="Multi-Model Anomaly Detection", layout="wide")

st.title("🔍 Multi-Model LLM Anomaly Detection System")
st.write("Fine-tuned LLaMA + Qwen + Ensemble (LoRA-powered)")

# Model selection
model_choice = st.selectbox(
    "Choose Model:",
    ["llama", "qwen", "ensemble"]
)

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df)

    # Ensure required columns exist
    required_cols = ["user_id", "amount", "location"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()

    # Run predictions
    st.subheader("Running Predictions...")
    results = []

    for idx, row in df.iterrows():
        user_id = row["user_id"]
        amount = row["amount"]
        location = row["location"]

        prompt = f"""
        You are an AI financial analyst.

        Table: user_id, amount, location
        Row: {user_id}, {amount}, {location}
        Context: Analyze for anomaly.

        Format:
        <LABEL> || <EXPLANATION>

        Answer:
        """

        out = router.predict(prompt, strategy=model_choice)
        label, explanation = out.split("||", 1)
        results.append({
            "user_id": user_id,
            "amount": amount,
            "location": location,
            "label": label.strip(),
            "explanation": explanation.strip()
        })

    # Convert to DataFrame
    result_df = pd.DataFrame(results)

    # Highlight anomalies
    def highlight_anomaly(row):
        color = "#ffcccc" if row["label"] == "ANOMALY" else "white"
        return [f"background-color: {color}"] * len(row)

    st.subheader("Analysis Results")
    st.dataframe(result_df.style.apply(highlight_anomaly, axis=1))

    # Download button
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="anomaly_results.csv",
        mime="text/csv"
    )
