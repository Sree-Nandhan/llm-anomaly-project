import gradio as gr
import pandas as pd
from inference_pipeline import predict

# ------------------------------------------------------------
# CSV BATCH FUNCTION
# ------------------------------------------------------------
def analyze_csv(file):
    if file is None:
        return None, None

    df = pd.read_csv(file.name)

    required_cols = {"user_id", "amount", "location"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain columns: user_id, amount, location")

    results = []
    for _, row in df.iterrows():
        out = predict(
            row["user_id"],
            row["amount"],
            row["location"]
        )
        results.append({
            "user_id": row["user_id"],
            "amount": row["amount"],
            "location": row["location"],
            "label": out["label"],
            "anomaly_prob": out["probabilities"][0],
            "normal_prob": out["probabilities"][1],
            "explanation": out["explanation"]
        })

    result_df = pd.DataFrame(results)

    output_path = "anomaly_results.csv"
    result_df.to_csv(output_path, index=False)

    return result_df, output_path

# ------------------------------------------------------------
# GRADIO UI (CSV-ONLY)
# ------------------------------------------------------------
with gr.Blocks(title="Hybrid AI Anomaly Detection (Batch)") as demo:
    gr.Markdown("## Hybrid AI Anomaly Detection System")
    gr.Markdown(
        """
This application performs **batch anomaly detection on transactional datasets**.

**Architecture**
- Machine Learning classifier for deterministic anomaly labeling
- LLM for human-readable explanations
- Designed for enterprise-style batch analysis
        """
    )

    csv_file = gr.File(
        label="Upload CSV (columns: user_id, amount, location)"
    )
    analyze_btn = gr.Button("Analyze CSV")

    output_table = gr.Dataframe(label="Analysis Results")
    download_file = gr.File(label="Download Results CSV")

    analyze_btn.click(
        analyze_csv,
        inputs=csv_file,
        outputs=[output_table, download_file]
    )

demo.launch()
