def normalize_label(label_raw: str) -> str:
    """
    Normalizes any LLM predicted label into either NORMAL or ANOMALY.
    """

    text = label_raw.lower().strip()

    # Normalize common anomaly-like phrases
    anomaly_keywords = [
        "anomal", "anomaly", "anom", 
        "extra", "extreme", "extremely",
        "rare", "unusual", "suspicious",
        "extraordinary"
    ]

    for k in anomaly_keywords:
        if text.startswith(k) or k in text:
            return "ANOMALY"

    # Normalize common normal-like phrases
    normal_keywords = [
        "normal", "typical", "regular", "common"
    ]

    for k in normal_keywords:
        if text.startswith(k) or k in text:
            return "NORMAL"

    # Fallback
    return "ANOMALY"
