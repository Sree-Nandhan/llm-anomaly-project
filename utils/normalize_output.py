import re

def normalize_output(text: str):
    """
    Normalize LLM output into the format:
        <LABEL> || <EXPLANATION>
    """

    # Clean raw text
    t = text.strip()

    # Remove XML-like tags
    t = re.sub(r"<[^>]+>", " ", t)

    # Remove non ASCII (e.g. Qwen Chinese tokens)
    t = re.sub(r"[^\x00-\x7F]+", " ", t)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # --------------------
    # 1. Determine LABEL
    # --------------------
    label = "NORMAL"

    # Explicit label cases
    if re.search(r"\bANOMALY\b", t, re.IGNORECASE):
        label = "ANOMALY"
    elif re.search(r"\bANOMALOUS\b", t, re.IGNORECASE):
        label = "ANOMALY"
    elif re.search(r"\bABNORMAL\b", t, re.IGNORECASE):
        label = "ANOMALY"
    elif re.search(r"\bSUSPICIOUS\b", t, re.IGNORECASE):
        label = "ANOMALY"

    # --------------------
    # 2. Extract Explanation
    # --------------------

    explanation = ""

    # Case 1: If model already used ||
    if "||" in t:
        explanation = t.split("||", 1)[1].strip()

    else:
        # Case 2: Extract everything AFTER the label keyword
        match = re.search(r"(ANOMALY|ANOMALOUS|ABNORMAL|SUSPICIOUS)(.*)", t, re.IGNORECASE)
        if match:
            explanation = match.group(2).strip()

        # If still empty, fallback to everything after 'Answer:'
        if explanation == "":
            match = re.search(r"Answer:(.*)", t, re.IGNORECASE)
            if match:
                explanation = match.group(1).strip()

        # If STILL empty, fallback to the whole text
        if explanation == "":
            explanation = t

    # Cleanup explanation
    explanation = explanation.replace("Answer:", "")
    explanation = re.sub(r"Format:.*", " ", explanation, flags=re.IGNORECASE)
    explanation = re.sub(r"\s+", " ", explanation).strip()

    # If explanation still empty
    if explanation == "":
        explanation = "The model did not explain its reasoning."

    # Truncate explanation for UI readability
    if len(explanation) > 250:
        explanation = explanation[:250] + "..."

    # --------------------
    # 3. Final output
    # --------------------
    return f"{label} || {explanation}"
