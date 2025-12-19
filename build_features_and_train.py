# build_features_and_train.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Paths
CSV_PATH = "data/final_clean_transactions.csv"   # generated earlier
OUT_DIR = "classifier_out"
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)
print("Loaded rows:", len(df))
# Ensure label is exactly NORMAL / ANOMALY
df["label"] = df["label"].str.upper().map(lambda x: "ANOMALY" if "ANOM" in x else "NORMAL")

# Numeric features: amount, log_amount, zscore (computed on training set)
df["log_amount"] = np.log1p(df["amount"])

# We'll compute z-score using training mean/std
# split first to avoid leakage
train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])

# compute stats on train
amount_mean = train_df["amount"].mean()
amount_std = train_df["amount"].std(ddof=0) if train_df["amount"].std(ddof=0) > 0 else 1.0

def add_features(df_in):
    d = df_in.copy()
    d["z_amount"] = (d["amount"] - amount_mean) / amount_std
    d["is_small"] = (d["amount"] < 5).astype(int)
    d["is_large"] = (d["amount"] > 2000).astype(int)
    # could add more engineered features here
    return d

train_df = add_features(train_df)
val_df = add_features(val_df)

# Columns for model
num_features = ["amount", "log_amount", "z_amount", "is_small", "is_large"]
cat_features = ["location"]

# Preprocessing
num_transformer = Pipeline([
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

# Model pipeline
clf = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

X_train = train_df[num_features + cat_features]
y_train = train_df["label"]

X_val = val_df[num_features + cat_features]
y_val = val_df["label"]

print("Training classifier...")
clf.fit(X_train, y_train)

# Evaluate
pred_val = clf.predict(X_val)
print("Validation classification report")
print(classification_report(y_val, pred_val))
print("Confusion matrix\n", confusion_matrix(y_val, pred_val))

# Save artifacts
joblib.dump(clf, os.path.join(OUT_DIR, "rf_pipeline.joblib"))
meta = {
    "amount_mean": float(amount_mean),
    "amount_std": float(amount_std),
    "num_features": num_features,
    "cat_features": cat_features
}
joblib.dump(meta, os.path.join(OUT_DIR, "meta.joblib"))

print("Saved classifier and metadata to", OUT_DIR)
