"""Churn model bench: upload Telco customer rows, choose a model, and see how
well it predicts churn. Saved models are loaded only; nothing is retrained."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from metrics_utils import compute_metrics
from preprocessing import ID_COL, TARGET, clean_telco

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "Model"

st.set_page_config(page_title="Churn model bench", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    models = {n: joblib.load(MODEL_DIR / f) for n, f in meta["model_files"].items()}
    return meta, models


meta, models = load_artifacts()
FEATURES, CLASSES = meta["features"], meta["classes"]
POS = CLASSES.index(meta["positive_class"])

with st.sidebar:
    st.header("Customer data")
    uploaded = st.file_uploader("Upload test customers (CSV in the original Telco format)",
                                type="csv")
    sample = ROOT / "test_data.csv"
    if sample.exists():
        st.download_button("Download test_data.csv", sample.read_bytes(),
                           "test_data.csv", "text/csv")
    st.header("Model")
    model_name = st.selectbox("Classifier", list(models))

st.title("Which customers are about to leave?")
st.caption("Five classifiers trained on the Telco Customer Churn dataset. "
           "Upload held-out customers to see how each one performs.")

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to start. Below are the scores each "
            "model reached on the held-out 20% split during training.")
    st.dataframe(pd.DataFrame(meta["test_metrics"]).set_index("ML Model Name"),
                 use_container_width=True)
    st.stop()

raw = pd.read_csv(uploaded)
missing = [c for c in FEATURES + [TARGET] if c not in raw.columns]
if missing:
    st.error(f"Missing columns: {', '.join(missing)}. The file needs every Telco "
             "feature column plus Churn.")
    st.stop()

df = clean_telco(raw)
y = df[TARGET].map({c: i for i, c in enumerate(CLASSES)})
if y.isna().any():
    st.error(f"Churn must contain only {' / '.join(CLASSES)}.")
    st.stop()
y = y.astype(int).to_numpy()
X = df[FEATURES]

tab_model, tab_compare, tab_data = st.tabs(
    ["Selected model", "Compare all five", "Uploaded customers"])

with tab_model:
    model = models[model_name]
    y_pred, y_proba = model.predict(X), model.predict_proba(X)
    scores = compute_metrics(y, y_pred, y_proba, len(CLASSES))
    st.subheader(f"{model_name} · {len(df)} customers")
    for col, (k, v) in zip(st.columns(6), scores.items()):
        col.metric(k, f"{v:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, y_pred, labels=range(len(CLASSES)))
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        names = ["Stayed", "Churned"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False,
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        missed = cm[POS].sum() - cm[POS, POS]
        st.caption(f"{cm[POS, POS]} churners caught, {missed} missed, "
                   f"{cm[1 - POS, POS]} loyal customers wrongly flagged.")
    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y, y_pred, labels=range(len(CLASSES)),
                                    target_names=CLASSES, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

    st.markdown("**Highest churn risk according to this model**")
    risk = raw[[c for c in [ID_COL, "Contract", "tenure", "MonthlyCharges", TARGET]
                if c in raw.columns]].copy()
    risk["Churn probability"] = y_proba[:, POS].round(3)
    st.dataframe(risk.sort_values("Churn probability", ascending=False).head(15),
                 use_container_width=True, hide_index=True)

with tab_compare:
    rows = {n: compute_metrics(y, m.predict(X), m.predict_proba(X), len(CLASSES))
            for n, m in models.items()}
    table = pd.DataFrame(rows).T.round(4)
    table.index.name = "ML Model Name"
    st.dataframe(table.style.highlight_max(axis=0, color="#fde2c8"),
                 use_container_width=True)
    metric = st.radio("Chart", list(table.columns), horizontal=True, index=5)
    st.bar_chart(table[metric])

with tab_data:
    st.write(f"{raw.shape[0]} customers, {raw.shape[1]} columns")
    st.dataframe(raw.head(20), use_container_width=True)
    st.bar_chart(raw[TARGET].value_counts())
