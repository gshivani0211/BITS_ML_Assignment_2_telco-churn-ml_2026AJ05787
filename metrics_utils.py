"""Shared metric computation, used by both training and the Streamlit app
so the numbers in the README and in the app are calculated the same way."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Return the six required metrics as a dict.

    Binary problems use the positive class (label 1) for precision/recall/F1.
    Multi-class problems use weighted averages and one-vs-rest AUC.
    """
    avg = "binary" if n_classes == 2 else "weighted"
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                average="weighted", labels=np.arange(n_classes))
    except ValueError:  # e.g. only one class present in the uploaded file
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
