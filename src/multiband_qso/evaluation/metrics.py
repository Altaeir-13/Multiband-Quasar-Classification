from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    """Compute benchmark classification metrics."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    per_class = {
        class_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    }
    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true_array,
            y_pred_array,
            labels=labels,
            target_names=list(class_names),
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            y_true_array,
            y_pred_array,
            labels=labels,
        ).tolist(),
    }
