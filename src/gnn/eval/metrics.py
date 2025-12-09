"""Model metric calculations."""

from typing import Dict, Tuple

import torch
from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """Compute accuracy and macro F1 for classification outputs."""
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    y_true = targets.cpu().numpy()
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
    }

