"""Lightweight interpretation stubs."""

from typing import Dict

import torch


def explain_attention(attention_weights: torch.Tensor) -> Dict[str, float]:
    """Summarize attention weights; replace with richer explainers later."""
    return {
        "mean_attention": float(attention_weights.mean().detach().cpu()),
        "max_attention": float(attention_weights.max().detach().cpu()),
    }

