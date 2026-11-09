"""Legacy models package shim delegating to src.modeling."""

from src.modeling.evaluate import compute_metrics
from src.modeling.explain import ChurnExplainer
from src.modeling.factory import ModelFactory, PipelineFactory

__all__ = [
    "ModelFactory",
    "PipelineFactory",
    "compute_metrics",
    "ChurnExplainer",
]
