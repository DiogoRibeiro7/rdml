"""Regularized Distance Metric Learning research software."""

from rdml.core import RDML, FitDiagnostics, paper_step_size, project_psd, squared_mahalanobis
from rdml.evaluation import accuracy_score, knn_predict

__all__ = [
    "RDML",
    "FitDiagnostics",
    "accuracy_score",
    "knn_predict",
    "paper_step_size",
    "project_psd",
    "squared_mahalanobis",
]
__version__ = "0.1.0"
