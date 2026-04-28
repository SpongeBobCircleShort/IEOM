"""Deep temporal models (PyTorch GRU default + pure-Python fallback backend)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None

from hesitation.ml.logistic import BinaryLogisticRegression, OVRLogisticModel

if nn is not None:

    class TorchGRUMultiHead(nn.Module):
        """Default deep model: shared GRU encoder + 3 prediction heads."""

        def __init__(self, input_dim: int, hidden_dim: int, n_state_classes: int) -> None:
            super().__init__()
            self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            self.dropout = nn.Dropout(p=0.1)
            self.state_head = nn.Linear(hidden_dim, n_state_classes)
            self.future_hes_head = nn.Linear(hidden_dim, 1)
            self.future_corr_head = nn.Linear(hidden_dim, 1)

        def forward(self, x: Any) -> tuple[Any, Any, Any]:
            out, _ = self.gru(x)
            h = self.dropout(out[:, -1, :])
            return self.state_head(h), self.future_hes_head(h), self.future_corr_head(h)

else:

    class TorchGRUMultiHead:  # pragma: no cover - only used when torch unavailable
        """Runtime placeholder when torch is not available."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("PyTorch is required for TorchGRUMultiHead")


@dataclass(slots=True)
class FallbackDeepModel:
    """Fallback temporal baseline when torch is unavailable.

    Uses flattened sequence features with one-vs-rest + binary logistic heads.
    """

    classes: list[str]
    state_model: OVRLogisticModel
    future_hes_model: BinaryLogisticRegression
    future_corr_model: BinaryLogisticRegression

    def predict_state_proba(self, x: list[list[float]]) -> list[dict[str, float]]:
        return self.state_model.predict_proba(x)

    def predict_state(self, x: list[list[float]]) -> list[str]:
        return self.state_model.predict(x)

    def predict_future(self, x: list[list[float]]) -> tuple[list[float], list[float]]:
        return self.future_hes_model.predict_proba(x), self.future_corr_model.predict_proba(x)
