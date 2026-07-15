"""Abstract interfaces for deployment backends (no inference in Phase C.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class InferenceBackend(ABC):
    """Contract for model backends implemented in later Phase C steps."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable deployable model identifier."""

    @abstractmethod
    def load(self) -> None:
        """Load weights into memory."""

    @abstractmethod
    def predict(self, ct_path: Path, *, mask_path: Path | None = None) -> dict[str, Any]:
        """Run synchronous inference and return a clinical result envelope."""
