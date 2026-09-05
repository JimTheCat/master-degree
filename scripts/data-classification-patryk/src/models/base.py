from abc import ABC, abstractmethod

from src.schema.models import Prediction, Speech


class ClassifierModel(ABC):
    """Abstract base class for all classifiers (transformer, LLM wrappers)."""

    @abstractmethod
    async def classify(self, speech: Speech) -> Prediction:
        """Classify a single speech."""
        ...

    @abstractmethod
    async def classify_batch(self, speeches: list[Speech]) -> list[Prediction]:
        """Classify a batch of speeches."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...
