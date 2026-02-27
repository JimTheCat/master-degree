from typing import List, Any


class MethodInterface:
    """Unified interface for all detection methods.
    Implementations should return (model, predictions) where model may be None for rule-based methods.
    Predictions must be a list or numpy array with integer labels 0/1.
    """
    def fit(self, X_train: List[str], y_train: List[int], params: dict = None) -> Any:
        raise NotImplementedError

    def predict(self, X: List[str]) -> List[int]:
        raise NotImplementedError