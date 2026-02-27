from hate_speech.services.methods.base import MethodInterface
from hate_speech.services.methods.formal.service import FormalRegex
from hate_speech.services.methods.statistical.service import StatisticalBase


class HybridVoting(MethodInterface):
    """Ensemble voting between a rule-based and a statistical classifier."""

    def __init__(self, stat_method='nb', formal_pattern=None):
        self.formal = FormalRegex(formal_pattern)
        self.stat = StatisticalBase()
        self._fitted = False

    def fit(self, X_train, y_train, params=None):
        # Fit statistical with given submethod
        params = params or {}
        stat_params = {'submethod': params.get('submethod', 'nb')}
        stat_params.update(params.get('stat_params', {}))
        self.stat.fit(X_train, y_train, stat_params)
        self._fitted = True
        return self

    def predict(self, X):
        formal_preds = self.formal.predict(X)
        stat_preds = self.stat.predict(X)
        out = []
        for f, s in zip(formal_preds, stat_preds):
            # if formal says hate -> keep it; otherwise use stat
            out.append(1 if f == 1 else s)
        return out
