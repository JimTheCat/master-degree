from typing import Tuple, Any

from hate_speech.services.methods.formal.service import FormalRegex, FormalNegationHeuristic
from hate_speech.services.methods.hybrid.service import HybridVoting
from hate_speech.services.methods.neural.service import NeuralBert, NeuralLSTM
from hate_speech.services.methods.statistical.service import StatisticalBase

METHOD_MAP = {
    'formal_regex': FormalRegex,
    'formal_negation': FormalNegationHeuristic,
    'stat_nb': StatisticalBase,
    'stat_svm': StatisticalBase,
    'stat_logreg': StatisticalBase,
    'stat_randomforest': StatisticalBase,
    'neural_bert': NeuralBert,
    'neural_lstm': NeuralLSTM,
    'hybrid_voting': HybridVoting,
}


def run_method(method_key: str, X_train, y_train, X_test, params: dict = None) -> Tuple[Any, list]:
    """Unified runner that returns (model_obj, preds)."""
    params = params or {}

    if method_key.startswith('formal'):
        cls = METHOD_MAP[method_key]
        m = cls()
        m.fit(X_train, y_train, params)
        preds = m.predict(X_test)
        return m, preds

    if method_key.startswith('stat'):
        sub = method_key.split('_', 1)[1] if '_' in method_key else 'nb'
        m = StatisticalBase()
        p = dict(params)  # kopia, żeby nie modyfikować oryginału
        p['submethod'] = sub
        m.fit(X_train, y_train, p)
        preds = m.predict(X_test)
        return m, preds

    if method_key.startswith('neural'):
        sub = method_key.split('_', 1)[1] if '_' in method_key else 'bert'
        m = NeuralBert() if sub == 'bert' else NeuralLSTM()
        m.fit(X_train, y_train, params)
        preds = m.predict(X_test)
        return m, preds

    if method_key.startswith('hybrid'):
        m = HybridVoting()
        m.fit(X_train, y_train, params)
        preds = m.predict(X_test)
        return m, preds

    raise ValueError('Unknown method')
