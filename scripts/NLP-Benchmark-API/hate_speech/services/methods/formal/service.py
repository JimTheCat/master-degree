import re
from typing import List, Any

from hate_speech.services.methods.base import MethodInterface


class FormalRegex(MethodInterface):
    def __init__(self, pattern: str = None):
        self.pattern = re.compile(pattern or r"\b(nienawidze?|wulgaryzm|obraza|obraźn[ia]\w*)\b", flags=re.I | re.U)

    def fit(self, X_train: List[str], y_train: List[int], params: dict = None) -> Any:
        # no training for regex
        return None

    def predict(self, X: List[str]) -> List[int]:
        preds = [1 if self.pattern.search(text or "") else 0 for text in X]
        return preds

# small grammar-based example using simple token checks
class FormalNegationHeuristic(MethodInterface):
    def __init__(self, hate_terms=None, negation_tokens=None):
        self.hate_terms = set(hate_terms or ["nienawidze", "obraza", "nienawisci"])
        self.negation_tokens = set(negation_tokens or ["nie", "bez", "nigdy"])

    def fit(self, X_train, y_train, params=None):
        return None

    def predict(self, X: List[str]) -> List[int]:
        out = []
        for text in X:
            tokens = (text or "").lower().split()
            score = 0
            for i, t in enumerate(tokens):
                if any(ht in t for ht in self.hate_terms):
                    # check window for negation before the term
                    window = tokens[max(0, i-3):i]
                    if any(n in window for n in self.negation_tokens):
                        score -= 1
                    else:
                        score += 1
            out.append(1 if score > 0 else 0)
        return out