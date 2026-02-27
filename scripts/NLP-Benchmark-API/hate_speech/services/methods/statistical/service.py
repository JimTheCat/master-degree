from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib

from hate_speech.services.methods.base import MethodInterface


class StatisticalBase(MethodInterface):
    def __init__(self):
        self.vectorizer = None
        self.model = None

    def fit(self, X_train: List[str], y_train: List[int], params: dict = None):
        params = params or {}
        method = params.pop('submethod', 'svm')  # używamy tylko do wyboru modelu

        # Oddziel parametry vectorizera od modelu
        vectorizer_params = {}
        model_params = {}
        for k, v in params.items():
            if k in {'max_features', 'ngram_range', 'min_df', 'max_df'}:
                vectorizer_params[k] = v
            else:
                model_params[k] = v

        # Wektoryzacja
        self.vectorizer = TfidfVectorizer(**vectorizer_params)
        Xv = self.vectorizer.fit_transform(X_train)

        # Wybór modelu
        if method == 'svm':
            self.model = SVC(**model_params)
        elif method == 'logreg':
            self.model = LogisticRegression(max_iter=1000, **model_params)
        elif method == 'randomforest':
            self.model = RandomForestClassifier(**model_params)
        elif method == 'nb':
            self.model = MultinomialNB(**model_params)
        else:
            raise ValueError(f'Unknown statistical submethod: {method}')

        self.model.fit(Xv, y_train)
        return self

    def predict(self, X: List[str]) -> List[int]:
        Xv = self.vectorizer.transform(X)
        return self.model.predict(Xv).tolist()

    def save(self, path_prefix: str):
        joblib.dump(
            {'vectorizer': self.vectorizer, 'model': self.model},
            f"{path_prefix}_stat.joblib"
        )
