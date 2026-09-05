"""Classical ML baseline: TF-IDF + One-vs-Rest Logistic Regression."""

import asyncio
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from src.models.base import ClassifierModel
from src.schema.labels import ALL_LABELS
from src.schema.models import AnnotatedSpeech, Prediction, Speech


# Polish stop words — short curated list (sklearn has no Polish builtin).
POLISH_STOP_WORDS = [
    "a", "aby", "ach", "acz", "aczkolwiek", "aj", "albo", "ale", "ależ", "ani",
    "az", "aż", "bardziej", "bardzo", "bo", "bowiem", "by", "byli", "bynajmniej",
    "być", "był", "była", "było", "były", "będzie", "będą", "cala", "cały",
    "ci", "cię", "ciebie", "co", "cokolwiek", "coś", "czasami", "czasem",
    "czemu", "czy", "czyli", "daleko", "dla", "dlaczego", "dlatego", "do",
    "dobrze", "dokąd", "dość", "dużo", "dwa", "dwaj", "dwie", "dwoje", "dziś",
    "dzisiaj", "gdy", "gdyby", "gdyż", "gdzie", "gdziekolwiek", "gdzieś",
    "i", "ich", "ile", "im", "inna", "inne", "inny", "innych", "iż", "ja",
    "ją", "jak", "jakaś", "jakby", "jaki", "jakichś", "jakie", "jakiś",
    "jakiż", "jakkolwiek", "jako", "jakoś", "je", "jeden", "jedna", "jedno",
    "jednak", "jednakże", "jego", "jej", "jemu", "jest", "jestem", "jeszcze",
    "jeśli", "jeżeli", "już", "ją", "każdy", "kiedy", "kilka", "kimś", "kto",
    "ktokolwiek", "ktoś", "która", "które", "którego", "której", "który",
    "których", "którym", "którzy", "ku", "lub", "ma", "mają", "mało", "mam",
    "mi", "mimo", "między", "mną", "mnie", "mogą", "moi", "moim", "moja",
    "moje", "może", "możliwe", "można", "mój", "mu", "musi", "my", "na",
    "nad", "nam", "nami", "nas", "nasi", "nasz", "nasza", "nasze", "naszego",
    "naszych", "natomiast", "natychmiast", "nawet", "nią", "nic", "nich",
    "nie", "niech", "niego", "niej", "niemu", "nigdy", "nim", "nimi", "niż",
    "no", "o", "obok", "od", "około", "on", "ona", "one", "oni", "ono",
    "oraz", "oto", "owszem", "pan", "pana", "pani", "po", "pod", "podczas",
    "pomimo", "ponad", "ponieważ", "powinien", "powinna", "powinni", "powinno",
    "poza", "prawie", "przecież", "przed", "przede", "przedtem", "przez",
    "przy", "roku", "również", "sam", "sama", "są", "się", "skąd", "sobie",
    "sobą", "sposób", "swoje", "ta", "tak", "taka", "taki", "takie", "także",
    "tam", "te", "tego", "tej", "temu", "ten", "teraz", "też", "to", "tobą",
    "tobie", "toteż", "trzeba", "tu", "tutaj", "twoi", "twoim", "twoja",
    "twoje", "twym", "twój", "ty", "tych", "tylko", "tym", "u", "w", "wam",
    "wami", "was", "wasi", "wasz", "wasza", "wasze", "we", "według", "wiele",
    "wielu", "więc", "więcej", "wszyscy", "wszystkich", "wszystkie", "wszystkim",
    "wszystko", "wtedy", "wy", "właśnie", "z", "za", "zapewne", "zawsze",
    "ze", "zł", "znowu", "znów", "został", "żaden", "żadna", "żadne", "żadnych",
    "że", "żeby",
]


class TfidfLogRegBaseline(ClassifierModel):
    """TF-IDF + One-vs-Rest LogisticRegression for multi-label classification."""

    def __init__(
        self,
        max_features: int = 20000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        threshold: float = 0.5,
        C: float = 1.0,
        random_state: int = 42,
    ):
        self._model_name = "tfidf-logreg"
        self.threshold = threshold
        self.label_to_idx = {label: i for i, label in enumerate(ALL_LABELS)}
        self._per_label_thresholds: dict[str, float] | None = None

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words=POLISH_STOP_WORDS,
            lowercase=True,
            sublinear_tf=True,
        )
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=C,
                random_state=random_state,
                solver="liblinear",
            )
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def _build_label_matrix(self, train_data: list[AnnotatedSpeech]) -> np.ndarray:
        Y = np.zeros((len(train_data), len(ALL_LABELS)), dtype=int)
        for i, entry in enumerate(train_data):
            if entry.gold:
                for label in entry.gold.labels:
                    if label in self.label_to_idx:
                        Y[i, self.label_to_idx[label]] = 1
        return Y

    def fit(self, train_data: list[AnnotatedSpeech]) -> dict:
        texts = [entry.speech.text for entry in train_data]
        Y = self._build_label_matrix(train_data)

        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, Y)

        return {
            "n_train": len(train_data),
            "n_features": X.shape[1],
            "n_labels": Y.shape[1],
        }

    def set_thresholds(self, thresholds: dict[str, float]):
        self._per_label_thresholds = thresholds

    def _predict_batch_sync(self, speeches: list[Speech]) -> list[Prediction]:
        texts = [s.text for s in speeches]
        start = time.perf_counter()
        X = self.vectorizer.transform(texts)
        proba = self.classifier.predict_proba(X)
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_speech_latency = elapsed_ms / max(len(speeches), 1)

        predictions = []
        for i, speech in enumerate(speeches):
            scores = {}
            active_labels = []
            for j, label_name in enumerate(ALL_LABELS):
                score = float(proba[i, j])
                scores[label_name] = score
                threshold = (
                    self._per_label_thresholds[label_name]
                    if self._per_label_thresholds and label_name in self._per_label_thresholds
                    else self.threshold
                )
                if score >= threshold:
                    active_labels.append(label_name)
            predictions.append(Prediction(
                speech_id=speech.speech_id,
                labels=active_labels,
                label_scores=scores,
                model_name=self._model_name,
                latency_ms=per_speech_latency,
            ))
        return predictions

    async def classify(self, speech: Speech) -> Prediction:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._predict_batch_sync, [speech])
        return results[0]

    async def classify_batch(self, speeches: list[Speech]) -> list[Prediction]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_batch_sync, speeches)
