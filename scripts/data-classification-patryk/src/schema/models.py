from datetime import date
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.schema.labels import ALL_LABELS


class Speech(BaseModel):
    speech_id: str
    text: str
    speaker: str
    party: str
    political_group: str | None = None
    date: date
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoldAnnotation(BaseModel):
    speech_id: str
    labels: list[str]
    annotator: str | None = None
    annotation_date: date | None = None
    notes: str | None = None

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, v: list[str]) -> list[str]:
        invalid = [label for label in v if label not in ALL_LABELS]
        if invalid:
            raise ValueError(f"Invalid labels: {invalid}. Valid labels: {ALL_LABELS}")
        return v


class Prediction(BaseModel):
    speech_id: str
    labels: list[str]
    label_scores: dict[str, float] = Field(default_factory=dict)
    model_name: str
    prompt_variant: str | None = None
    raw_response: str | None = None
    latency_ms: float | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, v: list[str]) -> list[str]:
        invalid = [label for label in v if label not in ALL_LABELS]
        if invalid:
            raise ValueError(f"Invalid labels: {invalid}. Valid labels: {ALL_LABELS}")
        return v


class AnnotatedSpeech(BaseModel):
    speech: Speech
    gold: GoldAnnotation | None = None
    prediction: Prediction | None = None


class GoldStandardEntry(BaseModel):
    """Single entry in gold_standard/speeches.jsonl — speech + inline annotations."""
    speech_id: str
    text: str
    speaker: str
    party: str
    political_group: str | None = None
    date: date
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    annotations: GoldAnnotation | None = None

    def to_speech(self) -> Speech:
        return Speech(
            speech_id=self.speech_id,
            text=self.text,
            speaker=self.speaker,
            party=self.party,
            political_group=self.political_group,
            date=self.date,
            session_id=self.session_id,
            metadata=self.metadata,
        )

    def to_annotated_speech(self) -> AnnotatedSpeech:
        return AnnotatedSpeech(
            speech=self.to_speech(),
            gold=self.annotations,
        )
