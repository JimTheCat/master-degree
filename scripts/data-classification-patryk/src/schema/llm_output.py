"""Pydantic schema for LLM structured output."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

LabelName = Literal[
    "AGRESJA_WERBALNA",
    "STRATEGIA_STRACHU",
    "DEHUMANIZACJA_POGARDA",
    "DUMA_I_SUKCES",
    "MESJANIZM_MORALNY",
    "POLARYZACJA_MY_ONI",
    "AD_HOMINEM",
    "OBLEZIONA_TWIERDZA",
    "PRZYPISYWANIE_ZLYCH_INTENCJI",
    "WHATABOUTISM",
    "SOFIZMAT_ROZSZERZENIA",
    "DOWOD_ANEGDOTYCZNY",
    "APEL_O_JEDNOSC",
]


class LabelConfidence(BaseModel):
    """Single label score — list form enforces enum better than dict keys."""

    label: LabelName
    score: float = Field(ge=0.0, le=1.0)

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Clamp out-of-range scores to [0, 1].

        Local models (Gemma, Bielik) sometimes return scores > 1 (e.g. 1.1, 10.0)
        when using json_mode instead of a constrained schema.
        """
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, v))


class LLMPredictionSchema(BaseModel):
    """Enforced output schema bound via LangChain's with_structured_output.

    Reasoning field was dropped — open-text generation triggers repetition
    loops on weak local models and isn't consumed downstream anyway.
    """

    labels: list[LabelName] = Field(
        default_factory=list,
        description="Wykryte etykiety (0-13 z dozwolonego zbioru).",
    )
    confidence: list[LabelConfidence] = Field(
        default_factory=list,
        description="Pewność per etykieta (0.0-1.0).",
    )

    @field_validator("labels", mode="before")
    @classmethod
    def drop_empty_labels(cls, v: list) -> list:
        """Filter out empty strings before Literal validation.

        Some models (e.g. Gemini) return [""] instead of [] for no labels.
        """
        if isinstance(v, list):
            return [item for item in v if item != "" and item is not None]
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def drop_empty_confidence(cls, v: list) -> list:
        """Same guard for confidence list."""
        if isinstance(v, list):
            return [item for item in v if item]
        return v

    def confidence_dict(self) -> dict[str, float]:
        return {c.label: c.score for c in self.confidence}
