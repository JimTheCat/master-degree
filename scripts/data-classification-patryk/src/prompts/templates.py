"""Prompt templates for parliamentary speech classification."""

from src.schema.labels import (
    ALL_LABEL_INFO,
    EMOTION_INFO,
    RHETORICAL_INFO,
    EmotionLabel,
    RhetoricalLabel,
)


def _format_label_definitions(include_examples: bool = False) -> str:
    """Build the label definitions section for prompts."""
    lines = ["## Kategorie emocji:\n"]
    for emotion, info in EMOTION_INFO.items():
        line = f"- **{info.name}**: {info.description_pl}"
        if include_examples and info.examples:
            examples_str = ", ".join(f'"{ex}"' for ex in info.examples)
            line += f" Przykłady: {examples_str}"
        lines.append(line)

    lines.append("\n## Techniki retoryczne:\n")
    for technique, info in RHETORICAL_INFO.items():
        line = f"- **{info.name}**: {info.description_pl}"
        if include_examples and info.examples:
            examples_str = ", ".join(f'"{ex}"' for ex in info.examples)
            line += f" Przykłady: {examples_str}"
        lines.append(line)

    return "\n".join(lines)


SYSTEM_PROMPT_BASE = """Jesteś ekspertem od analizy dyskursu politycznego i retoryki parlamentarnej.
Twoim zadaniem jest klasyfikacja wypowiedzi parlamentarnych pod kątem emocji i technik retorycznych.

Każda wypowiedź może zawierać WIELE etykiet jednocześnie (multi-label classification).
Przypisz TYLKO te etykiety, które są wyraźnie obecne w tekście.

{label_definitions}

## Format odpowiedzi

Odpowiedz WYŁĄCZNIE poprawnym JSON-em w następującym formacie:
```json
{{
  "labels": ["ETYKIETA_1", "ETYKIETA_2"],
  "confidence": [
    {{"label": "ETYKIETA_1", "score": 0.85}},
    {{"label": "ETYKIETA_2", "score": 0.72}}
  ]
}}
```

Jeśli wypowiedź nie zawiera żadnej z kategorii, zwróć pustą listę labels."""


SYSTEM_PROMPT_DETAILED = """Jesteś ekspertem od analizy dyskursu politycznego i retoryki parlamentarnej w polskim Sejmie.
Twoim zadaniem jest precyzyjna klasyfikacja wypowiedzi parlamentarnych pod kątem emocji i technik retorycznych.

Każda wypowiedź może zawierać WIELE etykiet jednocześnie (multi-label classification).
Przypisz TYLKO te etykiety, które są wyraźnie obecne w tekście.

{label_definitions}

## Kryteria decyzyjne

1. Etykietę przypisuj tylko gdy jest WYRAŹNIE wyrażona, nie domniemana.
2. Rozróżniaj AGRESJA_WERBALNA (bezpośredni atak emocjonalny) od AD_HOMINEM (atak na osobę zamiast argumentu).
3. STRATEGIA_STRACHU wymaga konkretnego straszenia, nie samego wyrażenia troski.
4. OBLEZIONA_TWIERDZA dotyczy przedstawiania SWOJEJ grupy jako ofiary, nie ogólnej krytyki.
5. WHATABOUTISM to konkretne odwracanie uwagi przez "a wy...", nie każde porównanie historyczne.
6. MESJANIZM_MORALNY wymaga poczucia MISJI, nie samej pozytywnej oceny własnych działań.
7. Jedna wypowiedź może mieć 0-13 etykiet — nie wymuszaj etykiet tam, gdzie ich nie ma.

## Format odpowiedzi

Odpowiedz WYŁĄCZNIE poprawnym JSON-em:
```json
{{
  "labels": ["ETYKIETA_1", "ETYKIETA_2"],
  "confidence": [
    {{"label": "ETYKIETA_1", "score": 0.85}},
    {{"label": "ETYKIETA_2", "score": 0.72}}
  ]
}}
```

Jeśli wypowiedź nie zawiera żadnej z kategorii, zwróć pustą listę labels."""


def build_system_prompt(variant: str = "zeroshot") -> str:
    """Build the system prompt for a given variant."""
    include_examples = variant in ("detailed", "fewshot_detailed")

    label_defs = _format_label_definitions(include_examples=include_examples)

    if variant in ("detailed", "fewshot_detailed"):
        return SYSTEM_PROMPT_DETAILED.format(label_definitions=label_defs)
    else:
        return SYSTEM_PROMPT_BASE.format(label_definitions=label_defs)


def build_user_prompt(
    speech_text: str,
    few_shot_examples: list[dict] | None = None,
) -> str:
    """Build the user prompt with optional few-shot examples.

    Args:
        speech_text: The speech text to classify.
        few_shot_examples: List of dicts with 'text' and 'labels' keys.
    """
    parts = []

    if few_shot_examples:
        parts.append("Oto przykłady zaklasyfikowanych wypowiedzi:\n")
        for i, ex in enumerate(few_shot_examples, 1):
            labels_str = ", ".join(f'"{l}"' for l in ex["labels"])
            parts.append(f"### Przykład {i}:")
            parts.append(f"Wypowiedź: \"{ex['text'][:500]}\"")
            parts.append(f"Etykiety: [{labels_str}]\n")
        parts.append("---\n")

    parts.append("Zaklasyfikuj poniższą wypowiedź parlamentarną:\n")
    parts.append(f"\"{speech_text}\"")

    return "\n".join(parts)
