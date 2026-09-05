"""Experiment runner — loads config and creates the appropriate classifier."""

from pathlib import Path

import yaml

from src.models.base import ClassifierModel
from src.prompts.few_shot import select_examples
from src.prompts.registry import PromptRegistry
from src.schema.models import AnnotatedSpeech


def load_config(config_path: str) -> dict:
    """Load a YAML config, resolving _base_ inheritance."""
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "_base_" in config:
        base_path = config_path.parent / config.pop("_base_")
        base_config = load_config(str(base_path))
        config = _deep_merge(base_config, config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def create_classifier(
    config: dict,
    train_data: list[AnnotatedSpeech] | None = None,
) -> ClassifierModel:
    """Factory function to create the appropriate classifier from config.

    Args:
        config: Loaded experiment config dict.
        train_data: Training data (used for few-shot example selection in LLM classifiers).
    """
    model_type = config.get("model", {}).get("type", "openai")
    prompt_registry = PromptRegistry(config)

    # Prepare few-shot examples if needed
    few_shot_examples = None
    if prompt_registry.uses_few_shot and train_data:
        prompt_config = config.get("prompt", {})
        few_shot_examples = select_examples(
            pool=train_data,
            strategy=prompt_config.get("example_selection", "stratified"),
            n=prompt_config.get("num_examples", 3),
            seed=config.get("project", {}).get("random_seed", 42),
        )

    if model_type == "transformer":
        from src.models.transformer_baseline import TransformerBaseline

        model_config = config.get("model", {})
        return TransformerBaseline(
            model_name=model_config.get("name", "allegro/herbert-base-cased"),
            max_length=model_config.get("max_length", 512),
        )
    elif model_type in ("openai", "anthropic", "google_genai", "lmstudio"):
        from src.models.langchain_classifier import LangChainClassifier

        return LangChainClassifier(config, prompt_registry, few_shot_examples)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
