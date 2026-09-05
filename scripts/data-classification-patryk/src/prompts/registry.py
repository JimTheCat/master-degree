"""Prompt registry — manages prompt variants and their configuration."""

import hashlib
from dataclasses import dataclass

from src.prompts.templates import build_system_prompt, build_user_prompt


@dataclass
class PromptVariant:
    name: str
    system_prompt: str
    num_examples: int
    example_selection: str
    version_hash: str


class PromptRegistry:
    """Registry for prompt variants used in experiments."""

    def __init__(self, config: dict):
        prompt_config = config.get("prompt", {})
        self.variant_name = prompt_config.get("variant", "zeroshot")
        self.num_examples = prompt_config.get("num_examples", 0)
        self.example_selection = prompt_config.get("example_selection", "random")

        self._system_prompt = build_system_prompt(self.variant_name)
        self._version_hash = hashlib.sha256(
            self._system_prompt.encode()
        ).hexdigest()[:12]

    @property
    def variant(self) -> PromptVariant:
        return PromptVariant(
            name=self.variant_name,
            system_prompt=self._system_prompt,
            num_examples=self.num_examples,
            example_selection=self.example_selection,
            version_hash=self._version_hash,
        )

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def build_user_prompt(
        self,
        speech_text: str,
        few_shot_examples: list[dict] | None = None,
    ) -> str:
        """Build user prompt, adding few-shot examples if configured."""
        if self.num_examples > 0 and few_shot_examples:
            examples = few_shot_examples[: self.num_examples]
        else:
            examples = None
        return build_user_prompt(speech_text, few_shot_examples=examples)

    @property
    def uses_few_shot(self) -> bool:
        return self.variant_name in ("fewshot", "fewshot_detailed")
