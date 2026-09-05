"""Unified LangChain-based classifier — replaces per-provider wrappers.

Supports OpenAI, Anthropic, Google Gemini, and LMStudio (OpenAI-compatible local
endpoint, used for Bielik / Gemma 4). Structured output is enforced via Pydantic
schema bound through LangChain's with_structured_output.
"""

import asyncio
import os
import re
import time

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

from src.models.base import ClassifierModel
from src.prompts.registry import PromptRegistry
from src.schema.llm_output import LLMPredictionSchema
from src.schema.models import Prediction, Speech

PROVIDER_CONCURRENCY: dict[str, int] = {
    "openai": 20,
    "anthropic": 15,
    "google_genai": 4,
    # LMStudio (llama.cpp) splits the loaded context window across parallel
    # slots — 4 concurrent requests quarter the usable context and long
    # speeches start failing with "Context size has been exceeded".
    # Override with `model.concurrency` if the server has a large context.
    "lmstudio": 1,
}

# LMStudio-hosted open models rarely expose reliable tool-calling; force json_mode.
# Other providers use the provider's native default (function_calling / json_schema).
PROVIDER_STRUCTURED_METHOD: dict[str, str | None] = {
    "openai": None,
    "anthropic": None,
    # Gemini function-calling mode truncates Literal enum values (returns "D" etc.).
    # json_mode forces raw JSON output validated post-hoc by Pydantic.
    "google_genai": "json_mode",
    # LMStudio only supports json_schema or text — json_mode not available.
    # Mitigate verbose confidence output by raising max_completion_tokens in model config.
    "lmstudio": "json_schema",
}


def _build_llm(
    provider: str,
    model_name: str,
    temperature: float | None,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    extra: dict | None = None,
) -> BaseChatModel:
    extra = extra or {}
    if provider == "lmstudio":
        base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        kwargs: dict = dict(
            base_url=base_url,
            api_key=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
            model=model_name,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **extra,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        # Raw request fields (e.g. chat_template_kwargs to disable thinking on
        # reasoning models like Qwen3) — passed straight to the server.
        extra_body = kwargs.pop("extra_body", None)
        if extra_body:
            kwargs["extra_body"] = extra_body
        return ChatOpenAI(**kwargs)
    llm_kwargs: dict = dict(
        model=model_name,
        model_provider=provider,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        **extra,
    )
    if temperature is not None:
        llm_kwargs["temperature"] = temperature
    return init_chat_model(**llm_kwargs)


def _extract_valid_labels(text: str) -> list[str]:
    """Return exact taxonomy labels found in a raw (possibly malformed) response.

    Prefers the content of a "labels": [...] fragment when present, so label
    names repeated in a confidence section are not double-counted.
    """
    from src.schema.labels import ALL_LABELS

    if not text:
        return []
    match = re.search(r'"labels"\s*:\s*\[([^\]]*)', text)
    haystack = match.group(1) if match else text
    return [label for label in ALL_LABELS if label in haystack]


class LangChainClassifier(ClassifierModel):
    """Provider-agnostic classifier backed by LangChain."""

    def __init__(
        self,
        config: dict,
        prompt_registry: PromptRegistry,
        few_shot_examples: list[dict] | None = None,
    ):
        model_config = config.get("model", {})
        self.provider: str = model_config.get("type", "openai")
        self._model_name_str: str = model_config.get("name", "gpt-4o-mini")
        self.temperature: float | None = model_config.get("temperature", None)
        self.max_completion_tokens: int = model_config.get("max_completion_tokens", 500)
        # Local models can take minutes per long speech; a hung request without a
        # timeout stalls the whole batch, so both are configurable per model config.
        self.request_timeout: float = float(model_config.get("request_timeout", 300))
        self.max_retries: int = int(model_config.get("max_retries", 2))
        self.prompt_registry = prompt_registry
        self.few_shot_examples = few_shot_examples

        extra_kwargs: dict = {}
        for key in ("frequency_penalty", "presence_penalty", "top_p", "seed", "extra_body"):
            if key in model_config:
                extra_kwargs[key] = model_config[key]

        llm = _build_llm(
            self.provider,
            self._model_name_str,
            self.temperature,
            self.max_completion_tokens,
            timeout=self.request_timeout,
            max_retries=self.max_retries,
            extra=extra_kwargs,
        )
        structured_kwargs: dict = {"include_raw": True}
        method = model_config.get(
            "structured_output_method",
            PROVIDER_STRUCTURED_METHOD.get(self.provider),
        )
        if method:
            structured_kwargs["method"] = method
        self._structured_llm = llm.with_structured_output(
            LLMPredictionSchema,
            **structured_kwargs,
        )
        self._concurrency = int(
            model_config.get("concurrency", PROVIDER_CONCURRENCY.get(self.provider, 10))
        )
        # LMStudio reasoning-model workaround (see _classify_lmstudio_raw).
        self._raw_client = None
        self._prefer_raw = False

    @property
    def model_name(self) -> str:
        return self._model_name_str

    def _build_messages(self, speech_text: str) -> list:
        examples = None
        if self.prompt_registry.uses_few_shot and self.few_shot_examples:
            examples = self.few_shot_examples[: self.prompt_registry.num_examples]
        from src.prompts.templates import build_user_prompt

        user_text = build_user_prompt(speech_text, few_shot_examples=examples)
        return [
            SystemMessage(content=self.prompt_registry.system_prompt),
            HumanMessage(content=user_text),
        ]

    @traceable(name="langchain_classify", run_type="llm")
    async def classify(self, speech: Speech) -> Prediction:
        messages = self._build_messages(speech.text)

        if self.provider == "lmstudio" and self._prefer_raw:
            return await self._classify_lmstudio_raw(speech.speech_id, messages)

        start = time.perf_counter()
        try:
            result = await self._structured_llm.ainvoke(messages)
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            # LengthFinishReasonError — response truncated before JSON closed.
            # Try to salvage labels from partial raw content via regex.
            exc_str = str(exc)
            if "LengthFinishReasonError" in type(exc).__name__ or "length" in exc_str.lower():
                salvaged = self._salvage_truncated(exc_str, speech.speech_id, latency)
                if salvaged is not None:
                    return salvaged
            raise
        latency = (time.perf_counter() - start) * 1000
        prediction = self._result_to_prediction(speech.speech_id, result, latency)

        # Reasoning models on LMStudio (e.g. Qwen3) put the grammar-forced JSON
        # in the thinking channel, which LangChain drops entirely. Retry via the
        # raw OpenAI client; if that works, use it directly for later speeches.
        if (
            self.provider == "lmstudio"
            and (prediction.raw_response or "").startswith("PARSE_FAILED")
        ):
            raw_prediction = await self._classify_lmstudio_raw(speech.speech_id, messages)
            if not (raw_prediction.raw_response or "").startswith("PARSE_FAILED"):
                self._prefer_raw = True
                return raw_prediction
        return prediction

    async def _classify_lmstudio_raw(self, speech_id: str, messages: list) -> Prediction:
        """Direct OpenAI-client call for LMStudio, reading the reasoning channel.

        LMStudio wraps a reasoning model's whole output in a <think> block when
        the JSON grammar prevents closing it, so the JSON lands in
        message.reasoning_content — a field langchain-openai does not expose.
        """
        from openai import AsyncOpenAI

        if self._raw_client is None:
            self._raw_client = AsyncOpenAI(
                base_url=os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
                api_key=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

        payload = [
            {
                "role": "system" if isinstance(m, SystemMessage) else "user",
                "content": m.content,
            }
            for m in messages
        ]
        kwargs: dict = dict(
            model=self._model_name_str,
            messages=payload,
            max_tokens=self.max_completion_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "prediction",
                    "schema": LLMPredictionSchema.model_json_schema(),
                },
            },
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        start = time.perf_counter()
        response = await self._raw_client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - start) * 1000

        message = response.choices[0].message
        extras = message.model_extra or {}
        text = message.content or extras.get("reasoning_content") or ""
        usage = response.usage
        token_usage = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }

        try:
            parsed = LLMPredictionSchema.model_validate_json(text.strip())
        except Exception as exc:
            salvaged = _extract_valid_labels(text)
            prefix = "PARSE_SALVAGED" if salvaged else "PARSE_FAILED"
            return Prediction(
                speech_id=speech_id,
                labels=salvaged,
                label_scores={},
                model_name=self._model_name_str,
                prompt_variant=self.prompt_registry.variant.name,
                raw_response=f"{prefix} (raw fallback, {exc!r}): {text}"[:500],
                latency_ms=latency,
                token_usage=token_usage,
            )

        raw_note = text if message.content else f"REASONING_CHANNEL: {text}"
        return Prediction(
            speech_id=speech_id,
            labels=list(parsed.labels),
            label_scores=parsed.confidence_dict(),
            model_name=self._model_name_str,
            prompt_variant=self.prompt_registry.variant.name,
            raw_response=raw_note[:500],
            latency_ms=latency,
            token_usage=token_usage,
        )

    def _salvage_truncated(
        self, exc_str: str, speech_id: str, latency_ms: float
    ) -> "Prediction | None":
        """Best-effort label extraction from a truncated JSON response.

        Looks for any valid label strings in the raw exception text or
        partial JSON. Returns a Prediction with whatever was found, or None
        if nothing recoverable.
        """
        from src.schema.labels import ALL_LABELS

        found = [label for label in ALL_LABELS if label in exc_str]
        # Also try to extract from partial JSON fragment if present.
        json_match = re.search(r'"labels"\s*:\s*\[([^\]]*)', exc_str)
        if json_match:
            fragment = json_match.group(1)
            for label in ALL_LABELS:
                if label in fragment and label not in found:
                    found.append(label)
        return Prediction(
            speech_id=speech_id,
            labels=found,
            label_scores={},
            model_name=self._model_name_str,
            prompt_variant=self.prompt_registry.variant.name,
            raw_response=f"TRUNCATED (salvaged {len(found)} labels): {exc_str}"[:500],
            latency_ms=latency_ms,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0},
        )

    async def classify_batch(self, speeches: list[Speech]) -> list[Prediction]:
        semaphore = asyncio.Semaphore(self._concurrency)

        async def _run(speech: Speech) -> Prediction:
            async with semaphore:
                try:
                    return await self.classify(speech)
                except Exception as exc:
                    return Prediction(
                        speech_id=speech.speech_id,
                        labels=[],
                        label_scores={},
                        model_name=self._model_name_str,
                        prompt_variant=self.prompt_registry.variant.name,
                        raw_response=f"ERROR: {type(exc).__name__}: {exc}"[:500],
                        latency_ms=None,
                        token_usage={"prompt_tokens": 0, "completion_tokens": 0},
                    )

        return await asyncio.gather(*(_run(s) for s in speeches))

    def _result_to_prediction(
        self,
        speech_id: str,
        result: dict,
        latency_ms: float,
    ) -> Prediction:
        """Map the (raw + parsed) response dict into our Prediction schema."""
        parsed: LLMPredictionSchema | None = result.get("parsed")
        raw_msg = result.get("raw")
        parsing_error = result.get("parsing_error")

        raw_content = ""
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if raw_msg is not None:
            content = raw_msg.content
            raw_content = content if isinstance(content, str) else str(content)
            usage = getattr(raw_msg, "usage_metadata", None) or {}
            token_usage = {
                "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
                "completion_tokens": int(usage.get("output_tokens", 0) or 0),
            }

        if parsed is None:
            # Reasoning models (e.g. Qwen3 on LMStudio) emit the grammar-forced
            # JSON inside the thinking channel, leaving content empty — the
            # chat template opens a <think> block the JSON grammar can't close.
            # Recover the full schema from reasoning_content when possible.
            reasoning = ""
            if raw_msg is not None:
                extras = getattr(raw_msg, "additional_kwargs", None) or {}
                reasoning = extras.get("reasoning_content") or extras.get("reasoning") or ""
            if not raw_content and reasoning:
                try:
                    parsed = LLMPredictionSchema.model_validate_json(reasoning.strip())
                except Exception:
                    parsed = None
                if parsed is not None:
                    raw_content = f"REASONING_CHANNEL: {reasoning}"

        if parsed is None:
            # Schema validation failed (truncated JSON, invalid enum value, etc.).
            # Salvage whatever exact label names appear in the raw output instead
            # of failing the whole item — one bad response shouldn't zero a run.
            salvaged = _extract_valid_labels(raw_content) or _extract_valid_labels(reasoning)
            prefix = "PARSE_SALVAGED" if salvaged else "PARSE_FAILED"
            return Prediction(
                speech_id=speech_id,
                labels=salvaged,
                label_scores={},
                model_name=self._model_name_str,
                prompt_variant=self.prompt_registry.variant.name,
                raw_response=f"{prefix} ({parsing_error!r}): {raw_content or reasoning}"[:500],
                latency_ms=latency_ms,
                token_usage=token_usage,
            )

        return Prediction(
            speech_id=speech_id,
            labels=list(parsed.labels),
            label_scores=parsed.confidence_dict(),
            model_name=self._model_name_str,
            prompt_variant=self.prompt_registry.variant.name,
            raw_response=raw_content or parsed.model_dump_json(),
            latency_ms=latency_ms,
            token_usage=token_usage,
        )
