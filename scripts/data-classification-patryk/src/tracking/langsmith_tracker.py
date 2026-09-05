import os
import random
from typing import Any

from langsmith import Client


class LangSmithTracker:
    """Wrapper around LangSmith for LLM call tracing."""

    def __init__(
        self,
        project_name: str = "parliament-speech-analysis",
        trace_sample_rate: float = 1.0,
    ):
        self.project_name = project_name
        self.trace_sample_rate = trace_sample_rate
        self._client = None

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client()
        return self._client

    def should_trace(self) -> bool:
        """Determine if this call should be traced (based on sample rate)."""
        if self.trace_sample_rate >= 1.0:
            return True
        return random.random() < self.trace_sample_rate

    def create_dataset(
        self,
        dataset_name: str,
        speeches_with_gold: list[dict],
    ) -> str:
        """Upload gold standard as a LangSmith dataset for evaluation.

        Args:
            dataset_name: Name for the dataset in LangSmith.
            speeches_with_gold: List of dicts with 'input' and 'expected_output'.

        Returns:
            Dataset ID.
        """
        dataset = self.client.create_dataset(
            dataset_name=dataset_name,
            description="Gold standard parliamentary speeches for evaluation",
        )
        for item in speeches_with_gold:
            self.client.create_example(
                inputs=item["input"],
                outputs=item["expected_output"],
                dataset_id=dataset.id,
            )
        return str(dataset.id)

    def get_tracing_env(self) -> dict[str, str]:
        """Return environment variables needed for LangSmith tracing."""
        env = {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGSMITH_PROJECT": self.project_name,
        }
        api_key = os.environ.get("LANGSMITH_API_KEY")
        if api_key:
            env["LANGSMITH_API_KEY"] = api_key
        return env
