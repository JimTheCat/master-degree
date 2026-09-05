"""Batch processor for large-scale corpus classification with checkpointing."""

import asyncio
import json
import time
from pathlib import Path

from tqdm import tqdm

from src.models.base import ClassifierModel
from src.pipeline.cost_manager import CostTracker
from src.schema.models import Prediction, Speech
from src.tracking.unified_tracker import UnifiedTracker


class CheckpointManager:
    """Manages checkpointing for resumable batch processing."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.processed_ids: set[str] = set()

    def load(self) -> set[str]:
        """Load already-processed speech IDs from existing output file."""
        if not self.output_path.exists():
            return set()

        ids = set()
        with open(self.output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ids.add(data["speech_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

        self.processed_ids = ids
        return ids

    def is_processed(self, speech_id: str) -> bool:
        return speech_id in self.processed_ids

    def mark_processed(self, speech_id: str):
        self.processed_ids.add(speech_id)


class BatchProcessor:
    """Process a large corpus through a classifier with rate-limiting,
    checkpointing, and cost control."""

    def __init__(
        self,
        classifier: ClassifierModel,
        tracker: UnifiedTracker | None = None,
        max_concurrent: int = 20,
        cost_budget_usd: float = 100.0,
        cost_warn_usd: float = 50.0,
    ):
        self.classifier = classifier
        self.tracker = tracker
        self.max_concurrent = max_concurrent
        self.cost_tracker = CostTracker(
            budget_usd=cost_budget_usd,
            warn_at_usd=cost_warn_usd,
            _model_name=classifier.model_name,
        )

    async def process(
        self,
        speeches: list[Speech],
        output_path: str | Path,
    ) -> dict:
        """Process speeches with checkpointing and progress tracking.

        Returns a summary dict with counts and cost.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load checkpoint
        checkpoint = CheckpointManager(output_path)
        already_processed = checkpoint.load()
        remaining = [s for s in speeches if s.speech_id not in already_processed]

        total = len(speeches)
        skipped = total - len(remaining)

        if skipped > 0:
            print(f"Resuming: {skipped} already processed, {len(remaining)} remaining")

        semaphore = asyncio.Semaphore(self.max_concurrent)
        progress = tqdm(total=len(remaining), desc="Classifying", initial=0)
        errors = []
        start_time = time.perf_counter()

        # Open output file in append mode
        output_file = open(output_path, "a", encoding="utf-8")

        async def _classify_one(speech: Speech) -> Prediction | None:
            async with semaphore:
                try:
                    prediction = await self.classifier.classify(speech)

                    # Track cost
                    self.cost_tracker.record(
                        prediction.token_usage.get("prompt_tokens", 0),
                        prediction.token_usage.get("completion_tokens", 0),
                    )

                    # Check budget
                    ok, msg = self.cost_tracker.check_budget()
                    if not ok:
                        raise RuntimeError(msg)
                    if msg:
                        tqdm.write(msg)

                    # Write prediction
                    output_file.write(prediction.model_dump_json() + "\n")
                    checkpoint.mark_processed(speech.speech_id)
                    progress.update(1)

                    return prediction
                except RuntimeError:
                    raise  # Budget exceeded — stop
                except Exception as e:
                    errors.append({"speech_id": speech.speech_id, "error": str(e)})
                    progress.update(1)
                    return None

        # Process in chunks for periodic flushing
        chunk_size = 100
        for i in range(0, len(remaining), chunk_size):
            chunk = remaining[i : i + chunk_size]
            tasks = [_classify_one(s) for s in chunk]

            try:
                await asyncio.gather(*tasks)
            except RuntimeError as e:
                print(f"\nStopped: {e}")
                break

            output_file.flush()

            # Log progress to tracker
            if self.tracker:
                self.tracker.log_metric(
                    "processed_count",
                    float(skipped + i + len(chunk)),
                )
                self.tracker.log_metric(
                    "current_cost_usd",
                    self.cost_tracker.total_cost_usd,
                )

        output_file.close()
        progress.close()

        elapsed = time.perf_counter() - start_time
        processed_count = len(checkpoint.processed_ids)

        return {
            "total": total,
            "processed": processed_count,
            "skipped_checkpoint": skipped,
            "errors": len(errors),
            "error_details": errors[:20],  # First 20 errors
            "elapsed_seconds": elapsed,
            "total_cost_usd": self.cost_tracker.total_cost_usd,
            "total_input_tokens": self.cost_tracker.total_input_tokens,
            "total_output_tokens": self.cost_tracker.total_output_tokens,
        }
