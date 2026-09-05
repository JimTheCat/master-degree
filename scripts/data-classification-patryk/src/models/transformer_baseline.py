import asyncio
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.models.base import ClassifierModel
from src.schema.labels import ALL_LABELS
from src.schema.models import AnnotatedSpeech, Prediction, Speech


class SpeechDataset(TorchDataset):
    """PyTorch dataset for multi-label speech classification."""

    def __init__(
        self,
        speeches: list[AnnotatedSpeech],
        tokenizer,
        max_length: int = 512,
    ):
        self.speeches = speeches
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_idx = {label: i for i, label in enumerate(ALL_LABELS)}

    def __len__(self):
        return len(self.speeches)

    def __getitem__(self, idx):
        speech = self.speeches[idx]
        encoding = self.tokenizer(
            speech.speech.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Flatten from (1, seq_len) to (seq_len,)
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        # Build multi-label target
        labels = torch.zeros(len(ALL_LABELS), dtype=torch.float)
        if speech.gold:
            for label in speech.gold.labels:
                if label in self.label_to_idx:
                    labels[self.label_to_idx[label]] = 1.0
        item["labels"] = labels
        return item


class TransformerBaseline(ClassifierModel):
    """Fine-tuned transformer model for multi-label classification."""

    def __init__(
        self,
        model_name: str = "allegro/herbert-base-cased",
        max_length: int = 512,
        device: str | None = None,
    ):
        self._model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(ALL_LABELS),
            problem_type="multi_label_classification",
        )
        self.model.to(self.device)
        self.threshold = 0.5

    @property
    def model_name(self) -> str:
        return self._model_name

    def train(
        self,
        train_data: list[AnnotatedSpeech],
        val_data: list[AnnotatedSpeech],
        output_dir: str = "models/baseline",
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        early_stopping_patience: int = 3,
    ) -> dict:
        """Fine-tune the transformer model."""
        train_dataset = SpeechDataset(train_data, self.tokenizer, self.max_length)
        val_dataset = SpeechDataset(val_data, self.tokenizer, self.max_length)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=10,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )

        result = trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return {
            "train_loss": result.training_loss,
            "epochs_trained": result.global_step,
        }

    def load_checkpoint(self, checkpoint_dir: str):
        """Load a trained model from checkpoint."""
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model.to(self.device)

    def set_thresholds(self, thresholds: dict[str, float]):
        """Set per-label classification thresholds."""
        self._per_label_thresholds = thresholds

    def _predict_batch_sync(self, speeches: list[Speech]) -> list[Prediction]:
        """Synchronous batch prediction."""
        self.model.eval()
        predictions = []

        for speech in speeches:
            start = time.perf_counter()
            encoding = self.tokenizer(
                speech.text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits.squeeze(0)
                probs = torch.sigmoid(logits).cpu().numpy()

            latency = (time.perf_counter() - start) * 1000

            # Apply thresholds
            thresholds = getattr(self, "_per_label_thresholds", None)
            active_labels = []
            scores = {}
            for i, label_name in enumerate(ALL_LABELS):
                threshold = (
                    thresholds[label_name]
                    if thresholds and label_name in thresholds
                    else self.threshold
                )
                scores[label_name] = float(probs[i])
                if probs[i] >= threshold:
                    active_labels.append(label_name)

            predictions.append(Prediction(
                speech_id=speech.speech_id,
                labels=active_labels,
                label_scores=scores,
                model_name=self._model_name,
                latency_ms=latency,
            ))

        return predictions

    async def classify(self, speech: Speech) -> Prediction:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._predict_batch_sync, [speech])
        return results[0]

    async def classify_batch(self, speeches: list[Speech]) -> list[Prediction]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_batch_sync, speeches)
