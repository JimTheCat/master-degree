from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

from hate_speech.services.methods.base import MethodInterface


class NeuralBert(MethodInterface):
    def __init__(self, model_name: str = 'allegro/herbert-base-cased', num_labels: int = 2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def fit(self, X_train: List[str], y_train: List[int], params: dict = None):
        # Minimal wrapper to fine-tune - expects small datasets for demonstration
        enc_train = self.tokenizer(X_train, truncation=True, padding=True)

        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

        ds = SimpleDataset(enc_train, y_train)
        training_args = TrainingArguments(output_dir='./tmp_bert', num_train_epochs=(params or {}).get('epochs', 1),
                                          per_device_train_batch_size=(params or {}).get('batch_size', 8))
        trainer = Trainer(model=self.model, args=training_args, train_dataset=ds)
        trainer.train()
        return self

    def predict(self, X: List[str]) -> List[int]:
        enc = self.tokenizer(X, truncation=True, padding=True, return_tensors='pt')
        self.model.eval()
        with torch.no_grad():
            out = self.model(**{k: v for k, v in enc.items()})
            logits = out.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        return preds


class NeuralLSTM(MethodInterface):
    # Placeholder: for real use add embeddings (torchtext or pretrained) and proper training loop
    def __init__(self, vocab=None):
        self.vocab = vocab

    def fit(self, X_train, y_train, params=None):
        return self

    def predict(self, X):
        return [0] * len(X)
