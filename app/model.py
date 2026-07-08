"""Carga e inferencia del modelo de resumen facebook/bart-large-cnn."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/bart-large-cnn"
MODEL_VERSION = "1.0.0"
MAX_INPUT_TOKENS = 1024


def _pick_device() -> str:
    """Devuelve mps en Apple Silicon, cuda si hay GPU NVIDIA, o cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class SummarizationModel:
    """Tokenizer + modelo residentes en memoria con inferencia de resumen."""

    model_name: str = MODEL_NAME
    model_version: str = MODEL_VERSION
    device: str = field(default_factory=_pick_device)
    loaded: bool = False
    _tokenizer: AutoTokenizer | None = None
    _model: AutoModelForSeq2SeqLM | None = None

    def load(self) -> None:
        """Descarga (si hace falta) y carga el modelo en el dispositivo."""
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self.loaded = True

    def summarize(self, text: str, min_length: int = 30, max_length: int = 130) -> dict:
        """Resume el texto y devuelve resumen, tiempo (ms) y conteo de tokens."""
        if not self.loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("El modelo no está cargado.")

        start = time.perf_counter()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(self.device)
        input_tokens = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            summary_ids = self._model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                min_length=min_length,
                max_length=max_length,
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "summary": summary,
            "inference_time_ms": round(elapsed_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": int(summary_ids.shape[1]),
        }
