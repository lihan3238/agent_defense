from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


class OpenAITextEmbeddingBackend:
    """Paper-compatible embedding backend using ``text-embedding-3-large``."""

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        *,
        client: Any | None = None,
    ) -> None:
        self.model = model
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ImportError(
                    "The OpenAI embedding backend requires the optional 'melon-openai' dependencies"
                ) from error
            client = OpenAI()
        self.client = client
        self.request_count = 0

    def embed(self, texts: Sequence[str]) -> NDArray[np.float64]:
        if not texts:
            raise ValueError("At least one text is required for embedding")
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        self.request_count += 1
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts):
            raise ValueError("Embedding service returned an unexpected number of vectors")
        return np.asarray([item.embedding for item in ordered], dtype=np.float64)


class TransformersMeanPoolingEmbedder:
    """Local semantic fallback built on a Hugging Face encoder model."""

    def __init__(
        self,
        model_id_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        revision: str | None = None,
        device: str = "cpu",
        local_files_only: bool = False,
        max_length: int = 256,
    ) -> None:
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.model_id_or_path = model_id_or_path
        self.revision = revision
        self.device = device
        self.local_files_only = local_files_only
        self.max_length = max_length
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self.request_count = 0

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "The local semantic embedding backend requires the optional 'hf' dependencies"
            ) from error

        resolved_device = self.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id_or_path,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        model = AutoModel.from_pretrained(
            self.model_id_or_path,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        model.to(resolved_device)
        model.eval()
        self.device = resolved_device
        self._tokenizer = tokenizer
        self._model = model

    def embed(self, texts: Sequence[str]) -> NDArray[np.float64]:
        if not texts:
            raise ValueError("At least one text is required for embedding")
        self.ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        import torch

        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.inference_mode():
            output = self._model(**encoded)
        hidden = output.last_hidden_state
        attention = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp_min(1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        self.request_count += 1
        return pooled.detach().to(device="cpu", dtype=torch.float64).numpy()
