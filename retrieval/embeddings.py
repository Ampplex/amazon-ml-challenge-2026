"""
Dense embedding generation module using SentenceTransformers.

Generates normalized vector representations for business names, addresses,
and composite full records to power ANN search and semantic feature extraction.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from preprocessing.schema import Record

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates dense semantic embeddings for records."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load pretrained SentenceTransformer model. Fails fast if unavailable."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Startup sanity validation
            test_vec = self.model.encode(["startup check"], normalize_embeddings=True, convert_to_numpy=True)
            if test_vec.shape[1] != 384 or not np.isfinite(test_vec).all():
                raise ValueError(f"Unexpected embedding shape {test_vec.shape} or non-finite values")
            logger.info(f"Dense retrieval: ENABLED | Model: {self.model_name} | Dim: {test_vec.shape[1]} | Device: {self.device} | Status: VALIDATED")
        except Exception as e:
            logger.critical(f"FATAL: Failed to load SentenceTransformer model '{self.model_name}': {e}")
            raise RuntimeError(
                f"Dense embedding model '{self.model_name}' could not be loaded ({e}). "
                "Dense retrieval cannot proceed with uninitialized embeddings. "
                "Ensure network connectivity or weights are cached, or disable dense_ann in config."
            ) from e

    def encode_texts(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """Encode a list of text strings into L2-normalized dense embeddings.
        
        Args:
            texts: List of text strings.
            batch_size: Encoding batch size.
            
        Returns:
            np.ndarray of shape (len(texts), embedding_dim), normalized to unit length.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        if self.model is None:
            raise RuntimeError(f"EmbeddingGenerator({self.model_name}) is not initialized. Cannot encode texts.")

        # Handle empty/blank texts by substituting placeholder
        clean_texts = [t.strip() if (t and t.strip()) else "<empty>" for t in texts]

        embeddings = self.model.encode(
            clean_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def encode_record_field(self, records: Dict[str, Record], field: str = "name", batch_size: int = 256) -> Tuple[List[str], np.ndarray]:
        """Encode a specific field across a record dictionary.
        
        Args:
            records: Dict of entity_id -> Record.
            field: "name", "address", or "record".
            batch_size: Batch size for encoding.
            
        Returns:
            (entity_ids, embeddings_matrix)
        """
        entity_ids = list(records.keys())
        texts = []

        for eid in entity_ids:
            rec = records[eid]
            if field == "name":
                texts.append(rec.normalized_name)
            elif field == "address":
                texts.append(rec.normalized_address)
            else:
                texts.append(rec.record_text or f"{rec.normalized_name} | {rec.normalized_address} | {rec.country}")

        embeddings = self.encode_texts(texts, batch_size=batch_size)
        return entity_ids, embeddings
