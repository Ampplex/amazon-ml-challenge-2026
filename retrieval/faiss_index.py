"""
FAISS Approximate Nearest Neighbor (ANN) index manager.

Uses Inner Product index (equivalent to cosine similarity on unit-normalized vectors)
for sub-millisecond candidate retrieval. Supports IndexIVFPQ for massive 48x vector
compression (384 float32 -> 32 bytes) with dynamic centroid fallbacks and IndexFlatIP
for smaller partitions.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS-based vector index for semantic ANN candidate retrieval using IndexIVFPQ."""

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "ivfpq",
        nlist: int = 4096,
        nprobe: int = 16,
        pq_m: int = 32,
        pq_nbits: int = 8,
        training_sample_size: int = 100000,
    ):
        self.dimension = dimension
        self.index_type = (index_type or "ivfpq").lower()
        self.nlist = nlist
        self.nprobe = nprobe
        self.pq_m = pq_m
        self.pq_nbits = pq_nbits
        self.training_sample_size = training_sample_size
        self.index = None
        self.id_map: List[str] = []

    def train_and_init(self, sample_embeddings: np.ndarray, total_estimated: int = 100000) -> None:
        """Initialize and train FAISS index on a representative sample of vectors."""
        if len(sample_embeddings) == 0:
            return
        try:
            import faiss
            embs_f32 = sample_embeddings.astype(np.float32)
            n_samples = max(len(embs_f32), total_estimated)

            # Subsample if sample_embeddings is larger than training_sample_size
            if len(embs_f32) > self.training_sample_size:
                rng = np.random.RandomState(42)
                sub_indices = rng.choice(len(embs_f32), size=self.training_sample_size, replace=False)
                train_vecs = embs_f32[sub_indices]
            else:
                train_vecs = embs_f32

            min_ivfpq_samples = max(256, 1 << self.pq_nbits)
            if self.index_type == "ivfpq" and len(train_vecs) >= min_ivfpq_samples and n_samples >= 256:
                # Dynamic nlist calculation: min(config_nlist, max(32, int(sqrt(N))))
                effective_nlist = min(self.nlist, max(32, int(np.sqrt(n_samples))))
                # Ensure training vectors >= effective_nlist
                if len(train_vecs) < effective_nlist:
                    effective_nlist = max(4, len(train_vecs) // 4)

                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFPQ(
                    quantizer,
                    self.dimension,
                    effective_nlist,
                    self.pq_m,
                    self.pq_nbits,
                    faiss.METRIC_INNER_PRODUCT,
                )
                self.index.train(train_vecs)
                self.index.nprobe = min(self.nprobe, effective_nlist)
                logger.info(
                    f"FAISSIndex: IndexIVFPQ trained on {len(train_vecs)} samples "
                    f"(nlist={effective_nlist}, nprobe={self.index.nprobe}, pq_m={self.pq_m}, nbits={self.pq_nbits})"
                )
            elif self.index_type == "ivfflat" and len(train_vecs) >= 100 and n_samples >= 100:
                effective_nlist = min(self.nlist, max(4, int(np.sqrt(n_samples))))
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, effective_nlist, faiss.METRIC_INNER_PRODUCT
                )
                self.index.train(train_vecs)
                self.index.nprobe = min(self.nprobe, effective_nlist)
                logger.info(
                    f"FAISSIndex: IndexIVFFlat trained on {len(train_vecs)} samples "
                    f"(nlist={effective_nlist}, nprobe={self.index.nprobe})"
                )
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info(f"FAISSIndex: IndexFlatIP initialized for small corpus (n_samples={n_samples})")

            self.id_map = []
        except ImportError:
            logger.warning("faiss-cpu not installed. FAISS retrieval disabled.")
            self.index = None

    def add_batch(self, batch_embeddings: np.ndarray, batch_ids: List[str]) -> None:
        """Incrementally add a batch of vectors to the FAISS index."""
        if self.index is None or len(batch_ids) == 0:
            return
        embs_f32 = batch_embeddings.astype(np.float32)
        self.index.add(embs_f32)
        self.id_map.extend(batch_ids)

    def build_index(self, embeddings: np.ndarray, entity_ids: List[str]) -> None:
        """Build FAISS Approximate Nearest Neighbor (ANN) index from normalized embeddings.
        
        Args:
            embeddings: np.ndarray of shape (N, dimension), float32.
            entity_ids: List of N entity IDs corresponding to matrix rows.
        """
        if len(entity_ids) == 0:
            logger.warning("Empty embeddings passed to FAISSIndex. Index not built.")
            return

        self.train_and_init(embeddings, total_estimated=len(entity_ids))
        if self.index is not None:
            self.add_batch(embeddings, entity_ids)
            logger.info(f"FAISSIndex: built index with {self.index.ntotal} total vectors")

    def query(self, query_embeddings: np.ndarray, top_k: int = 30) -> List[List[Tuple[str, float, int]]]:
        """Batch query the FAISS index.
        
        Args:
            query_embeddings: np.ndarray of shape (B, dimension), float32.
            top_k: Number of nearest neighbors to retrieve.
            
        Returns:
            List of length B, where each element is a list of (candidate_id, similarity, rank).
        """
        if self.index is None or self.index.ntotal == 0 or len(query_embeddings) == 0:
            return [[] for _ in range(len(query_embeddings))]

        actual_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embeddings.astype(np.float32), actual_k)

        results = []
        for row_dist, row_idx in zip(distances, indices):
            row_results = []
            for rank, (score, idx) in enumerate(zip(row_dist, row_idx), 1):
                if 0 <= idx < len(self.id_map):
                    row_results.append((self.id_map[idx], float(score), rank))
            results.append(row_results)

        return results

    def query_single(self, query_vector: np.ndarray, top_k: int = 30) -> List[Tuple[str, float, int]]:
        """Query single vector. Returns list of (candidate_id, similarity, rank)."""
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        batch_results = self.query(query_vector, top_k=top_k)
        return batch_results[0] if batch_results else []
