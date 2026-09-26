"""
Multi-channel candidate generation with candidate compression.

Integrates:
1. Deterministic Multi-Key Exact Blocking (weighted evidence ranking)
2. Character n-gram TF-IDF Retrieval
3. Word n-gram TF-IDF Retrieval
4. BM25 Retrieval
5. Dense ANN Retrieval (FAISS + SentenceTransformers)
6. Reciprocal Rank Fusion (RRF) Candidate Compression
"""

import gc
import heapq
import logging
import math
from array import array
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing.schema import Record, CandidatePair
from retrieval.embeddings import EmbeddingGenerator
from retrieval.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


# =============================================================================
# Deterministic Exact Blocking
# =============================================================================

class ExactBlocker:
    """Deterministic exact-match blocking with weighted evidence scoring."""

    def __init__(self, config: dict):
        self.config = config
        self.indexes: Dict[str, Dict[str, List[str]]] = {}
        self.stop_words = {
            'and', 'the', 'of', 'in', 'at', 'by', 'for', 'with', 'pvt', 'ltd', 'inc',
            'corp', 'co', 'llc', 'services', 'service', 'company', 'center', 'centre',
            'group', 'india', 'solutions', 'enterprises', 'restaurant', 'store', 'shop'
        }

    def build_index(self, records: Dict[str, Record]) -> None:
        """Build inverted indexes for candidate records."""
        idx_cp = defaultdict(list)
        idx_pn = defaultdict(list)
        idx_tokens = defaultdict(list)
        idx_cnp = defaultdict(list)
        idx_num_tok = defaultdict(list)
        idx_distinctive = defaultdict(list)

        for rid, rec in records.items():
            if rec.source_id == "S1":
                continue

            country = rec.country
            postal = rec.postal_code
            city = rec.city
            tokens = rec.name_tokens
            compact = rec.compact_name
            nums = rec.numeric_tokens

            if country and postal:
                l = idx_cp[f"{country}|{postal}"]
                if len(l) < 50:
                    l.append(rid)
            if postal and tokens:
                l = idx_pn[f"{postal}|{tokens[0]}"]
                if len(l) < 50:
                    l.append(rid)
            if country and len(tokens) >= 2:
                l = idx_tokens[f"{country}|{tokens[0]}|{tokens[1]}"]
                if len(l) < 50:
                    l.append(rid)
            elif country and len(tokens) == 1:
                l = idx_tokens[f"{country}|{tokens[0]}"]
                if len(l) < 50:
                    l.append(rid)
            if country and compact and len(compact) >= 5:
                l = idx_cnp[f"{country}|{compact[:5]}"]
                if len(l) < 50:
                    l.append(rid)
            if country and nums and tokens:
                l = idx_num_tok[f"{country}|{nums[0]}|{tokens[0]}"]
                if len(l) < 50:
                    l.append(rid)
            if country:
                for t in tokens:
                    if len(t) >= 4 and t not in self.stop_words:
                        l = idx_distinctive[f"{country}|{t}"]
                        if len(l) < 50:
                            l.append(rid)

        self.indexes = {
            "country_postal": idx_cp,
            "postal_name": idx_pn,
            "country_tokens": idx_tokens,
            "country_compact": idx_cnp,
            "country_num_tok": idx_num_tok,
            "country_distinctive": idx_distinctive,
        }
        total = sum(len(v) for idx in self.indexes.values() for v in idx.values())
        logger.info(f"ExactBlocker: built {len(self.indexes)} inverted indexes with {total} entries")

    def query(self, s1_record: Record, max_candidates: int = 50) -> List[Tuple[str, float]]:
        """Find candidates using weighted multi-key exact blocking.
        
        Returns:
            List of (candidate_id, evidence_weight) sorted by descending evidence.
        """
        evidence_scores: Dict[str, float] = defaultdict(float)
        country = s1_record.country
        postal = s1_record.postal_code
        city = s1_record.city
        tokens = s1_record.name_tokens
        compact = s1_record.compact_name
        nums = s1_record.numeric_tokens

        # 1. Country + Postal (weight 3.0)
        if country and postal:
            for cid in self.indexes.get("country_postal", {}).get(f"{country}|{postal}", [])[:25]:
                evidence_scores[cid] += 3.0

        # 2. Country + Tokens (weight 2.5)
        if country and len(tokens) >= 2:
            for cid in self.indexes.get("country_tokens", {}).get(f"{country}|{tokens[0]}|{tokens[1]}", [])[:20]:
                evidence_scores[cid] += 2.5
        elif country and len(tokens) == 1:
            for cid in self.indexes.get("country_tokens", {}).get(f"{country}|{tokens[0]}", [])[:20]:
                evidence_scores[cid] += 2.5

        # 3. Country + Compact prefix (weight 2.0)
        if country and compact and len(compact) >= 5:
            for cid in self.indexes.get("country_compact", {}).get(f"{country}|{compact[:5]}", [])[:20]:
                evidence_scores[cid] += 2.0

        # 4. Country + Numeric address token + Name token (weight 2.5)
        if country and nums and tokens:
            for cid in self.indexes.get("country_num_tok", {}).get(f"{country}|{nums[0]}|{tokens[0]}", [])[:20]:
                evidence_scores[cid] += 2.5

        # 5. Distinctive name token (weight 1.5)
        if country:
            for t in tokens:
                if len(t) >= 4 and t not in self.stop_words:
                    for cid in self.indexes.get("country_distinctive", {}).get(f"{country}|{t}", [])[:15]:
                        evidence_scores[cid] += 1.5

        # 6. Postal + first name token (weight 2.0)
        if postal and tokens:
            for cid in self.indexes.get("postal_name", {}).get(f"{postal}|{tokens[0]}", [])[:15]:
                evidence_scores[cid] += 2.0

        # Deterministic sorting: highest evidence score first, tie-break by ID
        sorted_cands = sorted(evidence_scores.items(), key=lambda x: (-x[1], x[0]))
        return sorted_cands[:max_candidates]


# =============================================================================
# TF-IDF Retrieval
# =============================================================================

class TFIDFRetriever:
    """TF-IDF based sparse retrieval for candidate generation with fast sparse dot products."""

    def __init__(self, config: dict, field: str = "name", analyzer: str = "char_wb"):
        self.config = config
        self.field = field
        self.analyzer = analyzer

        tfidf_cfg = config.get("retrieval", {}).get("tfidf", {})
        if analyzer == "char_wb":
            if field == "address":
                ngram_range = tuple(tfidf_cfg.get("address_char_ngram_range", [3, 4]))
            else:
                ngram_range = tuple(tfidf_cfg.get("name_char_ngram_range", [3, 4]))
        else:
            if field == "address":
                ngram_range = tuple(tfidf_cfg.get("address_word_ngram_range", [1, 2]))
            else:
                ngram_range = tuple(tfidf_cfg.get("name_word_ngram_range", [1, 2]))

        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=tfidf_cfg.get("max_features", 50000),
            dtype=np.float32,
            sublinear_tf=True,
            norm="l2",
        )
        self.matrix = None
        self.id_list: List[str] = []

    def fit(self, records: Dict[str, Record]) -> None:
        """Fit TF-IDF on candidate records with bounded memory streaming."""
        texts = []
        ids = []
        for rid, rec in records.items():
            if rec.source_id == "S1":
                continue
            if self.field == "name":
                texts.append(rec.normalized_name)
            else:
                texts.append(rec.normalized_address)
            ids.append(rid)

        self.id_list = ids
        if not texts:
            return

        # Fit vocabulary on a representative sample (bounded by 100k) to prevent memory spikes
        vocab_sample_size = min(len(texts), 100000)
        self.vectorizer.fit(texts[:vocab_sample_size])

        self.matrix = self.vectorizer.transform(texts)
        del texts
        gc.collect()
        logger.info(f"TFIDFRetriever({self.field}, {self.analyzer}): fitted on {len(ids)} records")

    def query(self, s1_record: Record, top_k: int = 20) -> List[Tuple[str, float, int]]:
        """Query for similar records using fast sparse dot product. Returns (id, score, rank)."""
        if self.matrix is None:
            return []

        text = s1_record.normalized_name if self.field == "name" else s1_record.normalized_address
        if not text.strip():
            return []

        import time
        t0 = time.perf_counter()
        query_vec = self.vectorizer.transform([text])
        # Sparse dot product against L2-normalized candidate matrix (returns 1xN csr_matrix)
        dot_res = query_vec.dot(self.matrix.T)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        tfidf_warn_ms = self.config.get("retrieval", {}).get("tfidf", {}).get("tfidf_query_warn_ms", 100.0)
        if elapsed_ms > tfidf_warn_ms:
            logger.warning(
                f"TFIDFRetriever({self.field}, {self.analyzer}) slow query: "
                f"{elapsed_ms:.1f}ms > {tfidf_warn_ms}ms (nnz={dot_res.nnz}, s1_id={s1_record.record_id})"
            )
        if dot_res.nnz == 0:
            return []

        indices = dot_res.indices
        scores = dot_res.data

        # Filter low-similarity candidates
        pos_mask = scores > 0.05
        if not np.any(pos_mask):
            return []

        indices = indices[pos_mask]
        scores = scores[pos_mask]

        if len(scores) <= top_k:
            top_order = np.argsort(scores)[::-1]
        else:
            top_sub = np.argpartition(scores, -top_k)[-top_k:]
            top_order = top_sub[np.argsort(scores[top_sub])[::-1]]

        results = []
        for rank, idx in enumerate(top_order, 1):
            results.append((self.id_list[indices[idx]], float(scores[idx]), rank))

        return results


# =============================================================================
# BM25 Retrieval
# =============================================================================

class BM25Retriever:
    """True inverted-index BM25 retrieval combining name and address tokens."""

    def __init__(self, config: dict):
        self.config = config
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_lens: np.ndarray = np.array([], dtype=np.float32)
        self.avgdl: float = 1.0
        self.idf: Dict[str, float] = {}
        self.id_list: List[str] = []
        bm25_cfg = self.config.get("retrieval", {}).get("bm25", {})
        self.k1: float = float(bm25_cfg.get("k1", 1.5))
        self.b: float = float(bm25_cfg.get("b", 0.75))

    def fit(self, records: Dict[str, Record]) -> None:
        """Build true token inverted index from candidate records using compact 32-bit arrays."""
        self.inverted_index = defaultdict(lambda: array('i'))
        self.id_list = []
        doc_lens = []

        doc_idx = 0
        for rid, rec in records.items():
            if rec.source_id == "S1":
                continue
            tokens = rec.name_tokens + rec.address_tokens
            if not tokens:
                continue
            self.id_list.append(rid)
            doc_lens.append(len(tokens))

            # Count term frequencies in this document
            tf_map = Counter(tokens)
            for tok, count in tf_map.items():
                self.inverted_index[tok].append((doc_idx << 6) | min(count, 63))

            doc_idx += 1

        N = len(self.id_list)
        if N == 0:
            logger.warning("BM25Retriever: empty corpus, disabled")
            return

        self.doc_lens = np.array(doc_lens, dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens))

        # Compute Robertson-Spärck Jones / Okapi IDF with +1 smoothing for non-negativity
        # IDF(q) = ln(1 + (N - df + 0.5) / (df + 0.5))
        self.idf = {}
        max_df = int(0.40 * N)
        pruned_index = {}
        for tok, postings in self.inverted_index.items():
            df = len(postings)
            if df > max_df and N > 100:
                continue
            idf_val = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
            if idf_val > 0.05:
                self.idf[tok] = idf_val
                pruned_index[tok] = postings

        self.inverted_index = pruned_index
        logger.info(f"BM25Retriever: inverted index ready on {N} documents, {len(self.idf)} unique terms (avgdl={self.avgdl:.1f})")

    def query(self, s1_record: Record, top_k: int = 20) -> List[Tuple[str, float, int]]:
        """Query BM25 index via sparse score accumulation. Returns (id, score, rank)."""
        if not self.id_list or not self.idf:
            return []

        query_tokens = s1_record.name_tokens + s1_record.address_tokens
        if not query_tokens:
            return []

        doc_scores: Dict[int, float] = defaultdict(float)
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl
        doc_lens = self.doc_lens
        idf_dict = self.idf
        inv_idx = self.inverted_index

        for q in set(query_tokens):
            if q not in idf_dict:
                continue
            idf_q = idf_dict[q]
            postings = inv_idx[q]
            for packed in postings:
                doc_idx = packed >> 6
                tf = packed & 63
                dl = doc_lens[doc_idx]
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                doc_scores[doc_idx] += idf_q * (tf * (k1 + 1.0) / denom)

        if not doc_scores:
            return []

        if len(doc_scores) <= top_k:
            top_items = sorted(doc_scores.items(), key=lambda x: -x[1])
        else:
            top_items = heapq.nlargest(top_k, doc_scores.items(), key=lambda x: x[1])

        results = []
        for rank, (doc_idx, score) in enumerate(top_items, 1):
            if score > 0.0:
                results.append((self.id_list[doc_idx], float(score), rank))

        return results


# =============================================================================
# Candidate Generator (Orchestrator)
# =============================================================================

class CandidateGenerator:
    """Multi-channel candidate generation with RRF compression.
    
    Channels:
    - Exact Blocking (Deterministic Weighted Inverted Index)
    - Character TF-IDF
    - Word TF-IDF
    - BM25
    - Dense Semantic ANN (FAISS)
    """

    def __init__(self, config: dict, embedding_gen: Optional[EmbeddingGenerator] = None):
        self.config = config
        self.embedding_gen = embedding_gen
        self.exact_blocker = ExactBlocker(config)
        self.name_char_tfidf = TFIDFRetriever(config, field="name", analyzer="char_wb")
        self.name_word_tfidf = TFIDFRetriever(config, field="name", analyzer="word")
        self.address_char_tfidf = TFIDFRetriever(config, field="address", analyzer="word")
        self.bm25 = BM25Retriever(config)

        dense_cfg = config.get("retrieval", {}).get("dense_ann", {})
        nlist = dense_cfg.get("nlist", 4096)
        nprobe = dense_cfg.get("nprobe", 16)
        dim = dense_cfg.get("dimension", 384)
        index_type = dense_cfg.get("index_type", "ivfpq")
        pq_m = dense_cfg.get("pq_m", 32)
        pq_nbits = dense_cfg.get("pq_nbits", 8)
        training_sample_size = dense_cfg.get("training_sample_size", 100000)

        self.name_ann_index = FAISSIndex(
            dimension=dim, index_type=index_type, nlist=nlist, nprobe=nprobe,
            pq_m=pq_m, pq_nbits=pq_nbits, training_sample_size=training_sample_size
        )
        self.address_ann_index = FAISSIndex(
            dimension=dim, index_type=index_type, nlist=nlist, nprobe=nprobe,
            pq_m=pq_m, pq_nbits=pq_nbits, training_sample_size=training_sample_size
        )
        self.record_ann_index = FAISSIndex(
            dimension=dim, index_type=index_type, nlist=nlist, nprobe=nprobe,
            pq_m=pq_m, pq_nbits=pq_nbits, training_sample_size=training_sample_size
        )

        self.filter_country_mismatch = config.get("retrieval", {}).get("filter_country_mismatch", True)
        self.country_filter_hits: int = 0
        self.all_candidate_records: Dict[str, Record] = {}

    def _build_dense_index_batched(self, faiss_idx: FAISSIndex, records: Dict[str, Record], field: str, batch_size: int = 4096) -> None:
        """Stream record embeddings in compact batches directly into FAISS to prevent memory spikes."""
        c_ids = list(records.keys())
        if not c_ids:
            return

        # 1. Train on representative sample (bounded by training_sample_size)
        dense_cfg = self.config.get("retrieval", {}).get("dense_ann", {})
        max_train_samples = dense_cfg.get("training_sample_size", 100000)
        init_sample_size = min(len(c_ids), max_train_samples)
        sample_ids = c_ids[:init_sample_size]
        sample_recs = {rid: records[rid] for rid in sample_ids}
        _, sample_embs = self.embedding_gen.encode_record_field(sample_recs, field=field)
        faiss_idx.train_and_init(sample_embs, total_estimated=len(c_ids))
        faiss_idx.add_batch(sample_embs, sample_ids)
        del sample_embs, sample_recs

        # 2. Add remaining in streaming batches
        for b_start in range(init_sample_size, len(c_ids), batch_size):
            b_ids = c_ids[b_start:b_start + batch_size]
            b_recs = {rid: records[rid] for rid in b_ids}
            _, b_embs = self.embedding_gen.encode_record_field(b_recs, field=field)
            faiss_idx.add_batch(b_embs, b_ids)
            del b_embs, b_recs

    def setup(self, s1_records: Dict[str, Record],
              s2_records: Optional[Dict[str, Record]] = None,
              s3_records: Optional[Dict[str, Record]] = None,
              candidate_records: Optional[Dict[str, Record]] = None,
              build_dense_index: bool = True) -> None:
        """Initialize all retrieval channels across candidate records."""
        if candidate_records is not None:
            self.all_candidate_records = candidate_records
        else:
            self.all_candidate_records = {**(s2_records or {}), **(s3_records or {})}

        logger.info("Setting up multi-channel retrieval...")
        self.exact_blocker.build_index(self.all_candidate_records)
        self.name_char_tfidf.fit(self.all_candidate_records)
        self.name_word_tfidf.fit(self.all_candidate_records)
        self.address_char_tfidf.fit(self.all_candidate_records)
        self.bm25.fit(self.all_candidate_records)

        # Build dense ANN index in streaming batches (strictly bounded memory)
        dense_cfg = self.config.get("retrieval", {}).get("dense_ann", {})
        if build_dense_index and self.embedding_gen and dense_cfg.get("enabled", True):
            batch_size = dense_cfg.get("batch_size", 4096)
            logger.info("Building dense FAISS ANN indexes in streaming batches (name, address, record)...")
            self._build_dense_index_batched(self.name_ann_index, self.all_candidate_records, field="name", batch_size=batch_size)
            self._build_dense_index_batched(self.address_ann_index, self.all_candidate_records, field="address", batch_size=batch_size)
            self._build_dense_index_batched(self.record_ann_index, self.all_candidate_records, field="record", batch_size=batch_size)
            logger.info(f"FAISS ANN ready with {len(self.all_candidate_records)} candidate vectors across 3 indexes")

        logger.info("All retrieval channels ready")

    def generate_candidates(self, s1_record: Record,
                            s1_name_emb: Optional[np.ndarray] = None,
                            s1_addr_emb: Optional[np.ndarray] = None,
                            s1_rec_emb: Optional[np.ndarray] = None,
                            return_raw: bool = False):
        """Generate candidates for a single S1 entity through all channels."""
        s1_id = s1_record.record_id
        candidate_dict: Dict[str, CandidatePair] = {}

        def _get_or_create(cand_id: str) -> Optional[CandidatePair]:
            if self.filter_country_mismatch and s1_record.country:
                cand_rec = self.all_candidate_records.get(cand_id)
                if cand_rec and cand_rec.country and cand_rec.country != s1_record.country:
                    self.country_filter_hits += 1
                    return None

            if cand_id not in candidate_dict:
                source = "S2" if cand_id.startswith("S2") else "S3"
                candidate_dict[cand_id] = CandidatePair(
                    s1_id=s1_id, candidate_id=cand_id, candidate_source=source
                )
            return candidate_dict[cand_id]

        # 1. Deterministic Multi-Key Exact Blocking
        exact_results = self.exact_blocker.query(s1_record, max_candidates=40)
        for rank, (cand_id, weight) in enumerate(exact_results, 1):
            pair = _get_or_create(cand_id)
            if pair is not None:
                pair.found_by_exact = True

        # 2. Character TF-IDF Retrieval (Name)
        for cand_id, score, rank in self.name_char_tfidf.query(s1_record, top_k=20):
            pair = _get_or_create(cand_id)
            if pair is not None:
                pair.found_by_char_tfidf = True
                pair.char_tfidf_rank = rank
                pair.char_tfidf_score = score

        # 3. Word TF-IDF Retrieval (Name)
        for cand_id, score, rank in self.name_word_tfidf.query(s1_record, top_k=15):
            pair = _get_or_create(cand_id)
            if pair is not None:
                pair.found_by_word_tfidf = True
                pair.word_tfidf_rank = rank
                pair.word_tfidf_score = score

        # 4. Address Character TF-IDF Retrieval
        for cand_id, score, rank in self.address_char_tfidf.query(s1_record, top_k=15):
            pair = _get_or_create(cand_id)
            if pair is not None:
                pair.found_by_address_tfidf = True
                pair.address_tfidf_rank = rank
                pair.address_tfidf_score = score

        # 5. BM25 Retrieval (ACTIVE)
        bm25_top_k = self.config.get("retrieval", {}).get("bm25", {}).get("top_k", 20)
        for cand_id, score, rank in self.bm25.query(s1_record, top_k=bm25_top_k):
            pair = _get_or_create(cand_id)
            if pair is not None:
                pair.found_by_bm25 = True
                pair.bm25_rank = rank
                pair.bm25_score = score

        # 6. Dense ANN Retrieval (ACTIVE)
        if s1_name_emb is not None and s1_record.normalized_name and self.name_ann_index.index is not None:
            for cand_id, sim, rank in self.name_ann_index.query_single(s1_name_emb, top_k=20):
                pair = _get_or_create(cand_id)
                if pair is not None:
                    pair.found_by_name_ann = True
                    pair.name_ann_rank = rank
                    pair.name_ann_score = sim

        if s1_addr_emb is not None and s1_record.normalized_address and self.address_ann_index.index is not None:
            for cand_id, sim, rank in self.address_ann_index.query_single(s1_addr_emb, top_k=15):
                pair = _get_or_create(cand_id)
                if pair is not None:
                    pair.found_by_address_ann = True
                    pair.address_ann_rank = rank
                    pair.address_ann_score = sim

        if s1_rec_emb is not None and s1_record.record_text and self.record_ann_index.index is not None:
            for cand_id, sim, rank in self.record_ann_index.query_single(s1_rec_emb, top_k=20):
                pair = _get_or_create(cand_id)
                if pair is not None:
                    pair.found_by_record_ann = True
                    pair.record_ann_rank = rank
                    pair.record_ann_score = sim

        # Compute retrieval votes
        for pair in candidate_dict.values():
            pair.retrieval_votes = sum([
                pair.found_by_exact,
                pair.found_by_char_tfidf,
                pair.found_by_word_tfidf,
                pair.found_by_address_tfidf,
                pair.found_by_bm25,
                pair.found_by_name_ann,
                pair.found_by_address_ann,
                pair.found_by_record_ann,
            ])

        # Candidate Compression (RRF + top-K)
        raw_candidates = list(candidate_dict.values())
        compressed_candidates = self.compress_candidates(raw_candidates)
        if return_raw:
            return compressed_candidates, raw_candidates
        return compressed_candidates

    def compress_candidates(self, candidates: List[CandidatePair]) -> List[CandidatePair]:
        """Apply candidate compression: RRF scoring, source quotas, deterministic top-K."""
        if not candidates:
            return []

        comp_cfg = self.config.get("candidate_compression", {})
        max_k = comp_cfg.get("max_candidates_per_entity", 40)
        s2_max = comp_cfg.get("s2_max_candidates", 20)
        s3_max = comp_cfg.get("s3_max_candidates", 20)
        rrf_k = comp_cfg.get("rrf", {}).get("k", 60)

        # Compute Reciprocal Rank Fusion (RRF) score
        for pair in candidates:
            score = 0.0
            if pair.found_by_exact:
                score += 2.0 / rrf_k  # Exact blocking gets highest prior

            for rank in [
                pair.char_tfidf_rank,
                pair.word_tfidf_rank,
                pair.address_tfidf_rank,
                pair.bm25_rank,
                pair.name_ann_rank,
                pair.address_ann_rank,
                pair.record_ann_rank,
            ]:
                if rank > 0:
                    score += 1.0 / (rrf_k + rank)

            # Boost multi-channel matches
            if pair.retrieval_votes >= 2:
                score *= 1.25

            pair.rrf_score = score

        # Source-aware candidate quotas: ensure S2 does not crowd out S3
        s2_cands = [p for p in candidates if p.candidate_source == "S2"]
        s3_cands = [p for p in candidates if p.candidate_source == "S3"]
        other_cands = [p for p in candidates if p.candidate_source not in ("S2", "S3")]

        s2_cands.sort(key=lambda p: (-p.rrf_score, p.candidate_id))
        s3_cands.sort(key=lambda p: (-p.rrf_score, p.candidate_id))
        other_cands.sort(key=lambda p: (-p.rrf_score, p.candidate_id))

        selected = s2_cands[:s2_max] + s3_cands[:s3_max] + other_cands
        selected.sort(key=lambda p: (-p.rrf_score, p.candidate_id))
        return selected[:max_k]

    def iter_candidate_batches(self, s1_records: Dict[str, Record],
                               batch_size: int = 2048,
                               return_raw: bool = False):
        """Yield candidate batches for S1 entities with bounded memory batching.
        
        Releases dense embedding tensors after every batch.
        
        Yields:
            batch_candidates: Dict[str, List[CandidatePair]] (or (batch_candidates, batch_raw) if return_raw=True)
        """
        s1_ids = list(s1_records.keys())
        has_ann = (self.embedding_gen is not None and self.name_ann_index.index is not None)

        for b_start in range(0, len(s1_ids), batch_size):
            b_ids = s1_ids[b_start:b_start + batch_size]
            b_records = {rid: s1_records[rid] for rid in b_ids}

            batch_embs = {}
            if has_ann:
                _, b_name = self.embedding_gen.encode_record_field(b_records, field="name")
                _, b_addr = self.embedding_gen.encode_record_field(b_records, field="address")
                _, b_rec = self.embedding_gen.encode_record_field(b_records, field="record")
                for i, rid in enumerate(b_ids):
                    batch_embs[rid] = {
                        "name_emb": b_name[i],
                        "address_emb": b_addr[i],
                        "record_emb": b_rec[i]
                    }

            batch_candidates = {}
            batch_raw = {} if return_raw else None

            for s1_id in b_ids:
                s1_rec = s1_records[s1_id]
                e_dict = batch_embs.get(s1_id, {})
                n_emb = e_dict.get("name_emb")
                a_emb = e_dict.get("address_emb")
                r_emb = e_dict.get("record_emb")

                if return_raw:
                    cands, raw = self.generate_candidates(
                        s1_rec, s1_name_emb=n_emb, s1_addr_emb=a_emb, s1_rec_emb=r_emb, return_raw=True
                    )
                    batch_candidates[s1_id] = cands
                    batch_raw[s1_id] = raw
                else:
                    batch_candidates[s1_id] = self.generate_candidates(
                        s1_rec, s1_name_emb=n_emb, s1_addr_emb=a_emb, s1_rec_emb=r_emb, return_raw=False
                    )

            del batch_embs
            if return_raw:
                yield batch_candidates, batch_raw
            else:
                yield batch_candidates

    def generate_all_candidates(self, s1_records: Dict[str, Record],
                                 precomputed_s1_embs: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                                 dense_batch_size: int = 2048,
                                 batch_log_every: int = 10000,
                                 return_raw: bool = False):
        """Generate candidates for all S1 entities with bounded memory batching."""
        all_candidates = {}
        all_raw = {} if return_raw else None
        s1_ids = list(s1_records.keys())

        batch_gen = self.iter_candidate_batches(s1_records, batch_size=dense_batch_size, return_raw=return_raw)
        processed = 0
        for batch_item in batch_gen:
            if return_raw:
                b_cands, b_raw = batch_item
                all_candidates.update(b_cands)
                all_raw.update(b_raw)
                processed += len(b_cands)
            else:
                b_cands = batch_item
                all_candidates.update(b_cands)
                processed += len(b_cands)

            if processed % batch_log_every < dense_batch_size or processed == len(s1_ids):
                logger.info(f"Generated candidates for {processed:,}/{len(s1_ids):,} entities...")

        total_pairs = sum(len(v) for v in all_candidates.values())
        avg_per_entity = total_pairs / max(len(all_candidates), 1)
        logger.info(f"Candidate generation done: {total_pairs:,} total pairs (avg {avg_per_entity:.1f} per S1)")
        if self.country_filter_hits > 0:
            logger.info(f"Country filter audit: prevented {self.country_filter_hits:,} cross-country candidate pairs")

        if return_raw:
            return all_candidates, all_raw
        return all_candidates

    @staticmethod
    def save_candidate_pairs(all_candidates: Dict[str, List[CandidatePair]],
                              s1_ids: List[str], output_path: str) -> None:
        """Write candidate_pairs.tsv."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in s1_ids:
                pairs = all_candidates.get(s1_id, [])
                seen = set()
                unique = [p.candidate_id for p in pairs if not (p.candidate_id in seen or seen.add(p.candidate_id))]
                f.write(f"{s1_id}\t{','.join(unique)}\n")

        logger.info(f"Saved candidate_pairs.tsv to {output_path}")
