"""
Feature engineering: lexical, semantic, structural, rarity, and retrieval features.
Combined into a single module for efficiency.
"""

import logging
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from preprocessing.schema import Record, CandidatePair, PairFeatures
from preprocessing.normalize_names import extract_normalized_name_tokens
from preprocessing.normalize_addresses import extract_normalized_address_tokens

logger = logging.getLogger(__name__)


# =============================================================================
# String Similarity Utilities
# =============================================================================

def _safe_ratio(a: float, b: float) -> float:
    """Safe division returning 0.0 on zero denominator."""
    return a / b if b > 0 else 0.0


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return _safe_ratio(intersection, union)


def token_overlap_ratio(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Overlap ratio: |intersection| / |smaller set|."""
    if not tokens_a or not tokens_b:
        return 0.0
    sa, sb = set(tokens_a), set(tokens_b)
    return _safe_ratio(len(sa & sb), min(len(sa), len(sb)))


def token_containment(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Containment: |intersection| / |tokens_a|."""
    if not tokens_a:
        return 0.0
    sa, sb = set(tokens_a), set(tokens_b)
    return _safe_ratio(len(sa & sb), len(sa))


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (1 - normalized_distance)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    try:
        from rapidfuzz.distance import Levenshtein
        dist = Levenshtein.distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1.0 - _safe_ratio(dist, max_len)
    except ImportError:
        # Fallback: simple ratio
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        # Simple character overlap ratio as fallback
        common = sum(1 for a, b in zip(s1, s2) if a == b)
        return _safe_ratio(common, max_len)


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    try:
        from rapidfuzz.distance import JaroWinkler
        return JaroWinkler.similarity(s1, s2)
    except ImportError:
        return levenshtein_similarity(s1, s2)


def length_ratio(s1: str, s2: str) -> float:
    """Length ratio: min/max of string lengths."""
    l1, l2 = len(s1), len(s2)
    if l1 == 0 and l2 == 0:
        return 1.0
    return _safe_ratio(min(l1, l2), max(l1, l2))


def common_prefix_ratio(s1: str, s2: str) -> float:
    """Length of common prefix / max length."""
    if not s1 or not s2:
        return 0.0
    common = 0
    for a, b in zip(s1, s2):
        if a == b:
            common += 1
        else:
            break
    return _safe_ratio(common, max(len(s1), len(s2)))


# =============================================================================
# Lexical Features
# =============================================================================

def compute_lexical_features(s1: Record, candidate: Record) -> Dict[str, float]:
    """Compute lexical similarity features between two records."""
    features = {}

    # --- Name features ---
    n1 = s1.normalized_name
    n2 = candidate.normalized_name
    t1 = set(s1.name_tokens)
    t2 = set(candidate.name_tokens)

    features["name_exact"] = 1.0 if s1.original_name == candidate.original_name and s1.original_name else 0.0
    features["name_normalized_exact"] = 1.0 if n1 == n2 and n1 else 0.0
    features["name_jaccard"] = jaccard_similarity(t1, t2)
    features["name_token_overlap"] = token_overlap_ratio(s1.name_tokens, candidate.name_tokens)
    features["name_token_containment"] = token_containment(s1.name_tokens, candidate.name_tokens)
    features["name_levenshtein"] = levenshtein_similarity(n1, n2)
    features["name_jaro_winkler"] = jaro_winkler_similarity(n1, n2)
    features["name_length_ratio"] = length_ratio(n1, n2)
    features["name_common_token_count"] = float(len(t1 & t2))
    features["name_prefix_match"] = common_prefix_ratio(n1, n2)

    # Sorted-token exact match (handles word-order transpositions)
    sorted1 = " ".join(sorted(s1.name_tokens))
    sorted2 = " ".join(sorted(candidate.name_tokens))
    features["name_sorted_exact"] = 1.0 if sorted1 == sorted2 and sorted1 else 0.0

    # Compact name match
    features["compact_name_exact"] = 1.0 if s1.compact_name == candidate.compact_name and s1.compact_name else 0.0

    # --- Address features ---
    a1 = s1.normalized_address
    a2 = candidate.normalized_address
    at1 = set(s1.address_tokens)
    at2 = set(candidate.address_tokens)

    features["address_jaccard"] = jaccard_similarity(at1, at2)
    features["address_token_overlap"] = token_overlap_ratio(s1.address_tokens, candidate.address_tokens)
    features["address_levenshtein"] = levenshtein_similarity(a1, a2)
    features["address_length_ratio"] = length_ratio(a1, a2)

    # Numeric overlap in addresses
    num1 = set(s1.numeric_tokens)
    num2 = set(candidate.numeric_tokens)
    features["numeric_overlap"] = jaccard_similarity(num1, num2)

    return features


# =============================================================================
# Structural Features
# =============================================================================

def compute_structural_features(s1: Record, candidate: Record) -> Dict[str, float]:
    """Compute structural/component-level features."""
    features = {}

    features["country_exact"] = 1.0 if s1.country == candidate.country and s1.country else 0.0

    # Postal code
    p1, p2 = s1.postal_code, candidate.postal_code
    features["postal_present_both"] = 1.0 if p1 and p2 else 0.0
    features["postal_exact"] = 1.0 if p1 and p2 and p1 == p2 else 0.0
    if p1 and p2:
        common = 0
        for a, b in zip(p1, p2):
            if a == b:
                common += 1
            else:
                break
        features["postal_prefix_match"] = _safe_ratio(common, max(len(p1), len(p2)))
    else:
        features["postal_prefix_match"] = 0.0

    # House number
    h1, h2 = s1.house_number, candidate.house_number
    features["house_number_present_both"] = 1.0 if h1 and h2 else 0.0
    features["house_number_exact"] = 1.0 if h1 and h2 and h1 == h2 else 0.0

    # City
    c1, c2 = s1.city, candidate.city
    features["city_present_both"] = 1.0 if c1 and c2 else 0.0
    features["city_exact"] = 1.0 if c1 and c2 and c1 == c2 else 0.0

    # State
    st1, st2 = s1.state, candidate.state
    features["state_present_both"] = 1.0 if st1 and st2 else 0.0
    features["state_exact"] = 1.0 if st1 and st2 and st1 == st2 else 0.0

    # Presence flags
    features["both_name_present"] = 1.0 if s1.normalized_name and candidate.normalized_name else 0.0
    features["both_address_present"] = 1.0 if s1.normalized_address and candidate.normalized_address else 0.0

    # Length ratios
    features["name_len_ratio"] = length_ratio(s1.normalized_name, candidate.normalized_name)
    features["address_len_ratio"] = length_ratio(s1.normalized_address, candidate.normalized_address)

    # Numeric token count difference
    features["numeric_count_diff"] = abs(len(s1.numeric_tokens) - len(candidate.numeric_tokens))

    # Source type
    features["candidate_is_s2"] = 1.0 if candidate.source_id == "S2" else 0.0
    features["candidate_is_s3"] = 1.0 if candidate.source_id == "S3" else 0.0

    # Landmark overlap
    if s1.landmark and candidate.landmark:
        lm1 = set(s1.landmark.lower().split())
        lm2 = set(candidate.landmark.lower().split())
        features["landmark_overlap"] = jaccard_similarity(lm1, lm2)
    else:
        features["landmark_overlap"] = 0.0

    return features


# =============================================================================
# Semantic Features
# =============================================================================

def compute_semantic_features(s1: Record, candidate: Record,
                              pair: Optional[CandidatePair] = None,
                              embeddings_cache: Optional[Dict] = None) -> Dict[str, float]:
    """Compute semantic embedding-based features.
    
    If embeddings_cache is provided, computes exact dot products.
    Otherwise, reads cosine similarities directly from CandidatePair ANN retrieval scores
    with empty-text semantic masking, eliminating multi-gigabyte embedding caches.
    """
    features = {}
    if embeddings_cache is not None:
        s1_emb = embeddings_cache.get(s1.record_id, {})
        cand_emb = embeddings_cache.get(candidate.record_id, {})

        for field in ["name", "address", "record"]:
            key = f"{field}_emb"
            e1 = s1_emb.get(key)
            e2 = cand_emb.get(key)
            if e1 is not None and e2 is not None:
                # Cosine similarity via dot product (embeddings are L2-normalized)
                features[f"{field}_cosine"] = float(np.dot(e1, e2))
            else:
                features[f"{field}_cosine"] = 0.0
        return features

    # Direct extraction from CandidatePair with semantic masking
    if pair is not None:
        features["name_cosine"] = float(pair.name_ann_score) if (s1.normalized_name and candidate.normalized_name) else 0.0
        features["address_cosine"] = float(pair.address_ann_score) if (s1.normalized_address and candidate.normalized_address) else 0.0
        features["record_cosine"] = float(pair.record_ann_score)
    else:
        features["name_cosine"] = 0.0
        features["address_cosine"] = 0.0
        features["record_cosine"] = 0.0

    return features


# =============================================================================
# Rarity Features
# =============================================================================

class RarityFeatureComputer:
    """Computes token rarity (IDF) features."""

    def __init__(self):
        self.token_df: Counter = Counter()  # document frequency
        self.total_docs: int = 0
        self.idf_cache: Dict[str, float] = {}
        self.median_idf: float = 0.0

    def fit(self, all_records: Dict[str, Record]) -> None:
        """Compute token document frequencies across all records."""
        self.total_docs = len(all_records)
        self.token_df = Counter()

        for record in all_records.values():
            unique_tokens = set(record.name_tokens) | set(record.address_tokens)
            for token in unique_tokens:
                self.token_df[token] += 1

        # Compute IDF
        for token, df in self.token_df.items():
            self.idf_cache[token] = math.log(self.total_docs / (1 + df))

        if self.idf_cache:
            self.median_idf = float(np.median(list(self.idf_cache.values())))
        else:
            self.median_idf = 0.0

        logger.info(f"Rarity: {len(self.idf_cache)} unique tokens, median IDF={self.median_idf:.3f}")

    def fit_from_source_files(self, filepaths: List[str], chunksize: int = 250000) -> None:
        """Compute token document frequencies directly from raw candidate TSV files in streaming chunks.
        
        Guarantees zero S1 leakage by strictly processing candidate sources (S2 + S3) and bounds
        RAM usage by avoiding loading full Record objects into memory.
        """
        self.token_df = Counter()
        self.total_docs = 0

        for fp in filepaths:
            if not os.path.exists(fp):
                continue
            for chunk in pd.read_csv(fp, sep="\t", dtype=str, chunksize=chunksize, usecols=["business_name", "business_address"]):
                chunk = chunk.fillna("")
                names = chunk["business_name"].values
                addrs = chunk["business_address"].values
                for n_val, a_val in zip(names, addrs):
                    self.total_docs += 1
                    toks = set(extract_normalized_name_tokens(n_val)) | set(extract_normalized_address_tokens(a_val))
                    for t in toks:
                        self.token_df[t] += 1

        self.idf_cache = {}
        for token, df in self.token_df.items():
            self.idf_cache[token] = math.log(max(1, self.total_docs) / (1 + df))

        if self.idf_cache:
            self.median_idf = float(np.median(list(self.idf_cache.values())))
        else:
            self.median_idf = 0.0

        logger.info(f"Rarity (streaming): {len(self.idf_cache)} unique tokens from {self.total_docs} candidate docs, median IDF={self.median_idf:.3f}")

    def _get_idf(self, token: str) -> float:
        return self.idf_cache.get(token, self.median_idf)

    def compute_rarity_features(self, s1: Record, candidate: Record) -> Dict[str, float]:
        """Compute rarity-based features for a pair."""
        features = {}

        # Name token rarity
        shared_name = set(s1.name_tokens) & set(candidate.name_tokens)
        if shared_name:
            shared_idfs = [self._get_idf(t) for t in shared_name]
            rare_tokens = [t for t in shared_name if self._get_idf(t) > self.median_idf]
            features["name_rare_token_overlap"] = float(len(rare_tokens))
            features["name_max_shared_idf"] = max(shared_idfs)
            features["name_mean_shared_idf"] = float(np.mean(shared_idfs))
            features["name_sum_shared_idf"] = sum(shared_idfs)
        else:
            features["name_rare_token_overlap"] = 0.0
            features["name_max_shared_idf"] = 0.0
            features["name_mean_shared_idf"] = 0.0
            features["name_sum_shared_idf"] = 0.0

        # Address token rarity
        shared_addr = set(s1.address_tokens) & set(candidate.address_tokens)
        if shared_addr:
            shared_idfs = [self._get_idf(t) for t in shared_addr]
            features["address_rare_token_overlap"] = float(
                len([t for t in shared_addr if self._get_idf(t) > self.median_idf])
            )
            features["address_max_shared_idf"] = max(shared_idfs)
        else:
            features["address_rare_token_overlap"] = 0.0
            features["address_max_shared_idf"] = 0.0

        # Postal rarity
        if s1.postal_code:
            features["postal_rarity"] = self._get_idf(s1.postal_code)
        else:
            features["postal_rarity"] = 0.0

        return features


# =============================================================================
# Retrieval Features
# =============================================================================

def compute_retrieval_features(pair: CandidatePair) -> Dict[str, float]:
    """Extract features from the retrieval/blocking metadata."""
    features = {}

    features["found_exact"] = 1.0 if pair.found_by_exact else 0.0
    features["found_char_tfidf"] = 1.0 if pair.found_by_char_tfidf else 0.0
    features["found_word_tfidf"] = 1.0 if pair.found_by_word_tfidf else 0.0
    features["found_address_tfidf"] = 1.0 if pair.found_by_address_tfidf else 0.0
    features["found_bm25"] = 1.0 if pair.found_by_bm25 else 0.0
    features["found_name_ann"] = 1.0 if pair.found_by_name_ann else 0.0
    features["found_address_ann"] = 1.0 if pair.found_by_address_ann else 0.0
    features["found_record_ann"] = 1.0 if pair.found_by_record_ann else 0.0

    # Retrieval similarity scores (continuous features)
    features["char_tfidf_score"] = float(pair.char_tfidf_score)
    features["word_tfidf_score"] = float(pair.word_tfidf_score)
    features["address_tfidf_score"] = float(pair.address_tfidf_score)
    features["bm25_score"] = float(pair.bm25_score)
    features["name_ann_score"] = float(pair.name_ann_score)
    features["address_ann_score"] = float(pair.address_ann_score)
    features["record_ann_score"] = float(pair.record_ann_score)

    features["retrieval_votes"] = float(pair.retrieval_votes)
    features["rrf_score"] = pair.rrf_score
    features["is_multi_channel"] = 1.0 if pair.retrieval_votes >= 2 else 0.0

    # Rank features (0 = not found)
    ranks = []
    for rank_val in [pair.name_ann_rank, pair.address_ann_rank, pair.record_ann_rank,
                     pair.char_tfidf_rank, pair.word_tfidf_rank, pair.address_tfidf_rank, pair.bm25_rank]:
        if rank_val > 0:
            ranks.append(rank_val)

    features["best_rank"] = float(min(ranks)) if ranks else 0.0
    features["mean_rank"] = float(np.mean(ranks)) if ranks else 0.0
    features["name_ann_rank"] = float(pair.name_ann_rank)
    features["address_ann_rank"] = float(pair.address_ann_rank)
    features["record_ann_rank"] = float(pair.record_ann_rank)
    features["bm25_rank"] = float(pair.bm25_rank)

    return features


# =============================================================================
# Master Feature Engine
# =============================================================================

class FeatureEngine:
    """Master feature engineering module that orchestrates all feature computers."""

    def __init__(self, config: dict):
        self.config = config
        self.rarity_computer = RarityFeatureComputer()
        self.embeddings_cache: Optional[Dict] = None
        self._feature_names: Optional[List[str]] = None

    def setup(self, records: Dict[str, Record],
              embeddings_cache: Optional[Dict] = None) -> None:
        """Initialize feature computers with candidate corpus data (preventing S1 leakage)."""
        # Fit rarity strictly on candidate universe (S2 + S3) to prevent unsupervised holdout S1 leakage
        cand_records = {k: v for k, v in records.items() if v.source_id != "S1"}
        fit_records = cand_records if cand_records else records
        self.rarity_computer.fit(fit_records)
        self.embeddings_cache = embeddings_cache
        logger.info("FeatureEngine setup complete")

    def set_embeddings_cache(self, embeddings_cache: Optional[Dict] = None) -> None:
        """Update embeddings cache while preserving trained global rarity statistics."""
        self.embeddings_cache = embeddings_cache

    def compute_pair_features(self, s1: Record, candidate: Record,
                               pair: CandidatePair) -> PairFeatures:
        """Compute all features for a single candidate pair."""
        all_features = {}

        # Lexical features
        all_features.update(compute_lexical_features(s1, candidate))

        # Structural features
        all_features.update(compute_structural_features(s1, candidate))

        # Semantic features
        all_features.update(compute_semantic_features(
            s1, candidate, pair=pair, embeddings_cache=self.embeddings_cache
        ))

        # Rarity features
        all_features.update(self.rarity_computer.compute_rarity_features(s1, candidate))

        # Retrieval features
        all_features.update(compute_retrieval_features(pair))

        pf = PairFeatures(
            s1_id=s1.record_id,
            candidate_id=candidate.record_id,
            features=all_features,
        )

        # Cache feature names
        if self._feature_names is None:
            self._feature_names = sorted(all_features.keys())

        return pf

    def get_feature_names(self) -> List[str]:
        """Return sorted list of feature names."""
        return self._feature_names or []

    def features_to_arrays(self, all_features: List[PairFeatures]) -> Tuple:
        """Convert PairFeatures list to numpy arrays.
        
        Returns:
            (X, s1_ids, candidate_ids, labels)
        """
        if not all_features:
            return np.array([]), [], [], np.array([])

        names = self.get_feature_names()
        X = np.zeros((len(all_features), len(names)), dtype=np.float32)
        s1_ids = []
        cand_ids = []
        labels = []

        for i, pf in enumerate(all_features):
            for j, name in enumerate(names):
                X[i, j] = pf.features.get(name, 0.0)
            s1_ids.append(pf.s1_id)
            cand_ids.append(pf.candidate_id)
            labels.append(pf.label)

        return X, s1_ids, cand_ids, np.array(labels)
