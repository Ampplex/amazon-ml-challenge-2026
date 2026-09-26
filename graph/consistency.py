"""
Graph and cross-source consistency module.

Computes mutual consistency between Source 2 and Source 3 candidates
for the same Source 1 entity (triangle support / cross-source consistency).
"""

import logging
from typing import Dict, List, Set, Tuple
from rapidfuzz.distance import JaroWinkler
from preprocessing.schema import Record, CandidatePair

logger = logging.getLogger(__name__)


def compute_cross_source_consistency(
    s1_id: str,
    candidates: List[CandidatePair],
    candidate_records: Dict[str, Record],
    name_similarity_threshold: float = 0.8
) -> Dict[str, float]:
    """Compute triangle/cross-source consistency score for candidate pairs.
    
    If S1 matches S2_i and S3_j, and S2_i and S3_j are also highly similar,
    both receive high cross-source consistency support.
    
    Args:
        s1_id: Source 1 entity ID.
        candidates: List of CandidatePair for this S1 entity.
        candidate_records: Map of candidate_id -> Record.
        name_similarity_threshold: Threshold above which S2-S3 pair is considered compatible.
        
    Returns:
        Dict mapping candidate_id -> consistency_score (0.0 to 1.0).
    """
    s2_cands = [c.candidate_id for c in candidates if c.candidate_source == "S2"]
    s3_cands = [c.candidate_id for c in candidates if c.candidate_source == "S3"]

    consistency_scores: Dict[str, float] = {c.candidate_id: 0.0 for c in candidates}

    if not s2_cands or not s3_cands:
        return consistency_scores

    # Cross-compare S2 and S3 candidates
    for s2_id in s2_cands:
        r2 = candidate_records.get(s2_id)
        if not r2:
            continue
        max_sim_for_s2 = 0.0
        for s3_id in s3_cands:
            r3 = candidate_records.get(s3_id)
            if not r3:
                continue

            # Compare names and country/city
            if r2.country and r3.country and r2.country != r3.country:
                continue

            sim = JaroWinkler.similarity(r2.normalized_name, r3.normalized_name)
            if r2.postal_code and r3.postal_code and r2.postal_code == r3.postal_code:
                sim = min(1.0, sim + 0.1)

            if sim > max_sim_for_s2:
                max_sim_for_s2 = sim

            if sim >= name_similarity_threshold:
                consistency_scores[s3_id] = max(consistency_scores[s3_id], sim)

        consistency_scores[s2_id] = max_sim_for_s2

    return consistency_scores
