"""
Hard negative mining module.

Iteratively identifies false positives with high model confidence
and incorporates them into training to refine model discrimination.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from preprocessing.schema import CandidatePair, PairFeatures

logger = logging.getLogger(__name__)


def mine_hard_negatives(
    model,
    X_candidates: np.ndarray,
    s1_ids: List[str],
    candidate_ids: List[str],
    ground_truth: Dict[str, List[str]],
    score_threshold: float = 0.4,
    top_k_per_s1: int = 3
) -> List[Tuple[str, str, int]]:
    """Mine hard negative pairs where model predicted high score for non-match.
    
    Args:
        model: Trained classifier with predict() method.
        X_candidates: Feature matrix of candidate pairs.
        s1_ids: S1 entity IDs for each row in X_candidates.
        candidate_ids: Candidate IDs for each row in X_candidates.
        ground_truth: True matching labels.
        score_threshold: Minimum predicted score to be considered hard false positive.
        top_k_per_s1: Maximum hard negatives to collect per S1 entity.
        
    Returns:
        List of (s1_id, candidate_id, 0) hard negative pairs.
    """
    preds = model.predict(X_candidates)
    
    s1_false_positives: Dict[str, List[Tuple[str, float]]] = {}

    for i in range(len(preds)):
        score = float(preds[i])
        s1 = s1_ids[i]
        cand = candidate_ids[i]

        true_matches = set(ground_truth.get(s1, []))
        if cand not in true_matches and score >= score_threshold:
            if s1 not in s1_false_positives:
                s1_false_positives[s1] = []
            s1_false_positives[s1].append((cand, score))

    hard_negatives = []
    for s1, fp_list in s1_false_positives.items():
        fp_list.sort(key=lambda x: -x[1])
        for cand, _ in fp_list[:top_k_per_s1]:
            hard_negatives.append((s1, cand, 0))

    logger.info(f"Mined {len(hard_negatives)} hard negative pairs from {len(s1_false_positives)} entities")
    return hard_negatives


def mine_hard_negatives_streaming(
    model,
    chunk_paths: List[str],
    ground_truth: Dict[str, List[str]],
    score_threshold: float = 0.35,
    top_k_per_s1: int = 5,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], List[str]]:
    """Stream-mine hard negatives across disk-backed candidate feature chunks without loading all pairs into RAM.
    
    Returns:
        (X_extra, y_extra, extra_s1, extra_cand)
    """
    import os
    from collections import defaultdict

    if not chunk_paths:
        return None, None, [], []

    # Map s1_id -> list of (score, chunk_idx, row_idx)
    s1_fps: Dict[str, List[Tuple[float, int, int]]] = {}

    for c_idx, path in enumerate(chunk_paths):
        if not os.path.exists(path):
            continue
        data = np.load(path, allow_pickle=True)
        X_chunk = data["X"]
        s1_chunk = data["s1_ids"]
        cand_chunk = data["cand_ids"]

        preds = model.predict(X_chunk)
        for row_i in range(len(preds)):
            score = float(preds[row_i])
            s1 = str(s1_chunk[row_i])
            cand = str(cand_chunk[row_i])

            true_matches = set(ground_truth.get(s1, []))
            if cand not in true_matches and score >= score_threshold:
                if s1 not in s1_fps:
                    s1_fps[s1] = []
                s1_fps[s1].append((score, c_idx, row_i))

    # For each S1 entity, retain top-k hard negatives
    selected_by_chunk: Dict[int, List[int]] = defaultdict(list)
    total_selected = 0
    for s1, fp_list in s1_fps.items():
        fp_list.sort(key=lambda x: -x[0])
        for _, c_idx, row_i in fp_list[:top_k_per_s1]:
            selected_by_chunk[c_idx].append(row_i)
            total_selected += 1

    if total_selected == 0:
        logger.info("No high-confidence false positives found across unmined candidate chunks.")
        return None, None, [], []

    # Second pass: read only selected rows from each chunk
    extra_X_list = []
    extra_y_list = []
    extra_s1 = []
    extra_cand = []

    for c_idx, row_indices in selected_by_chunk.items():
        path = chunk_paths[c_idx]
        data = np.load(path, allow_pickle=True)
        X_c = data["X"][row_indices]
        y_c = data["y"][row_indices]
        s1_c = data["s1_ids"][row_indices]
        cand_c = data["cand_ids"][row_indices]

        extra_X_list.append(X_c)
        extra_y_list.append(y_c)
        extra_s1.extend([str(s) for s in s1_c])
        extra_cand.extend([str(c) for c in cand_c])

    X_extra = np.vstack(extra_X_list)
    y_extra = np.concatenate(extra_y_list)
    logger.info(f"Stream-mined {len(extra_s1):,} hard negative pairs across {len(s1_fps):,} entities")
    return X_extra, y_extra, extra_s1, extra_cand
