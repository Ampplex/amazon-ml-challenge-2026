"""
Training pair construction module.

Generates positive pairs from ground truth and hard negative pairs
from retrieval/blocking outputs, ensuring entity-level splits to prevent leakage.
"""

import logging
import random
from typing import Dict, List, Set, Tuple

from preprocessing.schema import CandidatePair

logger = logging.getLogger(__name__)


def build_positive_pairs_from_candidates(
    candidates_dict: Dict[str, List[CandidatePair]],
    ground_truth: Dict[str, List[str]]
) -> Tuple[List[Tuple[str, str, int]], int, int]:
    """Build positive training pairs strictly from retrieved candidates.
    
    Ensures the model trains on the exact distribution of pairs reachable at inference.
    Matches in ground truth that were not retrieved are logged as blocking failures.
    
    Returns:
        (positive_pairs, total_ground_truth_matches, retrieved_ground_truth_matches)
    """
    pos_pairs = []
    total_gt = 0
    retrieved_gt = 0

    for s1_id, matched_ids in ground_truth.items():
        true_set = set(matched_ids)
        total_gt += len(true_set)

        retrieved_ids = {c.candidate_id for c in candidates_dict.get(s1_id, [])}
        captured = true_set & retrieved_ids
        retrieved_gt += len(captured)

        for mid in captured:
            pos_pairs.append((s1_id, mid, 1))

    missed = total_gt - retrieved_gt
    logger.info(
        f"Training Positives: {len(pos_pairs)} pairs captured from candidates "
        f"({retrieved_gt}/{total_gt} GT matches, {missed} blocking misses)"
    )
    return pos_pairs, total_gt, retrieved_gt


def build_negative_pairs_from_candidates(
    candidates_dict: Dict[str, List[CandidatePair]],
    ground_truth: Dict[str, List[str]],
    negative_ratio: int = 4,
    singleton_sample_count: int = 2,
    seed: int = 42
) -> List[Tuple[str, str, int]]:
    """Build hard negative pairs from retrieved candidate sets.
    
    Candidates that were retrieved but do not appear in ground truth
    are high-quality hard negatives.
    
    Args:
        candidates_dict: Map of s1_id -> list of CandidatePair.
        ground_truth: Map of s1_id -> true matched IDs.
        negative_ratio: Max negatives per positive match.
        singleton_sample_count: Max negatives to sample for singletons (0 matches).
        seed: Random seed.
        
    Returns:
        List of (s1_id, candidate_id, 0).
    """
    random.seed(seed)
    neg_pairs = []

    for s1_id, cand_list in candidates_dict.items():
        true_matches = set(ground_truth.get(s1_id, []))
        
        # Hard candidates that are NOT true matches
        hard_cands = [c.candidate_id for c in cand_list if c.candidate_id not in true_matches]

        if not hard_cands:
            continue

        if len(true_matches) > 0:
            target_count = min(len(hard_cands), len(true_matches) * negative_ratio)
        else:
            target_count = min(len(hard_cands), singleton_sample_count)

        if target_count > 0:
            # Sort or sample: top retrieved negatives are the hardest!
            # Since cand_list is already sorted by RRF / retrieval score, first elements are hardest
            selected = hard_cands[:target_count]
            for cid in selected:
                neg_pairs.append((s1_id, cid, 0))

    logger.info(f"Built {len(neg_pairs)} hard negative training pairs")
    return neg_pairs


def split_by_s1_entity(
    s1_ids: List[str],
    ground_truth: Dict[str, List[str]],
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[Set[str], Set[str]]:
    """Split S1 entities into train and validation sets to ensure zero leakage.
    
    Args:
        s1_ids: All S1 entity IDs.
        ground_truth: Ground truth mapping.
        val_ratio: Fraction for validation.
        seed: Random seed.
        
    Returns:
        (train_s1_ids, val_s1_ids) as sets.
    """
    random.seed(seed)
    shuffled = list(s1_ids)
    random.shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)
    val_s1_ids = set(shuffled[:val_size])
    train_s1_ids = set(shuffled[val_size:])

    logger.info(f"Split S1 entities: {len(train_s1_ids)} train, {len(val_s1_ids)} validation")
    return train_s1_ids, val_s1_ids


def split_train_cal_holdout(
    s1_ids: List[str],
    cal_ratio: float = 0.10,
    holdout_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Strict 3-way entity-level split: Train, Calibration, and Untouched Holdout.
    
    Guarantees:
        train ∩ calibration = ∅
        train ∩ holdout = ∅
        calibration ∩ holdout = ∅
    
    Returns:
        (train_ids, cal_ids, holdout_ids)
    """
    random.seed(seed)
    shuffled = list(s1_ids)
    random.shuffle(shuffled)

    n_holdout = int(len(shuffled) * holdout_ratio)
    n_cal = int(len(shuffled) * cal_ratio)

    holdout_ids = set(shuffled[:n_holdout])
    cal_ids = set(shuffled[n_holdout:n_holdout + n_cal])
    train_ids = set(shuffled[n_holdout + n_cal:])

    assert len(train_ids & cal_ids) == 0, "Train and Calibration overlap!"
    assert len(train_ids & holdout_ids) == 0, "Train and Holdout overlap!"
    assert len(cal_ids & holdout_ids) == 0, "Calibration and Holdout overlap!"

    logger.info(
        f"3-Way Split: {len(train_ids)} Train, {len(cal_ids)} Calibration, "
        f"{len(holdout_ids)} Untouched Holdout"
    )
    return train_ids, cal_ids, holdout_ids
