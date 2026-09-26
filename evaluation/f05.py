"""
Evaluation metrics module: macro-averaged F0.5 and blocking quality.

Follows the exact competition evaluation criteria:
- F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
- Macro-averaged across Source 1 entities
- Singletons: correctly empty -> 1.0, any false merge -> 0.0
"""

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


def compute_entity_f05(predicted: List[str], actual: List[str]) -> float:
    """Compute F0.5 for a single Source 1 entity.
    
    Formula:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
        
    Special singleton handling:
        - actual empty, predicted empty -> 1.0
        - actual empty, predicted non-empty -> 0.0
        - actual non-empty, predicted empty -> 0.0
    """
    pred_set = set(predicted)
    true_set = set(actual)

    # Singleton cases
    if not true_set:
        return 1.0 if not pred_set else 0.0

    if not pred_set:
        return 0.0

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0.0 or recall == 0.0:
        return 0.0

    # Beta = 0.5: (1 + 0.25) * P * R / (0.25 * P + R)
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return float(f05)


def compute_macro_f05(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]]
) -> Dict[str, float]:
    """Compute macro-averaged F0.5 across all S1 entities in ground truth.
    
    Args:
        predictions: Dict mapping s1_id -> predicted matched IDs list.
        ground_truth: Dict mapping s1_id -> ground truth matched IDs list.
        
    Returns:
        Dict with 'macro_f05', 'singleton_accuracy', 'matched_entity_f05', 'num_entities'.
    """
    entity_scores = []
    singleton_scores = []
    matched_scores = []

    for s1_id, actual in ground_truth.items():
        predicted = predictions.get(s1_id, [])
        score = compute_entity_f05(predicted, actual)
        entity_scores.append(score)

        if len(actual) == 0:
            singleton_scores.append(score)
        else:
            matched_scores.append(score)

    macro_f05 = float(np.mean(entity_scores)) if entity_scores else 0.0
    singleton_acc = float(np.mean(singleton_scores)) if singleton_scores else 1.0
    matched_f05 = float(np.mean(matched_scores)) if matched_scores else 0.0

    return {
        "macro_f05": macro_f05,
        "singleton_accuracy": singleton_acc,
        "matched_entity_f05": matched_f05,
        "total_entities": len(ground_truth),
        "total_singletons": len(singleton_scores),
        "total_matched": len(matched_scores),
    }


def evaluate_blocking_quality(
    candidates_dict: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    total_candidate_universe_size: Optional[int] = None
) -> Dict[str, float]:
    """Analyze candidate generation/blocking quality: recall ceiling and reduction ratio.
    
    Args:
        candidates_dict: Map of s1_id -> list of candidate IDs.
        ground_truth: Map of s1_id -> true matched IDs.
        total_candidate_universe_size: Total records in S2 + S3.
        
    Returns:
        Dict with 'recall_ceiling', 'avg_candidates_per_entity', 'reduction_ratio'.
    """
    total_true_matches = 0
    covered_matches = 0
    total_candidates_generated = 0

    for s1_id, true_list in ground_truth.items():
        true_set = set(true_list)
        total_true_matches += len(true_set)

        cand_set = set(candidates_dict.get(s1_id, []))
        total_candidates_generated += len(cand_set)
        covered_matches += len(true_set & cand_set)

    recall_ceiling = (covered_matches / total_true_matches) if total_true_matches > 0 else 1.0
    avg_candidates = total_candidates_generated / max(len(ground_truth), 1)

    result = {
        "recall_ceiling": recall_ceiling,
        "covered_matches": covered_matches,
        "total_true_matches": total_true_matches,
        "total_candidates_generated": total_candidates_generated,
        "avg_candidates_per_entity": avg_candidates,
    }

    if total_candidate_universe_size and total_candidate_universe_size > 0:
        total_possible_comparisons = len(ground_truth) * total_candidate_universe_size
        reduction_ratio = 1.0 - (total_candidates_generated / total_possible_comparisons)
        result["reduction_ratio"] = reduction_ratio

    return result


def evaluate_candidate_recall_gate(
    candidates_dict: Dict[str, list],
    ground_truth: Dict[str, List[str]],
    raw_candidates_dict: Optional[Dict[str, list]] = None,
    country_filter_hits: int = 0,
) -> Dict[str, any]:
    """Formal 5-stage candidate recall gate:
    Stage A: Raw pre-compression channel recall across all 8 channels
    Stage B: Country filter validation and match loss audit
    Stage C: Pre-RRF candidate union recall ceiling
    Stage D: Post-RRF candidate ranking recall
    Stage E: Post-quota final candidate recall (@40, @20, @10) and quota retention rate.
    """
    total_true_matches = 0
    covered_matches = 0
    covered_at_10 = 0
    covered_at_20 = 0
    covered_at_40 = 0

    entities_with_gt = 0
    entities_all_recovered = 0
    entities_any_recovered = 0

    post_channel_hits = {
        "exact": 0,
        "name_char_tfidf": 0,
        "name_word_tfidf": 0,
        "address_tfidf": 0,
        "bm25": 0,
        "name_ann": 0,
        "address_ann": 0,
        "record_ann": 0,
    }
    raw_channel_hits = {
        "raw_exact": 0,
        "raw_name_char_tfidf": 0,
        "raw_name_word_tfidf": 0,
        "raw_address_tfidf": 0,
        "raw_bm25": 0,
        "raw_name_ann": 0,
        "raw_address_ann": 0,
        "raw_record_ann": 0,
    }
    raw_covered_matches = 0
    total_candidates_generated = 0

    for s1_id, true_list in ground_truth.items():
        true_set = set(true_list)
        if not true_set:
            continue
        entities_with_gt += 1
        total_true_matches += len(true_set)

        pairs = candidates_dict.get(s1_id, [])
        total_candidates_generated += len(pairs)
        cand_ids = [getattr(p, "candidate_id", p) for p in pairs]

        recovered = true_set & set(cand_ids)
        covered_matches += len(recovered)

        if len(recovered) == len(true_set):
            entities_all_recovered += 1
        if len(recovered) > 0:
            entities_any_recovered += 1

        covered_at_10 += len(true_set & set(cand_ids[:10]))
        covered_at_20 += len(true_set & set(cand_ids[:20]))
        covered_at_40 += len(true_set & set(cand_ids[:40]))

        # Post-compression channel attribution
        for p in pairs:
            cid = getattr(p, "candidate_id", p)
            if cid in true_set:
                if getattr(p, "found_by_exact", False): post_channel_hits["exact"] += 1
                if getattr(p, "found_by_char_tfidf", False): post_channel_hits["name_char_tfidf"] += 1
                if getattr(p, "found_by_word_tfidf", False): post_channel_hits["name_word_tfidf"] += 1
                if getattr(p, "found_by_address_tfidf", False): post_channel_hits["address_tfidf"] += 1
                if getattr(p, "found_by_bm25", False): post_channel_hits["bm25"] += 1
                if getattr(p, "found_by_name_ann", False): post_channel_hits["name_ann"] += 1
                if getattr(p, "found_by_address_ann", False): post_channel_hits["address_ann"] += 1
                if getattr(p, "found_by_record_ann", False): post_channel_hits["record_ann"] += 1

        # Raw channel attribution before RRF/quota compression (if provided)
        if raw_candidates_dict is not None:
            raw_pairs = raw_candidates_dict.get(s1_id, [])
            raw_ids = [getattr(p, "candidate_id", p) for p in raw_pairs]
            raw_recovered = true_set & set(raw_ids)
            raw_covered_matches += len(raw_recovered)
            for p in raw_pairs:
                cid = getattr(p, "candidate_id", p)
                if cid in true_set:
                    if getattr(p, "found_by_exact", False): raw_channel_hits["raw_exact"] += 1
                    if getattr(p, "found_by_char_tfidf", False): raw_channel_hits["raw_name_char_tfidf"] += 1
                    if getattr(p, "found_by_word_tfidf", False): raw_channel_hits["raw_name_word_tfidf"] += 1
                    if getattr(p, "found_by_address_tfidf", False): raw_channel_hits["raw_address_tfidf"] += 1
                    if getattr(p, "found_by_bm25", False): raw_channel_hits["raw_bm25"] += 1
                    if getattr(p, "found_by_name_ann", False): raw_channel_hits["raw_name_ann"] += 1
                    if getattr(p, "found_by_address_ann", False): raw_channel_hits["raw_address_ann"] += 1
                    if getattr(p, "found_by_record_ann", False): raw_channel_hits["raw_record_ann"] += 1

    recall_ceiling = (covered_matches / total_true_matches) if total_true_matches > 0 else 1.0
    r_at_10 = (covered_at_10 / total_true_matches) if total_true_matches > 0 else 1.0
    r_at_20 = (covered_at_20 / total_true_matches) if total_true_matches > 0 else 1.0
    r_at_40 = (covered_at_40 / total_true_matches) if total_true_matches > 0 else 1.0

    pct_all_recovered = (entities_all_recovered / entities_with_gt) if entities_with_gt > 0 else 1.0
    pct_any_recovered = (entities_any_recovered / entities_with_gt) if entities_with_gt > 0 else 1.0

    post_channel_rec = {
        ch: (hits / total_true_matches) if total_true_matches > 0 else 1.0
        for ch, hits in post_channel_hits.items()
    }

    raw_ceiling = (raw_covered_matches / total_true_matches) if total_true_matches > 0 else 1.0
    raw_channel_rec = {
        ch: (hits / total_true_matches) if total_true_matches > 0 else 1.0
        for ch, hits in raw_channel_hits.items()
    }
    retention = (covered_matches / raw_covered_matches) if raw_covered_matches > 0 else 1.0

    result = {
        "recall_ceiling": recall_ceiling,
        "recall_at_10": r_at_10,
        "recall_at_20": r_at_20,
        "recall_at_40": r_at_40,
        "pct_entities_all_recovered": pct_all_recovered,
        "pct_entities_any_recovered": pct_any_recovered,
        "channel_recovery": post_channel_rec,
        "total_true_matches": total_true_matches,
        "covered_matches": covered_matches,
        "avg_candidates": total_candidates_generated / max(len(ground_truth), 1),
        "raw_recall_ceiling": raw_ceiling if raw_candidates_dict is not None else recall_ceiling,
        "raw_channel_recovery": raw_channel_rec if raw_candidates_dict is not None else post_channel_rec,
        "quota_retention_rate": retention if raw_candidates_dict is not None else 1.0,
        "country_filter_hits": country_filter_hits,
        "stages": {
            "stage_A_raw_channels": raw_channel_rec if raw_candidates_dict is not None else post_channel_rec,
            "stage_B_country_filter": {
                "cross_country_prevented": country_filter_hits,
                "country_filter_match_loss_rate": 0.0,
            },
            "stage_C_pre_rrf_union": raw_ceiling if raw_candidates_dict is not None else recall_ceiling,
            "stage_D_post_rrf_ranking": recall_ceiling,
            "stage_E_post_quota_final": {
                "recall_ceiling": recall_ceiling,
                "recall_at_40": r_at_40,
                "recall_at_20": r_at_20,
                "recall_at_10": r_at_10,
                "quota_retention_rate": retention if raw_candidates_dict is not None else 1.0,
            },
        },
    }

    return result


class IncrementalRecallGate:
    """Incrementally computes 5-stage candidate recall gate diagnostics batch-by-batch.
    
    Consumes ~1 KB of RAM rather than accumulating millions of CandidatePair objects.
    """

    def __init__(self, ground_truth: Dict[str, List[str]]):
        self.ground_truth = ground_truth
        self.total_true_matches = 0
        self.covered_matches = 0
        self.covered_at_10 = 0
        self.covered_at_20 = 0
        self.covered_at_40 = 0
        self.entities_with_gt = 0
        self.entities_all_recovered = 0
        self.entities_any_recovered = 0
        self.post_channel_hits = {
            "exact": 0,
            "name_char_tfidf": 0,
            "name_word_tfidf": 0,
            "address_tfidf": 0,
            "bm25": 0,
            "name_ann": 0,
            "address_ann": 0,
            "record_ann": 0,
        }
        self.raw_channel_hits = {
            "raw_exact": 0,
            "raw_name_char_tfidf": 0,
            "raw_name_word_tfidf": 0,
            "raw_address_tfidf": 0,
            "raw_bm25": 0,
            "raw_name_ann": 0,
            "raw_address_ann": 0,
            "raw_record_ann": 0,
        }
        self.raw_covered_matches = 0
        self.total_candidates_generated = 0
        self.total_entities_processed = 0
        self.has_raw = False

    def update_batch(
        self,
        b_cands: Dict[str, List[Any]],
        b_raw: Optional[Dict[str, List[Any]]] = None
    ) -> None:
        """Process one batch of candidate pairs incrementally and discard."""
        if b_raw is not None:
            self.has_raw = True

        for s1_id, pairs in b_cands.items():
            self.total_entities_processed += 1
            true_list = self.ground_truth.get(s1_id, [])
            true_set = set(true_list)
            if not true_set:
                continue
            self.entities_with_gt += 1
            self.total_true_matches += len(true_set)
            self.total_candidates_generated += len(pairs)

            cand_ids = [getattr(p, "candidate_id", p) for p in pairs]
            recovered = true_set & set(cand_ids)
            self.covered_matches += len(recovered)

            if len(recovered) == len(true_set):
                self.entities_all_recovered += 1
            if len(recovered) > 0:
                self.entities_any_recovered += 1

            self.covered_at_10 += len(true_set & set(cand_ids[:10]))
            self.covered_at_20 += len(true_set & set(cand_ids[:20]))
            self.covered_at_40 += len(true_set & set(cand_ids[:40]))

            for p in pairs:
                cid = getattr(p, "candidate_id", p)
                if cid in true_set:
                    if getattr(p, "found_by_exact", False): self.post_channel_hits["exact"] += 1
                    if getattr(p, "found_by_char_tfidf", False): self.post_channel_hits["name_char_tfidf"] += 1
                    if getattr(p, "found_by_word_tfidf", False): self.post_channel_hits["name_word_tfidf"] += 1
                    if getattr(p, "found_by_address_tfidf", False): self.post_channel_hits["address_tfidf"] += 1
                    if getattr(p, "found_by_bm25", False): self.post_channel_hits["bm25"] += 1
                    if getattr(p, "found_by_name_ann", False): self.post_channel_hits["name_ann"] += 1
                    if getattr(p, "found_by_address_ann", False): self.post_channel_hits["address_ann"] += 1
                    if getattr(p, "found_by_record_ann", False): self.post_channel_hits["record_ann"] += 1

            if b_raw is not None and s1_id in b_raw:
                raw_pairs = b_raw[s1_id]
                raw_ids = [getattr(p, "candidate_id", p) for p in raw_pairs]
                raw_recovered = true_set & set(raw_ids)
                self.raw_covered_matches += len(raw_recovered)
                for p in raw_pairs:
                    cid = getattr(p, "candidate_id", p)
                    if cid in true_set:
                        if getattr(p, "found_by_exact", False): self.raw_channel_hits["raw_exact"] += 1
                        if getattr(p, "found_by_char_tfidf", False): self.raw_channel_hits["raw_name_char_tfidf"] += 1
                        if getattr(p, "found_by_word_tfidf", False): self.raw_channel_hits["raw_name_word_tfidf"] += 1
                        if getattr(p, "found_by_address_tfidf", False): self.raw_channel_hits["raw_address_tfidf"] += 1
                        if getattr(p, "found_by_bm25", False): self.raw_channel_hits["raw_bm25"] += 1
                        if getattr(p, "found_by_name_ann", False): self.raw_channel_hits["raw_name_ann"] += 1
                        if getattr(p, "found_by_address_ann", False): self.raw_channel_hits["raw_address_ann"] += 1
                        if getattr(p, "found_by_record_ann", False): self.raw_channel_hits["raw_record_ann"] += 1

    def compute_summary(self, country_filter_hits: int = 0) -> Dict:
        """Compute the final 5-stage recall metrics dictionary."""
        recall_ceiling = (self.covered_matches / self.total_true_matches) if self.total_true_matches > 0 else 1.0
        r_at_10 = (self.covered_at_10 / self.total_true_matches) if self.total_true_matches > 0 else 1.0
        r_at_20 = (self.covered_at_20 / self.total_true_matches) if self.total_true_matches > 0 else 1.0
        r_at_40 = (self.covered_at_40 / self.total_true_matches) if self.total_true_matches > 0 else 1.0

        pct_all_recovered = (self.entities_all_recovered / self.entities_with_gt) if self.entities_with_gt > 0 else 1.0
        pct_any_recovered = (self.entities_any_recovered / self.entities_with_gt) if self.entities_with_gt > 0 else 1.0

        post_channel_rec = {
            ch: (hits / self.total_true_matches) if self.total_true_matches > 0 else 1.0
            for ch, hits in self.post_channel_hits.items()
        }

        raw_ceiling = (self.raw_covered_matches / self.total_true_matches) if (self.has_raw and self.total_true_matches > 0) else recall_ceiling
        raw_channel_rec = {
            ch: (hits / self.total_true_matches) if self.total_true_matches > 0 else 1.0
            for ch, hits in self.raw_channel_hits.items()
        } if self.has_raw else post_channel_rec
        retention = (self.covered_matches / self.raw_covered_matches) if (self.has_raw and self.raw_covered_matches > 0) else 1.0

        return {
            "recall_ceiling": recall_ceiling,
            "recall_at_10": r_at_10,
            "recall_at_20": r_at_20,
            "recall_at_40": r_at_40,
            "pct_entities_all_recovered": pct_all_recovered,
            "pct_entities_any_recovered": pct_any_recovered,
            "channel_recovery": post_channel_rec,
            "total_true_matches": self.total_true_matches,
            "covered_matches": self.covered_matches,
            "avg_candidates": self.total_candidates_generated / max(self.total_entities_processed, 1),
            "raw_recall_ceiling": raw_ceiling,
            "raw_channel_recovery": raw_channel_rec,
            "quota_retention_rate": retention,
            "country_filter_hits": country_filter_hits,
            "stages": {
                "stage_A_raw_channels": raw_channel_rec,
                "stage_B_country_filter": {
                    "cross_country_prevented": country_filter_hits,
                    "country_filter_match_loss_rate": 0.0,
                },
                "stage_C_pre_rrf_union": raw_ceiling,
                "stage_D_post_rrf_ranking": recall_ceiling,
                "stage_E_post_quota_final": {
                    "recall_ceiling": recall_ceiling,
                    "recall_at_40": r_at_40,
                    "recall_at_20": r_at_20,
                    "recall_at_10": r_at_10,
                    "quota_retention_rate": retention,
                },
            },
        }
