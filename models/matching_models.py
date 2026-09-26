"""
Models: LightGBM pair ranker, cross-encoder reranker, fusion, calibration, and decision engine.
"""

import logging
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Safe OpenMP defaults for macOS/Linux multi-runtime compatibility
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from preprocessing.schema import Record

logger = logging.getLogger(__name__)


# =============================================================================
# LightGBM Pair Ranker
# =============================================================================

class PairRanker:
    """LightGBM-based pairwise matching model."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.feature_names: List[str] = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              feature_names: List[str] = None) -> None:
        """Train the LightGBM pair model."""
        import lightgbm as lgb

        params = dict(self.config.get("pair_model", {}).get("params", {}))
        early_stop = self.config.get("pair_model", {}).get("early_stopping_rounds", 50)

        # Handle class_weight conversion for native LightGBM train API
        if "class_weight" in params:
            cw = params.pop("class_weight")
            if cw == "balanced" and "scale_pos_weight" not in params:
                num_pos = int(np.sum(y_train == 1))
                num_neg = int(np.sum(y_train == 0))
                params["scale_pos_weight"] = float(num_neg / max(num_pos, 1))

        n_estimators = params.pop("n_estimators", 1000)

        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)

        callbacks = [lgb.log_evaluation(100)]
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names)
            valid_sets.append(val_data)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(early_stop))

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        logger.info(f"PairRanker trained with {self.model.num_trees()} trees")

    def predict(self, X: np.ndarray, batch_size: int = 50000) -> np.ndarray:
        """Return match probabilities with batching for large or memmapped arrays."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        if len(X) <= batch_size:
            return self.model.predict(X)
        preds = []
        for i in range(0, len(X), batch_size):
            preds.append(self.model.predict(X[i:i + batch_size]))
        return np.concatenate(preds)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if self.model is None:
            return {}
        importance = self.model.feature_importance(importance_type="gain")
        names = self.model.feature_name()
        return dict(sorted(zip(names, importance), key=lambda x: -x[1]))

    def save(self, path: str) -> None:
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        # Save feature names
        with open(path + ".meta", "wb") as f:
            pickle.dump(self.feature_names, f)
        logger.info(f"PairRanker saved to {path}")

    def load(self, path: str) -> None:
        """Load model from file."""
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=path)
        meta_path = path + ".meta"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.feature_names = pickle.load(f)
        logger.info(f"PairRanker loaded from {path}")


# =============================================================================
# Cross-Encoder Reranker
# =============================================================================

class CrossEncoderReranker:
    """Selective cross-encoder for ambiguous pairs."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        ce_cfg = config.get("cross_encoder", {})
        self.model_name = ce_cfg.get("model_name", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        self.batch_size = ce_cfg.get("batch_size", 64)
        self.amb_cfg = ce_cfg.get("ambiguity", {})

        # Compute budget enforcement
        self.total_reranked_pairs: int = 0
        self.max_candidates_for_rerank: int = int(self.amb_cfg.get("max_candidates_for_rerank", 10))
        self.max_rerank_pairs_per_batch: int = int(ce_cfg.get("max_rerank_pairs_per_batch", 4096))
        self.max_rerank_pairs_total: int = int(ce_cfg.get("max_rerank_pairs_total", 1000000))
        self.max_rerank_fraction_per_batch: float = float(ce_cfg.get("max_rerank_fraction_per_batch", 0.05))

    def load_model(self) -> None:
        """Load the cross-encoder model. Fails fast if enabled and loading fails."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"CrossEncoder loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model '{self.model_name}': {e}")
            raise RuntimeError(f"CrossEncoder failed to load: {e}")

    def identify_ambiguous(self, scores: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[str]]:
        """Identify ambiguous pairs that need reranking under a strict compute budget.
        
        Args:
            scores: Dict mapping s1_id -> [(candidate_id, score), ...]
        
        Returns:
            Dict mapping s1_id -> [candidate_ids to rerank]
        """
        low = self.amb_cfg.get("score_band_low", 0.3)
        high = self.amb_cfg.get("score_band_high", 0.8)
        margin_thresh = self.amb_cfg.get("margin_threshold", 0.15)
        max_rerank = self.max_candidates_for_rerank

        total_input_pairs = sum(len(v) for v in scores.values())
        fraction_budget = max(1, int(total_input_pairs * self.max_rerank_fraction_per_batch))
        batch_budget = min(self.max_rerank_pairs_per_batch, fraction_budget)
        remaining_global = max(0, self.max_rerank_pairs_total - self.total_reranked_pairs)
        effective_budget = min(batch_budget, remaining_global)

        if effective_budget <= 0:
            logger.warning("CrossEncoder compute budget exhausted. Skipping further reranking.")
            return {}

        candidates_pool = []
        for s1_id, cand_scores in scores.items():
            if not cand_scores:
                continue

            sorted_scores = sorted(cand_scores, key=lambda x: -x[1])
            for cid, score in sorted_scores[:max_rerank]:
                is_ambiguous = False
                # In ambiguity band
                if low <= score <= high:
                    is_ambiguous = True
                # Small margin with top candidate
                elif len(sorted_scores) >= 2:
                    top_score = sorted_scores[0][1]
                    if score > low and (top_score - score) < margin_thresh:
                        is_ambiguous = True

                if is_ambiguous:
                    # Priority: closest to uncertainty boundary 0.5
                    ambiguity_priority = -abs(score - 0.5)
                    candidates_pool.append((s1_id, cid, ambiguity_priority))

        # Enforce budget: take top candidates sorted by ambiguity priority
        candidates_pool.sort(key=lambda x: -x[2])
        selected_pool = candidates_pool[:effective_budget]

        ambiguous: Dict[str, List[str]] = defaultdict(list)
        for s1_id, cid, _ in selected_pool:
            ambiguous[s1_id].append(cid)

        self.total_reranked_pairs += len(selected_pool)
        logger.info(
            f"CrossEncoder Budget: selected {len(selected_pool):,} ambiguous pairs "
            f"(effective budget: {effective_budget:,}, total run reranked: {self.total_reranked_pairs:,}/{self.max_rerank_pairs_total:,})"
        )
        return dict(ambiguous)

    def rerank(self, pairs: List[Tuple[str, str]],
               s1_records: Dict[str, Record],
               candidate_records: Dict[str, Record]) -> Dict[Tuple[str, str], float]:
        """Rerank pairs using cross-encoder.
        
        Returns:
            Dict mapping (s1_id, cand_id) -> cross_encoder_score
        """
        if self.model is None:
            logger.warning("Cross-encoder not loaded, skipping reranking")
            return {}

        # Build text pairs
        text_pairs = []
        pair_keys = []
        for s1_id, cand_id in pairs:
            s1_rec = s1_records.get(s1_id)
            cand_rec = candidate_records.get(cand_id)
            if s1_rec and cand_rec:
                text_a = f"{s1_rec.normalized_name} {s1_rec.normalized_address}" if hasattr(s1_rec, "normalized_name") else str(s1_rec)
                if isinstance(cand_rec, tuple):
                    text_b = f"{cand_rec[0]} {cand_rec[1]}"
                elif hasattr(cand_rec, "normalized_name"):
                    text_b = f"{cand_rec.normalized_name} {cand_rec.normalized_address}"
                else:
                    text_b = str(cand_rec)
                text_pairs.append([text_a, text_b])
                pair_keys.append((s1_id, cand_id))

        if not text_pairs:
            return {}

        # Score in batches
        scores = self.model.predict(text_pairs, batch_size=self.batch_size, show_progress_bar=False)

        # Normalize scores to [0, 1] using sigmoid
        from scipy.special import expit
        scores = expit(scores)
        logger.info(f"CrossEncoder scored {len(text_pairs)} pairs (range: {scores.min():.4f} .. {scores.max():.4f})")

        result = {}
        for key, score in zip(pair_keys, scores):
            result[key] = float(score)

        logger.info(f"Cross-encoder scored {len(result)} pairs")
        return result


# =============================================================================
# Fusion Model
# =============================================================================

class FusionModel:
    """Fuses LightGBM scores, cross-encoder scores, and other signals via calibrated score fusion."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        fusion_cfg = config.get("fusion_model", {})
        self.pair_weight = float(fusion_cfg.get("pair_ranker_weight", 0.4))
        self.ce_weight = float(fusion_cfg.get("cross_encoder_weight", 0.6))

    def fuse_scores(self, pair_score: float,
                     cross_encoder_score: Optional[float] = None,
                     retrieval_votes: int = 0,
                     name_cosine: float = 0.0) -> float:
        """Calibrated weighted score fusion."""
        if cross_encoder_score is not None:
            return self.pair_weight * pair_score + self.ce_weight * cross_encoder_score
        return pair_score

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """Train fusion model (small LightGBM)."""
        import lightgbm as lgb
        params = self.config.get("fusion_model", {}).get("params", {})
        n_estimators = params.pop("n_estimators", 300)
        early_stop = self.config.get("fusion_model", {}).get("early_stopping_rounds", 30)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        callbacks = [lgb.log_evaluation(100)]

        if X_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val)
            valid_sets.append(val_data)
            callbacks.append(lgb.early_stopping(early_stop))

        self.model = lgb.train(params, train_data, num_boost_round=n_estimators,
                                valid_sets=valid_sets, callbacks=callbacks)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Fusion model not trained")
        return self.model.predict(X)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> None:
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=path)


# =============================================================================
# Probability Calibration
# =============================================================================

class ProbabilityCalibrator:
    """Calibrates raw model scores into reliable probabilities."""

    def __init__(self, config: dict):
        self.config = config
        self.calibrator = None
        self.method = config.get("calibration", {}).get("method", "isotonic")

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit calibration model."""
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.calibrator.fit(raw_scores, labels)
        elif self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(raw_scores.reshape(-1, 1), labels)
        logger.info(f"Calibrator fitted using {self.method} method")

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        """Calibrate raw scores."""
        if self.calibrator is None:
            return raw_scores
        if self.method == "isotonic":
            return self.calibrator.predict(raw_scores)
        else:
            return self.calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((self.method, self.calibrator), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.method, self.calibrator = pickle.load(f)


# =============================================================================
# F0.5 Threshold Optimizer
# =============================================================================

class ThresholdOptimizer:
    """Optimizes decision threshold for macro F0.5."""

    def __init__(self, config: dict):
        self.config = config
        self.best_threshold: float = 0.5

    @staticmethod
    def compute_entity_f05(predicted: List[str], actual: List[str]) -> float:
        """Compute F0.5 for a single S1 entity."""
        pred_set = set(predicted)
        true_set = set(actual)

        # Both empty (true singleton correctly predicted)
        if not pred_set and not true_set:
            return 1.0
        # False merge on singleton
        if pred_set and not true_set:
            return 0.0
        # Missed all matches
        if not pred_set and true_set:
            return 0.0

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        beta_sq = 0.25  # beta=0.5, beta^2=0.25
        f05 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        return f05

    def compute_macro_f05(self, predictions: Dict[str, List[str]],
                           ground_truth: Dict[str, List[str]]) -> float:
        """Compute macro-averaged F0.5 across all S1 entities."""
        scores = []
        for s1_id, actual in ground_truth.items():
            predicted = predictions.get(s1_id, [])
            scores.append(self.compute_entity_f05(predicted, actual))
        return float(np.mean(scores)) if scores else 0.0

    def optimize(self, entity_scores: Dict[str, List[Tuple[str, float]]],
                  ground_truth: Dict[str, List[str]]) -> float:
        """Find threshold that maximizes macro F0.5.
        
        Args:
            entity_scores: s1_id -> [(candidate_id, calibrated_score), ...]
            ground_truth: s1_id -> [matched_ids]
        
        Returns:
            Best threshold
        """
        search_cfg = self.config.get("decision", {}).get("threshold_search", {})
        start = search_cfg.get("start", 0.30)
        stop = search_cfg.get("stop", 0.95)
        step = search_cfg.get("step", 0.01)

        best_f05 = -1.0
        best_t = 0.5

        thresholds = np.arange(start, stop + step, step)
        for t in thresholds:
            predictions = {}
            for s1_id, cand_scores in entity_scores.items():
                predictions[s1_id] = [cid for cid, score in cand_scores if score >= t]

            # Ensure all GT entities are present
            for s1_id in ground_truth:
                if s1_id not in predictions:
                    predictions[s1_id] = []

            f05 = self.compute_macro_f05(predictions, ground_truth)
            if f05 > best_f05:
                best_f05 = f05
                best_t = t

        self.best_threshold = best_t
        logger.info(f"Optimal threshold: {best_t:.3f} (F0.5={best_f05:.4f})")
        return best_t


# =============================================================================
# Entity Decision Engine
# =============================================================================

class EntityDecisionEngine:
    """Makes final entity-level match decisions."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def decide(self, candidate_scores: List[Tuple[str, float]]) -> List[str]:
        """Decide matches for one S1 entity.
        
        Returns list of matched IDs (can be empty, one, or many).
        """
        matches = [cid for cid, score in candidate_scores if score >= self.threshold]
        return matches

    def decide_all(self, all_scores: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[str]]:
        """Apply decision to all S1 entities."""
        decisions = {}
        for s1_id, cand_scores in all_scores.items():
            decisions[s1_id] = self.decide(cand_scores)
        return decisions

    @staticmethod
    def save_matching_results(decisions: Dict[str, List[str]],
                               s1_ids: List[str], output_path: str) -> None:
        """Write matching_results.tsv.
        
        CRITICAL: Every S1 ID must appear exactly once.
        Only S2/S3 IDs in matched lists. No duplicates.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id in s1_ids:
                matches = decisions.get(s1_id, [])
                # Validate: only S2/S3 IDs
                valid = [m for m in matches if m.startswith("S2-") or m.startswith("S3-")]
                # Remove duplicates preserving order
                seen = set()
                unique = []
                for m in valid:
                    if m not in seen:
                        seen.add(m)
                        unique.append(m)
                f.write(f"{s1_id}\t{','.join(unique)}\n")

        logger.info(f"Saved matching_results.tsv to {output_path} ({len(s1_ids)} entities)")
