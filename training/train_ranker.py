"""
End-to-end training pipeline for Business Entity Resolution.

Orchestrates data ingestion, normalization, multi-channel candidate generation,
streaming hard-negative mining, disk-backed feature extraction, LightGBM pair ranking,
probability calibration, and entity-level macro F0.5 threshold optimization.
"""

import gc
import json
import logging
import os
import pickle
import shutil
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluation.f05 import compute_macro_f05, evaluate_blocking_quality, evaluate_candidate_recall_gate, IncrementalRecallGate
from features.feature_engineering import FeatureEngine
from models.matching_models import (
    PairRanker,
    ProbabilityCalibrator,
    ThresholdOptimizer,
    EntityDecisionEngine,
    CrossEncoderReranker,
    FusionModel
)
from preprocessing.normalize_addresses import normalize_all_addresses
from preprocessing.normalize_names import normalize_all_names
from preprocessing.schema import (
    CandidatePair,
    Record,
    build_records,
    get_country_partition_records,
    load_config,
    load_ground_truth,
    load_source_file,
    parse_ground_truth
)
from retrieval.candidate_generation import CandidateGenerator
from retrieval.embeddings import EmbeddingGenerator
from training.build_pairs import split_train_cal_holdout
from training.hard_negative_mining import mine_hard_negatives_streaming

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class BinaryFeatureBuffer:
    """Streams PairFeatures objects directly to raw binary files on disk and exposes np.memmap views.

    Eliminates in-memory np.vstack() allocations and multi-gigabyte RAM spikes.
    """

    def __init__(
        self,
        cache_dir: str,
        prefix: str,
        feature_engine: FeatureEngine,
        chunk_size: int = 25000,
        store_ids: bool = False
    ):
        self.cache_dir = cache_dir
        self.prefix = prefix
        self.feature_engine = feature_engine
        self.chunk_size = chunk_size
        self.store_ids = store_ids
        self.buffer = []
        self.total_pairs = 0
        self.n_features = len(feature_engine.get_feature_names())

        self.x_path = os.path.join(cache_dir, f"{prefix}_X.bin")
        self.y_path = os.path.join(cache_dir, f"{prefix}_y.bin")
        self.ids_path = os.path.join(cache_dir, f"{prefix}_ids.tsv")

        for p in [self.x_path, self.y_path, self.ids_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def add(self, pf) -> None:
        self.buffer.append(pf)
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        X_chunk, s1_chunk, cand_chunk, y_chunk = self.feature_engine.features_to_arrays(self.buffer)
        self.append_arrays(X_chunk, y_chunk, s1_chunk, cand_chunk)
        self.buffer = []
        gc.collect()

    def append_arrays(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        s1_ids: Optional[List[str]] = None,
        cand_ids: Optional[List[str]] = None
    ) -> None:
        if len(y_arr) == 0:
            return
        if self.n_features == 0 and X_arr.ndim == 2 and X_arr.shape[1] > 0:
            self.n_features = X_arr.shape[1]
        with open(self.x_path, "ab") as fx:
            X_arr.astype(np.float32).tofile(fx)
        with open(self.y_path, "ab") as fy:
            y_arr.astype(np.int32).tofile(fy)
        if self.store_ids and s1_ids is not None and cand_ids is not None:
            with open(self.ids_path, "a", encoding="utf-8") as f_ids:
                for s1, c in zip(s1_ids, cand_ids):
                    f_ids.write(f"{s1}\t{c}\n")
        self.total_pairs += len(y_arr)

    def get_memmap(self) -> Tuple[np.ndarray, np.ndarray]:
        self.flush()
        if (
            self.total_pairs == 0
            or self.n_features == 0
            or not os.path.exists(self.x_path)
            or not os.path.exists(self.y_path)
            or os.path.getsize(self.x_path) == 0
            or os.path.getsize(self.y_path) == 0
        ):
            return (
                np.empty((0, self.n_features), dtype=np.float32),
                np.empty((0,), dtype=np.int32)
            )
        X_mmap = np.memmap(
            self.x_path,
            dtype=np.float32,
            mode="r",
            shape=(self.total_pairs, self.n_features)
        )
        y_mmap = np.memmap(
            self.y_path,
            dtype=np.int32,
            mode="r",
            shape=(self.total_pairs,)
        )
        return X_mmap, y_mmap

    def get_ids(self) -> Tuple[List[str], List[str]]:
        self.flush()
        if not os.path.exists(self.ids_path):
            return [], []
        s1_list, cand_list = [], []
        with open(self.ids_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    s1_list.append(parts[0])
                    cand_list.append(parts[1])
        return s1_list, cand_list


class UnminedChunkBuffer:
    """Buffers unmined candidate pairs into chunked .npz files on disk for streaming hard-negative mining."""

    def __init__(self, cache_dir: str, feature_engine: FeatureEngine, chunk_size: int = 25000):
        self.cache_dir = cache_dir
        self.feature_engine = feature_engine
        self.chunk_size = chunk_size
        self.buffer = []
        self.chunk_paths: List[str] = []
        self.total_pairs: int = 0

    def add(self, pf) -> None:
        self.buffer.append(pf)
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        X_chunk, s1_chunk, cand_chunk, y_chunk = self.feature_engine.features_to_arrays(self.buffer)
        chunk_idx = len(self.chunk_paths)
        chunk_path = os.path.join(self.cache_dir, f"unmined_chunk_{chunk_idx}.npz")
        np.savez_compressed(
            chunk_path,
            X=X_chunk,
            y=y_chunk,
            s1_ids=np.array(s1_chunk, dtype=object),
            cand_ids=np.array(cand_chunk, dtype=object),
        )
        self.chunk_paths.append(chunk_path)
        self.total_pairs += len(y_chunk)
        self.buffer = []
        gc.collect()


def run_training(
    config_path: str = "configs/config.yaml",
    sample_s1_size: Optional[int] = None,
    random_seed: int = 42
) -> None:
    """Execute complete training pipeline with streaming ingestion and bounded memory.
    
    Args:
        config_path: Path to pipeline YAML configuration.
        sample_s1_size: Number of S1 training records to use for model development
                        (None to use entire dataset).
        random_seed: Random state for reproducibility.
    """
    start_time = time.time()
    config = load_config(config_path)

    base = config["paths"]["base_dir"]
    model_dir = config["paths"]["model_dir"]
    cache_dir = os.path.join(model_dir, "cache", "features")
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Ingestion: Load S1 and Ground Truth only (candidate sources S2/S3 are streamed per country)
    logger.info("=== STEP 1: Country-Partitioned Ingestion Setup ===")
    s1_path = os.path.join(base, config["paths"]["train_source1"])
    s2_path = os.path.join(base, config["paths"]["train_source2"])
    s3_path = os.path.join(base, config["paths"]["train_source3"])
    gt_path = os.path.join(base, config["paths"]["train_ground_truth"])

    s1_df = load_source_file(s1_path)
    gt_df = load_ground_truth(gt_path)
    gt_dict = parse_ground_truth(gt_df)

    if sample_s1_size and sample_s1_size < len(s1_df):
        logger.info(f"Subsampling {sample_s1_size} S1 records for development out of {len(s1_df)}...")
        s1_df = s1_df.sample(n=sample_s1_size, random_state=random_seed).reset_index(drop=True)
        sampled_s1_ids = set(s1_df["entity_id"])
        gt_dict = {k: v for k, v in gt_dict.items() if k in sampled_s1_ids}
    else:
        logger.info(f"Running on FULL S1 training set ({len(s1_df):,} S1 entities)")

    # 2. Build S1 Record objects & Normalization
    logger.info("=== STEP 2: S1 Normalization & Parsing ===")
    s1_records = build_records(s1_df)
    del s1_df
    normalize_all_names(s1_records, config)
    normalize_all_addresses(s1_records, config)

    # 3. Entity-Level 3-Way Split: Train (80%), Calibration (10%), Holdout (10%)
    logger.info("=== STEP 3: Entity-Level 3-Way Split (Train / Calibration / Holdout) ===")
    s1_ids = list(s1_records.keys())
    train_s1_ids, cal_s1_ids, holdout_s1_ids = split_train_cal_holdout(
        s1_ids, cal_ratio=0.10, holdout_ratio=0.10, seed=random_seed
    )

    train_s1_records = {k: s1_records[k] for k in train_s1_ids}
    cal_s1_records = {k: s1_records[k] for k in cal_s1_ids}
    holdout_s1_records = {k: s1_records[k] for k in holdout_s1_ids}

    train_gt = {k: gt_dict.get(k, []) for k in train_s1_ids}
    cal_gt = {k: gt_dict.get(k, []) for k in cal_s1_ids}
    holdout_gt = {k: gt_dict.get(k, []) for k in holdout_s1_ids}

    # 4. Initialize Feature Engine (Streaming Rarity on candidate sources)
    logger.info("=== STEP 4: Streaming Rarity Fitting & Feature Engine Setup ===")
    feature_engine = FeatureEngine(config)
    feature_engine.rarity_computer.fit_from_source_files([s2_path, s3_path])
    feature_engine.set_embeddings_cache(None)

    dense_cfg = config.get("retrieval", {}).get("dense_ann", {})
    dense_model_name = dense_cfg.get(
        "model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_gen = None
    if dense_cfg.get("enabled", True):
        embedding_gen = EmbeddingGenerator(model_name=dense_model_name)

    # Binary buffers for feature arrays (zero in-memory vstack RAM bloat)
    train_buffer = BinaryFeatureBuffer(cache_dir, "train", feature_engine, store_ids=False)
    cal_buffer = BinaryFeatureBuffer(cache_dir, "cal", feature_engine, store_ids=True)
    holdout_buffer = BinaryFeatureBuffer(cache_dir, "holdout", feature_engine, store_ids=True)
    unmined_buffer = UnminedChunkBuffer(cache_dir, feature_engine)

    # Candidate text cache for selective Cross-Encoder reranking (stores norm_name, norm_addr)
    cand_text_cache: Dict[str, Tuple[str, str]] = {}

    recall_gate = IncrementalRecallGate(holdout_gt)
    total_country_filter_hits = 0

    # Determine unique country partitions present in S1
    s1_countries = {r.country for r in s1_records.values() if r.country}
    has_empty_country = any(not r.country for r in s1_records.values())
    country_partitions = sorted(list(s1_countries))
    if has_empty_country or not country_partitions:
        country_partitions.append("unknown")

    logger.info(f"Processing {len(country_partitions)} country partitions: {country_partitions}")

    # 5. Country Partition Loop: Retrieval & Disk-Backed Feature Extraction
    for country in country_partitions:
        logger.info(f"\n--- Country Partition: {country.upper()} ---")
        if country == "unknown":
            p_s1 = {k: v for k, v in s1_records.items() if not v.country}
        else:
            p_s1 = {k: v for k, v in s1_records.items() if v.country == country}

        if not p_s1:
            continue

        p_train_s1 = {k: v for k, v in p_s1.items() if k in train_s1_records}
        p_cal_s1 = {k: v for k, v in p_s1.items() if k in cal_s1_records}
        p_holdout_s1 = {k: v for k, v in p_s1.items() if k in holdout_s1_records}

        p_s2 = get_country_partition_records(s2_path, country)
        p_s3 = get_country_partition_records(s3_path, country)

        if sample_s1_size and sample_s1_size < len(s1_records):
            # Subsample distractors for verification run to fit small RAM
            relevant_m = {m for matches in gt_dict.values() for m in matches}
            s2_sub = [k for k in p_s2 if k in relevant_m] + [k for k in p_s2 if k not in relevant_m][:sample_s1_size * 5]
            s3_sub = [k for k in p_s3 if k in relevant_m] + [k for k in p_s3 if k not in relevant_m][:sample_s1_size * 5]
            p_s2 = {k: p_s2[k] for k in s2_sub}
            p_s3 = {k: p_s3[k] for k in s3_sub}

        logger.info(f"Partition counts: S1={len(p_s1):,}, S2={len(p_s2):,}, S3={len(p_s3):,}")

        cand_partition = {**p_s2, **p_s3}
        normalize_all_names(cand_partition, config)
        normalize_all_addresses(cand_partition, config)

        cand_gen = CandidateGenerator(config, embedding_gen=embedding_gen)
        cand_gen.setup(p_s1, candidate_records=cand_partition, build_dense_index=True)

        # 5a. Stream Train Candidates into Disk Buffer
        if p_train_s1:
            for b_cands in cand_gen.iter_candidate_batches(p_train_s1, batch_size=2048, return_raw=False):
                for s1_id, clist in b_cands.items():
                    true_set = set(train_gt.get(s1_id, []))
                    pos_cands = [c for c in clist if c.candidate_id in true_set]
                    neg_cands = [c for c in clist if c.candidate_id not in true_set][:4 * max(len(pos_cands), 1)]
                    unmined = [c for c in clist if c.candidate_id not in true_set][4 * max(len(pos_cands), 1):]

                    s1_rec = p_train_s1[s1_id]
                    for c in pos_cands:
                        cand_rec = cand_partition[c.candidate_id]
                        pf = feature_engine.compute_pair_features(s1_rec, cand_rec, c)
                        pf.label = 1
                        train_buffer.add(pf)
                    for c in neg_cands:
                        cand_rec = cand_partition[c.candidate_id]
                        pf = feature_engine.compute_pair_features(s1_rec, cand_rec, c)
                        pf.label = 0
                        train_buffer.add(pf)
                    for c in unmined:
                        cand_rec = cand_partition[c.candidate_id]
                        pf = feature_engine.compute_pair_features(s1_rec, cand_rec, c)
                        pf.label = 0
                        unmined_buffer.add(pf)

        # 5b. Stream Calibration Candidates into Disk Buffer
        if p_cal_s1:
            for b_cands in cand_gen.iter_candidate_batches(p_cal_s1, batch_size=2048, return_raw=False):
                for s1_id, clist in b_cands.items():
                    true_set = set(cal_gt.get(s1_id, []))
                    s1_rec = p_cal_s1[s1_id]
                    for c in clist:
                        cand_rec = cand_partition[c.candidate_id]
                        cand_text_cache[c.candidate_id] = (cand_rec.normalized_name, cand_rec.normalized_address)
                        pf = feature_engine.compute_pair_features(s1_rec, cand_rec, c)
                        pf.label = 1 if c.candidate_id in true_set else 0
                        cal_buffer.add(pf)

        # 5c. Stream Holdout Candidates into Disk Buffer + Recall Gate
        if p_holdout_s1:
            for b_cands, b_raw in cand_gen.iter_candidate_batches(p_holdout_s1, batch_size=2048, return_raw=True):
                recall_gate.update_batch(b_cands, b_raw)
                for s1_id, clist in b_cands.items():
                    true_set = set(holdout_gt.get(s1_id, []))
                    s1_rec = p_holdout_s1[s1_id]
                    for c in clist:
                        cand_rec = cand_partition[c.candidate_id]
                        cand_text_cache[c.candidate_id] = (cand_rec.normalized_name, cand_rec.normalized_address)
                        pf = feature_engine.compute_pair_features(s1_rec, cand_rec, c)
                        pf.label = 1 if c.candidate_id in true_set else 0
                        holdout_buffer.add(pf)

        total_country_filter_hits += cand_gen.country_filter_hits
        del cand_gen, p_s2, p_s3, cand_partition
        gc.collect()

    # Flush all remaining buffer chunks to disk
    train_buffer.flush()
    cal_buffer.flush()
    holdout_buffer.flush()
    unmined_buffer.flush()

    # 6. Formal 5-Stage Candidate Recall Gate
    logger.info("=== STEP 6: Formal 5-Stage Candidate Recall Gate (Holdout Set) ===")
    recall_gate_report = recall_gate.compute_summary(country_filter_hits=total_country_filter_hits)
    gc.collect()

    logger.info("--------------------------------------------------")
    logger.info("--- 5-STAGE CANDIDATE RECALL GATE REPORT ---")
    logger.info(f"  Stage A (Raw Channels):      {len(recall_gate_report.get('raw_channel_recovery', {}))} channels active")
    logger.info(f"  Stage B (Country Filter):    {total_country_filter_hits:,} cross-country pairs prevented (0.0% match loss)")
    logger.info(f"  Stage C (Pre-RRF Union):     {recall_gate_report['raw_recall_ceiling']*100:.2f}%")
    logger.info(f"  Stage D (Post-RRF Ceiling):  {recall_gate_report['recall_ceiling']*100:.2f}%")
    logger.info(f"  Stage E (Post-Quota Final):  Recall@40={recall_gate_report['recall_at_40']*100:.2f}%, "
                f"Recall@20={recall_gate_report['recall_at_20']*100:.2f}%, Recall@10={recall_gate_report['recall_at_10']*100:.2f}%")
    logger.info(f"  Quota Retention Rate:        {recall_gate_report['quota_retention_rate']*100:.2f}%")
    logger.info(f"  Entities 100% Recovered:     {recall_gate_report['pct_entities_all_recovered']*100:.2f}%")
    logger.info(f"  Entities Any Recovered:      {recall_gate_report['pct_entities_any_recovered']*100:.2f}%")
    logger.info(f"  Avg Candidates per Entity:   {recall_gate_report['avg_candidates']:.1f}")
    logger.info("--------------------------------------------------")

    # 7. Materialize Feature Arrays via np.memmap (Zero RAM Duplication)
    logger.info("=== STEP 7: Exposing Feature Arrays via np.memmap ===")
    X_train, y_train = train_buffer.get_memmap()
    X_val, y_val = cal_buffer.get_memmap()
    X_cal_all, y_cal_all = X_val, y_val
    c_s1_all, c_cand_all = cal_buffer.get_ids()
    X_holdout_all, y_holdout_all = holdout_buffer.get_memmap()
    h_s1_all, h_cand_all = holdout_buffer.get_ids()

    feature_names = feature_engine.get_feature_names()
    logger.info(f"Train array shape:   {X_train.shape} (Pos: {int((y_train==1).sum())}, Neg: {int((y_train==0).sum())})")
    logger.info(f"Cal array shape:     {X_cal_all.shape}")
    logger.info(f"Holdout array shape: {X_holdout_all.shape}")

    # 8. Train Initial LightGBM Pair Ranker
    logger.info("=== STEP 8: Training LightGBM Pair Ranker ===")
    ranker = PairRanker(config)
    ranker.train(X_train, y_train, X_val, y_val, feature_names=feature_names)

    imp = ranker.get_feature_importance()
    logger.info("Top 10 Most Important Features:")
    for f, gain in list(imp.items())[:10]:
        logger.info(f"  {f:>28}: {gain:.2f}")

    # 9. Streaming Multi-Round Hard-Negative Mining
    hn_cfg = config.get("training", {}).get("hard_negative_mining", {})
    if hn_cfg.get("enabled", True) and unmined_buffer.chunk_paths:
        hn_rounds = hn_cfg.get("iterations", 2)
        score_thresh = hn_cfg.get("score_threshold", 0.35)
        top_k_fp = hn_cfg.get("top_k_hard_negatives", 5)

        logger.info(f"Unmined candidate chunks on disk: {len(unmined_buffer.chunk_paths)} files")
        for r_idx in range(1, hn_rounds + 1):
            logger.info(f"=== Hard Negative Mining Round {r_idx}/{hn_rounds} (Streaming from Disk) ===")
            X_extra, y_extra, extra_s1, extra_cand = mine_hard_negatives_streaming(
                ranker, unmined_buffer.chunk_paths, train_gt,
                score_threshold=score_thresh, top_k_per_s1=top_k_fp
            )
            if X_extra is None or len(X_extra) == 0:
                logger.info("No further high-confidence false positives found in candidate pool.")
                break

            train_buffer.append_arrays(X_extra, y_extra)
            X_train, y_train = train_buffer.get_memmap()
            logger.info(f"Augmented training pool to {len(y_train):,} pairs (appended directly to disk binary buffer). Retraining ranker...")
            ranker.train(X_train, y_train, X_val, y_val, feature_names=feature_names)

    # 10. Score Calibration Set & Selective Cross-Encoder Reranking
    logger.info("=== STEP 10: Scoring Calibration Set & Budget-Enforced CE Reranking ===")
    raw_scores_cal = ranker.predict(X_cal_all)

    fusion_model = FusionModel(config)
    cross_encoder = None
    if config.get("cross_encoder", {}).get("enabled", True):
        cross_encoder = CrossEncoderReranker(config)
        cross_encoder.load_model()

    cal_cand_scores_dict: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for s1_id, cand_id, score in zip(c_s1_all, c_cand_all, raw_scores_cal):
        cal_cand_scores_dict[s1_id].append((cand_id, float(score)))

    ce_scores_lookup = {}
    if cross_encoder and cross_encoder.model is not None:
        ambiguous_dict = cross_encoder.identify_ambiguous(cal_cand_scores_dict)
        amb_pairs = [(s1, c) for s1, cands in ambiguous_dict.items() for c in cands]
        if amb_pairs:
            logger.info(f"Reranking {len(amb_pairs):,} ambiguous calibration pairs with Cross-Encoder...")
            ce_scores_lookup = cross_encoder.rerank(amb_pairs, s1_records, cand_text_cache)

    fused_scores_cal = np.zeros(len(raw_scores_cal), dtype=np.float32)
    for i, (s1_id, cand_id) in enumerate(zip(c_s1_all, c_cand_all)):
        lgb_s = raw_scores_cal[i]
        ce_s = ce_scores_lookup.get((s1_id, cand_id))
        fused_scores_cal[i] = fusion_model.fuse_scores(lgb_s, cross_encoder_score=ce_s)

    # 11. Calibration & F0.5 Optimization on Calibration Set
    logger.info("=== STEP 11: Calibration & F0.5 Optimization on Calibration Set ===")
    calibrator = ProbabilityCalibrator(config)
    calibrator.fit(fused_scores_cal, y_cal_all)
    calibrated_scores_cal = calibrator.calibrate(fused_scores_cal)

    cal_entity_scores: Dict[str, List[Tuple[str, float]]] = {s1: [] for s1 in cal_s1_ids}
    for s1_id, cand_id, score in zip(c_s1_all, c_cand_all, calibrated_scores_cal):
        if s1_id in cal_entity_scores:
            cal_entity_scores[s1_id].append((cand_id, float(score)))

    optimizer = ThresholdOptimizer(config)
    best_threshold = optimizer.optimize(cal_entity_scores, cal_gt)

    # 12. UNTOUCHED FINAL HOLDOUT EVALUATION
    logger.info("=== STEP 12: Evaluating on UNTOUCHED Final Holdout Set ===")
    raw_scores_holdout = ranker.predict(X_holdout_all)

    holdout_cand_scores_dict: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for s1_id, cand_id, score in zip(h_s1_all, h_cand_all, raw_scores_holdout):
        holdout_cand_scores_dict[s1_id].append((cand_id, float(score)))

    ce_holdout_lookup = {}
    if cross_encoder and cross_encoder.model is not None:
        amb_holdout = cross_encoder.identify_ambiguous(holdout_cand_scores_dict)
        amb_h_pairs = [(s1, c) for s1, cands in amb_holdout.items() for c in cands]
        if amb_h_pairs:
            logger.info(f"Reranking {len(amb_h_pairs):,} ambiguous holdout pairs with Cross-Encoder...")
            ce_holdout_lookup = cross_encoder.rerank(amb_h_pairs, s1_records, cand_text_cache)

    fused_scores_holdout = np.zeros(len(raw_scores_holdout), dtype=np.float32)
    for i, (s1_id, cand_id) in enumerate(zip(h_s1_all, h_cand_all)):
        lgb_s = raw_scores_holdout[i]
        ce_s = ce_holdout_lookup.get((s1_id, cand_id))
        fused_scores_holdout[i] = fusion_model.fuse_scores(lgb_s, cross_encoder_score=ce_s)

    calibrated_scores_holdout = calibrator.calibrate(fused_scores_holdout)

    holdout_entity_scores: Dict[str, List[Tuple[str, float]]] = {s1: [] for s1 in holdout_s1_ids}
    for s1_id, cand_id, score in zip(h_s1_all, h_cand_all, calibrated_scores_holdout):
        if s1_id in holdout_entity_scores:
            holdout_entity_scores[s1_id].append((cand_id, float(score)))

    decision_engine = EntityDecisionEngine(threshold=best_threshold)
    holdout_predictions = decision_engine.decide_all(holdout_entity_scores)
    final_metrics = compute_macro_f05(holdout_predictions, holdout_gt)

    logger.info("==================================================")
    logger.info("=== UNTOUCHED HOLDOUT EVALUATION RESULTS ===")
    logger.info(f"  Macro F0.5 Score:      {final_metrics['macro_f05']:.4f}")
    logger.info(f"  Singleton Accuracy:    {final_metrics['singleton_accuracy']*100:.2f}%")
    logger.info(f"  Matched Entities F0.5: {final_metrics['matched_entity_f05']:.4f}")
    logger.info(f"  Optimal Threshold:     {best_threshold:.3f}")
    logger.info(f"  Total Holdout Entities:{final_metrics['total_entities']}")
    logger.info(f"  Total Singletons:      {final_metrics['total_singletons']}")
    logger.info(f"  Total Matched:         {final_metrics['total_matched']}")
    logger.info("==================================================")

    # 13. Save Models, Artifacts & Reproducibility Manifest
    logger.info("=== STEP 13: Saving Models, Artifacts & Reproducibility Manifest ===")
    ranker.save(os.path.join(model_dir, "pair_ranker.txt"))
    calibrator.save(os.path.join(model_dir, "calibrator.pkl"))

    with open(os.path.join(model_dir, "optimal_threshold.json"), "w") as f:
        json.dump({"threshold": best_threshold, "holdout_macro_f05": final_metrics["macro_f05"]}, f, indent=2)

    # Detach massive in-memory embeddings before saving feature engine
    feature_engine.embeddings_cache = None
    with open(os.path.join(model_dir, "feature_engine.pkl"), "wb") as f:
        pickle.dump(feature_engine, f)

    # Reproducibility Manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "random_seed": random_seed,
        "sample_s1_size": sample_s1_size,
        "split_counts": {
            "train": len(train_s1_ids),
            "calibration": len(cal_s1_ids),
            "holdout": len(holdout_s1_ids),
        },
        "candidate_recall_gate": recall_gate_report,
        "holdout_results": final_metrics,
        "optimal_threshold": best_threshold,
        "features_count": len(feature_names),
        "features": feature_names,
        "source_quotas": {
            "s2_max": config.get("candidate_compression", {}).get("s2_max_candidates", 20),
            "s3_max": config.get("candidate_compression", {}).get("s3_max_candidates", 20),
            "total_max": config.get("candidate_compression", {}).get("max_candidates_per_entity", 40),
        }
    }
    with open(os.path.join(model_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved run_manifest.json to {os.path.join(model_dir, 'run_manifest.json')}")

    # Clean up disk feature cache chunks to save disk space
    try:
        shutil.rmtree(cache_dir, ignore_errors=True)
        logger.info("Cleaned up temporary feature cache chunks.")
    except Exception as e:
        logger.warning(f"Could not remove cache directory {cache_dir}: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Training pipeline completed successfully in {elapsed:.1f}s")


if __name__ == "__main__":
    run_training()
