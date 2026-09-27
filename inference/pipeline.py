"""
End-to-end inference pipeline for Business Entity Resolution.

Executes candidate generation, candidate compression, feature extraction,
scoring, calibration, entity-level decision, and generates both:
- outputs/candidate_pairs.tsv
- outputs/matching_results.tsv
Followed by automatic validation against official submission rules.
"""

import json
import logging
import os
import pickle
import subprocess
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from features.feature_engineering import FeatureEngine
from models.matching_models import (
    CrossEncoderReranker,
    EntityDecisionEngine,
    FusionModel,
    PairRanker,
    ProbabilityCalibrator
)
from preprocessing.normalize_addresses import normalize_all_addresses
from preprocessing.normalize_names import normalize_all_names
from preprocessing.schema import (
    CandidatePair,
    Record,
    build_records,
    get_country_partition_records,
    load_all_sources,
    load_config,
    load_source_file
)
from retrieval.candidate_generation import CandidateGenerator
from retrieval.embeddings import EmbeddingGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def run_inference(
    config_path: str = "configs/config.yaml",
    sample_test_size: Optional[int] = None,
    batch_size: int = 10000
) -> None:
    """Execute complete inference pipeline on the test dataset.
    
    Args:
        config_path: Path to configuration YAML.
        sample_test_size: Optional subset limit for testing/debugging.
        batch_size: Batch size for candidate generation and feature extraction.
    """
    start_time = time.time()
    config = load_config(config_path)

    base_dir = config["paths"]["base_dir"]
    model_dir = config["paths"]["model_dir"]
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    cand_out_path = os.path.join(base_dir, config["paths"]["candidate_pairs"])
    match_out_path = os.path.join(base_dir, config["paths"]["matching_results"])

    # 1. Load Trained Models & Artifacts
    logger.info("=== STEP 1: Loading Models & Artifacts ===")
    ranker = PairRanker(config)
    ranker.load(os.path.join(model_dir, "pair_ranker.txt"))

    calibrator = ProbabilityCalibrator(config)
    calibrator.load(os.path.join(model_dir, "calibrator.pkl"))

    with open(os.path.join(model_dir, "optimal_threshold.json"), "r") as f:
        thresh_info = json.load(f)
        optimal_threshold = thresh_info.get("threshold", 0.5)
    logger.info(f"Loaded optimal decision threshold: {optimal_threshold:.3f}")

    with open(os.path.join(model_dir, "feature_engine.pkl"), "rb") as f:
        feature_engine: FeatureEngine = pickle.load(f)

    # 2. Ingestion of Test Sources
    # 2. Ingestion of Test Source 1
    logger.info("=== STEP 2: Ingestion of Test Source 1 ===")
    s1_path = os.path.join(base_dir, config["paths"]["test_source1"])
    s2_path = os.path.join(base_dir, config["paths"]["test_source2"])
    s3_path = os.path.join(base_dir, config["paths"]["test_source3"])

    s1_df = load_source_file(s1_path)

    if sample_test_size and sample_test_size < len(s1_df):
        logger.info(f"Running inference on sample of {sample_test_size} S1 records for verification...")
        s1_df = s1_df.head(sample_test_size)

    s1_all_ids = list(s1_df["entity_id"])
    logger.info(f"Loaded {len(s1_df)} Test S1 entities")

    # Check for existing output files to support resume without wasted compute
    processed_s1_ids = set()
    if os.path.exists(cand_out_path) and os.path.exists(match_out_path):
        cand_lines = {}
        with open(cand_out_path, "r", encoding="utf-8") as f:
            header = f.readline()
            if "source1_entity_id" in header:
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if parts and parts[0]:
                        cand_lines[parts[0]] = line.rstrip("\r\n")
        match_lines = {}
        with open(match_out_path, "r", encoding="utf-8") as f:
            header = f.readline()
            if "source1_entity_id" in header:
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if parts and parts[0]:
                        match_lines[parts[0]] = line.rstrip("\r\n")

        common_ids = [sid for sid in cand_lines if sid in match_lines]
        if common_ids:
            processed_s1_ids = set(common_ids)
            # If there was an inconsistent trailing write, rewrite to strictly synchronized state
            if len(common_ids) != len(cand_lines) or len(common_ids) != len(match_lines):
                logger.warning(f"Resynchronizing output files to {len(common_ids):,} common flushed entities...")
                with open(cand_out_path, "w", encoding="utf-8") as f_cand:
                    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
                    for sid in common_ids:
                        f_cand.write(f"{cand_lines[sid]}\n")
                with open(match_out_path, "w", encoding="utf-8") as f_match:
                    f_match.write("source1_entity_id\tmatched_entity_ids\n")
                    for sid in common_ids:
                        f_match.write(f"{match_lines[sid]}\n")
            logger.info(f"RESUME DETECTED: found {len(processed_s1_ids):,} already processed entities on disk. Resuming...")

    if not processed_s1_ids:
        with open(cand_out_path, "w", encoding="utf-8") as f_cand, open(match_out_path, "w", encoding="utf-8") as f_match:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            f_match.write("source1_entity_id\tmatched_entity_ids\n")

    # Detect unique countries dynamically (open-set support)
    s1_df["country_clean"] = s1_df["country"].fillna("").astype(str).str.strip().str.lower()

    # Load embedding and reranker models once outside country loop
    dense_cfg = config.get("retrieval", {}).get("dense_ann", {})
    dense_model_name = dense_cfg.get(
        "model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    logger.info("Initializing global EmbeddingGenerator and CrossEncoder...")
    embedding_gen = None
    if dense_cfg.get("enabled", True):
        embedding_gen = EmbeddingGenerator(model_name=dense_model_name)
    fusion_model = FusionModel(config)
    cross_encoder = None
    if config.get("cross_encoder", {}).get("enabled", True):
        cross_encoder = CrossEncoderReranker(config)
        cross_encoder.load_model()  # Fails fast if model cannot be loaded

    unique_countries = [c for c in s1_df["country_clean"].unique() if c]
    has_empty = (s1_df["country_clean"] == "").any()
    partitions = list(unique_countries)
    if has_empty:
        partitions.append("unknown")
    logger.info(f"Partitions to process: {partitions}")

    s1_by_country: Dict[str, pd.DataFrame] = {}
    for c in partitions:
        c_target = "" if c == "unknown" else c
        s1_by_country[c] = s1_df[s1_df["country_clean"] == c_target]
    del s1_df
    import gc
    gc.collect()

    total_processed_s1 = len(processed_s1_ids)
    total_matches_formed = 0
    total_singletons = 0

    from preprocessing.schema import get_country_partition_records

    for country in partitions:
        logger.info(f"\n==========================================")
        logger.info(f"Processing Country Partition: {country.upper()}")
        logger.info(f"==========================================")

        s1_c = s1_by_country.pop(country, pd.DataFrame())
        if len(s1_c) == 0:
            continue

        if processed_s1_ids:
            s1_c = s1_c[~s1_c["entity_id"].isin(processed_s1_ids)]
            if len(s1_c) == 0:
                logger.info(f"All entities for partition '{country.upper()}' already completed on disk. Skipping.")
                continue

        s1_records = build_records(s1_c)
        del s1_c
        gc.collect()

        # Stream candidate partition records directly from TSV into a single dict without RAM duplication
        cand_records = get_country_partition_records(s2_path, country)
        get_country_partition_records(s3_path, country, existing_dict=cand_records)

        if sample_test_size:
            # Subsample distractors for verification run
            cand_sub = list(cand_records.keys())[:sample_test_size * 10]
            cand_records = {k: cand_records[k] for k in cand_sub}

        logger.info(f"Partition counts: S1={len(s1_records):,}, Candidates={len(cand_records):,}")

        # 3. Normalization & Parsing for this partition
        normalize_all_names(s1_records, config)
        normalize_all_names(cand_records, config)
        normalize_all_addresses(s1_records, config)
        normalize_all_addresses(cand_records, config)

        # 4. Multi-Channel Candidate Generation
        cand_gen = CandidateGenerator(config, embedding_gen=embedding_gen)
        cand_gen.setup(s1_records, candidate_records=cand_records, build_dense_index=False)

        # Clear embeddings cache to use direct CandidatePair semantic scores
        feature_engine.set_embeddings_cache(None)

        # Process S1 entities in batches
        s1_ids_list = list(s1_records.keys())
        sub_batch = 1000

        for b_start in range(0, len(s1_ids_list), sub_batch):
            b_ids = s1_ids_list[b_start:b_start + sub_batch]
            b_s1_records = {rid: s1_records[rid] for rid in b_ids}

            # Generate candidate pairs with memory-bounded dense encoding
            b_candidates = cand_gen.generate_all_candidates(
                b_s1_records, dense_batch_size=2048, batch_log_every=sub_batch
            )

            # Flatten pairs for scoring
            flat_pairs = []
            for s1_id, clist in b_candidates.items():
                for cp in clist:
                    flat_pairs.append((s1_id, cp.candidate_id, cp))

            # Score candidates
            entity_scores: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in b_ids}

            if flat_pairs:
                pf_list = []
                for s1_id, cand_id, pair_meta in flat_pairs:
                    s1_rec = s1_records.get(s1_id)
                    c_rec = cand_records.get(cand_id)
                    if s1_rec and c_rec:
                        pf = feature_engine.compute_pair_features(s1_rec, c_rec, pair_meta)
                        pf_list.append(pf)

                if pf_list:
                    X_b, b_s1_names, b_cand_names, _ = feature_engine.features_to_arrays(pf_list)
                    raw_preds = ranker.predict(X_b)

                    # Vectorized probability calibration across entire batch
                    calibrated_preds = calibrator.calibrate(raw_preds)

                    for s1_id, cand_id, score in zip(b_s1_names, b_cand_names, calibrated_preds):
                        entity_scores[s1_id].append((cand_id, float(score)))

            # Entity-level decision
            decision_engine = EntityDecisionEngine(threshold=optimal_threshold)
            b_decisions = decision_engine.decide_all(entity_scores)

            # Stream results to files
            with open(cand_out_path, "a", encoding="utf-8") as f_cand, open(match_out_path, "a", encoding="utf-8") as f_match:
                for s1_id in b_ids:
                    # Candidates
                    c_list = [c.candidate_id for c in b_candidates.get(s1_id, [])]
                    seen_c = set()
                    unique_c = [cid for cid in c_list if not (cid in seen_c or seen_c.add(cid))]
                    f_cand.write(f"{s1_id}\t{','.join(unique_c)}\n")

                    # Matches
                    m_list = b_decisions.get(s1_id, [])
                    valid_m = [m for m in m_list if m.startswith("S2-") or m.startswith("S3-")]
                    seen_m = set()
                    unique_m = [mid for mid in valid_m if not (mid in seen_m or seen_m.add(mid))]
                    f_match.write(f"{s1_id}\t{','.join(unique_m)}\n")

                    total_matches_formed += len(unique_m)
                    if not unique_m:
                        total_singletons += 1

            total_processed_s1 += len(b_ids)
            if total_processed_s1 % 5000 == 0 or b_start + sub_batch >= len(s1_ids_list):
                logger.info(f"Progress ({country.upper()}): {total_processed_s1:,} / {len(s1_all_ids):,} S1 entities done...")

            del b_s1_records, b_candidates, flat_pairs, entity_scores
            if 'pf_list' in locals():
                del pf_list
            if 'X_b' in locals():
                del X_b, raw_preds, batch_cand_scores
            gc.collect()

        # Free partition memory
        del s1_records, cand_records, cand_gen
        gc.collect()

    logger.info("==================================================")
    logger.info("=== INFERENCE PREDICTION SUMMARY ===")
    logger.info(f"  Total S1 Entities:     {total_processed_s1:,}")
    logger.info(f"  Singletons (0 matches): {total_singletons:,} ({total_singletons/max(total_processed_s1,1)*100:.1f}%)")
    logger.info(f"  Total Matches Formed:  {total_matches_formed:,}")
    logger.info(f"  Candidate file:        {cand_out_path}")
    logger.info(f"  Matching file:         {match_out_path}")
    logger.info("==================================================")

    # 9. Automatic Validation
    validator_path = os.path.join(base_dir, "utils/validate_submission.py")
    test_dir_rel = config.get("paths", {}).get("test_dir")
    test_dir_path = os.path.join(base_dir, test_dir_rel) if test_dir_rel else None

    if os.path.exists(validator_path) and test_dir_path and os.path.exists(test_dir_path):
        logger.info("=== STEP 9: Running Official Submission Validator ===")
        cmd = [
            "python3", validator_path,
            "--matching", match_out_path,
            "--candidate", cand_out_path,
            "--test-dir", test_dir_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("VALIDATOR ISSUES DETECTED:")
            print(result.stderr)
        else:
            logger.info("VALIDATION STATUS: PASS (exit 0) — Submission is 100% compliant!")

    elapsed = time.time() - start_time
    logger.info(f"Inference pipeline completed in {elapsed:.1f}s")


if __name__ == "__main__":
    run_inference()
