"""
End-to-end integration test verifying run_training and run_inference on synthetic partitions.
"""

import json
import os
import shutil
import tempfile
import unittest
import yaml
import pandas as pd

from inference.pipeline import run_inference
from training.train_ranker import run_training


class TestEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")

        # Create mini datasets
        s1_data = []
        s2_data = []
        s3_data = []
        gt_data = []

        countries = ["us", "in", ""]
        for i in range(30):
            c = countries[i % len(countries)]
            s1_id = f"S1-{i:03d}"
            name = f"Company Alpha {i} Logistics Ltd"
            addr = f"{100 + i} Main Street, Area {i}"
            s1_data.append({"entity_id": s1_id, "business_name": name, "business_address": addr, "country": c})

            s2_id = f"S2-{i:03d}"
            s2_data.append({"entity_id": s2_id, "business_name": name, "business_address": addr, "country": c})

            s3_id = f"S3-{i:03d}"
            s3_data.append({"entity_id": s3_id, "business_name": name, "business_address": addr, "country": c})

            # Matches
            gt_data.append({"source1_entity_id": s1_id, "matched_entity_ids": f"{s2_id},{s3_id}" if i % 2 == 0 else ""})

        # Add some distractors
        for j in range(10):
            s2_data.append({"entity_id": f"S2-dist-{j}", "business_name": f"Distractor Business {j}", "business_address": f"{900+j} Other Road", "country": "us"})
            s3_data.append({"entity_id": f"S3-dist-{j}", "business_name": f"Distractor Business {j}", "business_address": f"{900+j} Other Road", "country": "in"})

        train_s1 = os.path.join(self.temp_dir, "train_s1.tsv")
        train_s2 = os.path.join(self.temp_dir, "train_s2.tsv")
        train_s3 = os.path.join(self.temp_dir, "train_s3.tsv")
        train_gt = os.path.join(self.temp_dir, "train_gt.tsv")

        pd.DataFrame(s1_data).to_csv(train_s1, sep="\t", index=False)
        pd.DataFrame(s2_data).to_csv(train_s2, sep="\t", index=False)
        pd.DataFrame(s3_data).to_csv(train_s3, sep="\t", index=False)
        pd.DataFrame(gt_data).to_csv(train_gt, sep="\t", index=False)

        # Test files
        test_s1 = os.path.join(self.temp_dir, "test_s1.tsv")
        test_s2 = os.path.join(self.temp_dir, "test_s2.tsv")
        test_s3 = os.path.join(self.temp_dir, "test_s3.tsv")
        pd.DataFrame(s1_data[:10]).to_csv(test_s1, sep="\t", index=False)
        pd.DataFrame(s2_data).to_csv(test_s2, sep="\t", index=False)
        pd.DataFrame(s3_data).to_csv(test_s3, sep="\t", index=False)

        model_dir = os.path.join(self.temp_dir, "models")
        out_dir = os.path.join(self.temp_dir, "outputs")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        config = {
            "paths": {
                "base_dir": ".",
                "model_dir": model_dir,
                "output_dir": out_dir,
                "train_source1": train_s1,
                "train_source2": train_s2,
                "train_source3": train_s3,
                "train_ground_truth": train_gt,
                "test_source1": test_s1,
                "test_source2": test_s2,
                "test_source3": test_s3,
                "matching_results": os.path.join(out_dir, "matching_results.tsv"),
                "candidate_pairs": os.path.join(out_dir, "candidate_pairs.tsv"),
            },
            "normalization": {
                "name": {"char_ngram_range": [3, 4]},
                "address": {},
            },
            "retrieval": {
                "filter_country_mismatch": True,
                "exact_blocking": {"enabled": True},
                "tfidf": {
                    "enabled": True,
                    "name_char_ngram_range": [3, 4],
                    "name_word_ngram_range": [1, 2],
                    "address_char_ngram_range": [3, 4],
                    "max_features": 5000,
                    "top_k": 10,
                    "tfidf_query_warn_ms": 100.0,
                },
                "bm25": {"enabled": True, "k1": 1.5, "b": 0.75, "top_k": 10},
                "dense_ann": {
                    "enabled": False,  # disable dense model download in unit test for speed
                    "dimension": 384,
                    "index_type": "ivfpq",
                    "nlist": 32,
                    "nprobe": 4,
                    "pq_m": 32,
                    "pq_nbits": 8,
                    "training_sample_size": 1000,
                }
            },
            "candidate_compression": {
                "enabled": True,
                "rrf": {"enabled": True, "k": 60},
                "s2_max_candidates": 20,
                "s3_max_candidates": 20,
                "max_candidates_per_entity": 40,
            },
            "features": {"lexical": {}, "semantic": {}, "structural": {}, "rarity": {}, "retrieval": {}},
            "pair_model": {
                "type": "lightgbm",
                "params": {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "boosting_type": "gbdt",
                    "num_leaves": 15,
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "n_estimators": 50,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_samples": 2,
                    "verbose": -1,
                },
                "early_stopping_rounds": 10,
            },
            "cross_encoder": {
                "enabled": False,  # disable CE in fast test
                "max_candidates_for_rerank": 5,
                "max_rerank_pairs_per_batch": 100,
                "max_rerank_pairs_total": 1000,
                "max_rerank_fraction_per_batch": 0.1,
            },
            "fusion_model": {
                "type": "weighted_average",
                "pair_ranker_weight": 0.4,
                "cross_encoder_weight": 0.6,
            },
            "calibration": {"method": "isotonic"},
            "decision": {
                "threshold_search": {"start": 0.1, "stop": 0.9, "step": 0.1},
                "default_threshold": 0.5,
            },
            "training": {
                "validation_split": 0.2,
                "random_seed": 42,
                "hard_negative_mining": {
                    "enabled": True,
                    "iterations": 1,
                    "score_threshold": 0.2,
                    "top_k_hard_negatives": 2,
                }
            },
            "logging": {"level": "INFO"},
        }

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_train_and_infer(self):
        """Run training then inference on synthetic data and verify outputs."""
        # 1. Train
        run_training(config_path=self.config_path, random_seed=42)

        # Verify saved artifacts
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)
        model_dir = cfg["paths"]["model_dir"]
        self.assertTrue(os.path.exists(os.path.join(model_dir, "pair_ranker.txt")))
        self.assertTrue(os.path.exists(os.path.join(model_dir, "calibrator.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model_dir, "optimal_threshold.json")))
        self.assertTrue(os.path.exists(os.path.join(model_dir, "feature_engine.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model_dir, "run_manifest.json")))

        with open(os.path.join(model_dir, "run_manifest.json")) as f:
            manifest = json.load(f)
            self.assertIn("candidate_recall_gate", manifest)
            self.assertIn("stages", manifest["candidate_recall_gate"])

        # 2. Infer
        run_inference(config_path=self.config_path)

        cand_tsv = cfg["paths"]["candidate_pairs"]
        match_tsv = cfg["paths"]["matching_results"]
        self.assertTrue(os.path.exists(cand_tsv))
        self.assertTrue(os.path.exists(match_tsv))

        # Check candidate_pairs format
        cand_df = pd.read_csv(cand_tsv, sep="\t")
        self.assertEqual(list(cand_df.columns), ["source1_entity_id", "candidate_entity_ids"])
        self.assertEqual(len(cand_df), 10, "Should have exactly 10 test S1 rows")

        # Check matching_results format
        match_df = pd.read_csv(match_tsv, sep="\t")
        self.assertEqual(list(match_df.columns), ["source1_entity_id", "matched_entity_ids"])
        self.assertEqual(len(match_df), 10, "Should have exactly 10 test S1 rows")


if __name__ == "__main__":
    unittest.main()
