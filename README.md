# Multi-Source Business Entity Resolution — Amazon ML Challenge 2026

A production-grade, two-stage retrieval and matching architecture designed for the Amazon ML Challenge 2026 Business Entity Resolution task.

## Key Highlights & Innovations
- **Two-Stage Candidate Generation & Compression**: Multi-channel retrieval (Exact Inverted Indexes + Character TF-IDF + Word TF-IDF + BM25 + Dense ANN) combined with Reciprocal Rank Fusion (RRF) and structural filtering to keep candidate sets compact (maximizing the competition's candidate reduction ranking).
- **Comprehensive Feature Engineering**: 30+ lexical, structural, token rarity (IDF), cross-source triangle consistency, and retrieval confidence features.
- **Precision-Heavy Discriminative Matching**: LightGBM pair ranker with hard-negative mining, selective cross-encoder reranking for ambiguous pairs, and isotonic probability calibration.
- **F₀.₅ Entity-Level Decision Layer**: Explicit optimization on macro-averaged F₀.₅ per Source 1 entity, natively supporting 0 matches (singletons), 1 match, and multi-record matches (2+).
- **Open-Set Generalization**: Generalizes to unseen countries (France) without rigid hardcoded rules.

---

## Repository Structure

```text
├── configs/
│   └── config.yaml                 # Central configuration for data paths, retrieval, models
├── dataset/
│   ├── train/                      # Training files (train_source1.tsv, S2, S3, ground truth)
│   └── test/                       # Test files (test_source1.tsv, S2, S3)
├── preprocessing/
│   ├── schema.py                   # Canonical Record, CandidatePair, and I/O loaders
│   ├── normalize_names.py          # Multilingual name cleaning & abbreviation normalization
│   └── normalize_addresses.py      # Address component parsing (postal, house, city, landmark)
├── retrieval/
│   └── candidate_generation.py     # Exact blocking, TF-IDF, BM25, and RRF candidate compression
├── features/
│   └── feature_engineering.py      # Lexical, structural, rarity, semantic, and retrieval features
├── models/
│   └── matching_models.py          # LightGBM ranker, CrossEncoder, Calibrator, ThresholdOptimizer
├── graph/
│   └── consistency.py              # Cross-source S2-S3 triangle consistency
├── training/
│   ├── build_pairs.py              # Entity-level train/val split & hard negative pair sampler
│   ├── hard_negative_mining.py     # Iterative high-confidence false positive mining
│   └── train_ranker.py             # End-to-end training pipeline
├── inference/
│   └── pipeline.py                 # Test inference generating candidate_pairs.tsv & matching_results.tsv
├── evaluation/
│   └── f05.py                      # Macro F0.5 and blocking recall evaluation
├── outputs/
│   ├── candidate_pairs.tsv         # Final candidate sets fed to the matcher
│   └── matching_results.tsv        # Scored leaderboard match predictions
├── utils/
│   ├── validate_submission.py      # Official submission validator
│   └── package_submission.py       # Submission zip packager
├── main.py                         # Unified CLI entrypoint
└── requirements.txt                # Pinned dependencies
```

---

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. End-to-End Training
To run the full training pipeline on the training dataset:
```bash
python3 main.py --mode train
```
*Note: To run a fast prototype or debugging run on a sample of Source 1 records, pass `--sample-size`:*
```bash
python3 main.py --mode train --sample-size 25000
```

### 3. Inference on Test Set
To generate both `candidate_pairs.tsv` and `matching_results.tsv`:
```bash
python3 main.py --mode infer
```

### 4. Validate Submission
Verify all submission formatting constraints locally:
```bash
python3 main.py --mode validate
```

### 5. Package Submission Zip
Create the final submission archive `<team_name>_submission.zip`:
```bash
python3 utils/package_submission.py --team-name "AntigravityTeam"
```
