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

### 2. Dataset Setup

Download the competition dataset zip archive (e.g., `student_resource.zip` or `dataset.zip`) from the challenge portal and place it in the project root directory.

#### Automated Setup (Recommended)

Run the following commands to extract and link the dataset files into the expected directory structure:

```bash
# 1. Extract the downloaded zip into dataset/
unzip student_resource.zip -d dataset/

# 2. Link or copy the train and test files to dataset/train and dataset/test:
mkdir -p dataset/train dataset/test

# If extracted into the competition nested path:
if [ -d "dataset/student_resource/student_resource/dataset" ]; then
    ln -sf "$(pwd)/dataset/student_resource/student_resource/dataset/train/"*.tsv dataset/train/
    ln -sf "$(pwd)/dataset/student_resource/student_resource/dataset/test/"*.tsv dataset/test/
elif [ -d "dataset/dataset" ]; then
    ln -sf "$(pwd)/dataset/dataset/train/"*.tsv dataset/train/
    ln -sf "$(pwd)/dataset/dataset/test/"*.tsv dataset/test/
fi
```

#### Expected Directory Layout

Ensure your `dataset/` directory contains all 7 required TSV files:

```text
dataset/
├── train/
│   ├── train_source1.tsv         # Source 1 training entities
│   ├── train_source2.tsv         # Source 2 candidate corpus
│   ├── train_source3.tsv         # Source 3 candidate corpus
│   └── train_ground_truth.tsv    # Ground truth mapping (source1_entity_id -> matched_entity_ids)
└── test/
    ├── test_source1.tsv          # Source 1 test query entities
    ├── test_source2.tsv          # Source 2 candidate corpus
    └── test_source3.tsv          # Source 3 candidate corpus
```

#### Verify Dataset Setup

Run this one-line verification to confirm all 7 dataset files are in place:

```bash
python3 -c "
import os, sys
required = [
    'dataset/train/train_source1.tsv',
    'dataset/train/train_source2.tsv',
    'dataset/train/train_source3.tsv',
    'dataset/train/train_ground_truth.tsv',
    'dataset/test/test_source1.tsv',
    'dataset/test/test_source2.tsv',
    'dataset/test/test_source3.tsv'
]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    print('ERROR: Missing dataset files:\n  ' + '\n  '.join(missing))
    sys.exit(1)
print('SUCCESS: All 7 required dataset files are present and ready!')
"
```

### 3. End-to-End Training
To run the full training pipeline on the training dataset:
```bash
python3 main.py --mode train
```
*Note: To run a fast prototype or debugging run on a sample of Source 1 records, pass `--sample-size`:*
```bash
python3 main.py --mode train --sample-size 25000
```

### 4. Inference on Test Set
To generate both `candidate_pairs.tsv` and `matching_results.tsv`:
```bash
python3 main.py --mode infer
```

### 5. Validate Submission
Verify all submission formatting constraints locally:
```bash
python3 main.py --mode validate
```

### 6. Package Submission Zip
Create the final submission archive `<team_name>_submission.zip`:
```bash
python3 utils/package_submission.py --team-name "AntigravityTeam"
```
