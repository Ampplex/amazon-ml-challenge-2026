#!/bin/bash
set -eo pipefail

mkdir -p logs outputs

echo "=========================================================="
echo "Amazon ML Challenge 2026: Autonomous End-to-End Pipeline"
echo "=========================================================="
date

echo "=== [1/3] Training Phase: Steps 3-7 (Ranker, Hard Negatives, Calibration, Cross-Encoder, Holdout) ==="
if [ -f models/saved/pair_ranker.txt ] && [ -f models/saved/optimal_threshold.json ] && [ -f models/saved/calibrator.pkl ]; then
    echo "Training artifacts already present on disk (pair_ranker.txt, optimal_threshold.json, calibrator.pkl)."
    echo "Skipping to Phase 2 to avoid redundant retraining."
else
    python3 -u main.py --mode train 2>&1 | tee logs/training_full.log
fi

echo "=== [2/3] Inference Phase: Step 8 (Full Test Set Candidate Pairs & Matching Results) ==="
python3 -u main.py --mode infer 2>&1 | tee -a logs/inference_full.log

echo "=== [3/3] Validation Phase: Submission Rules Verification ==="
python3 -u main.py --mode validate 2>&1 | tee logs/validation.log

echo "=========================================================="
echo "Pipeline Completed Successfully!"
date
echo "=========================================================="
