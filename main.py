"""
Main CLI entrypoint for the Amazon ML Challenge 2026 Entity Resolution system.

Usage:
    python3 main.py --mode train [--sample-size 30000]
    python3 main.py --mode infer [--sample-size 10000]
    python3 main.py --mode full
    python3 main.py --mode validate
"""

import argparse
import logging
import os
import subprocess
import sys

from inference.pipeline import run_inference
from preprocessing.schema import load_config
from training.train_ranker import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")


def validate_outputs(config_path: str = "configs/config.yaml") -> int:
    """Run the official submission validator on outputs."""
    config = load_config(config_path)
    base_dir = config["paths"]["base_dir"]
    validator_path = os.path.join(base_dir, "utils/validate_submission.py")
    matching_path = os.path.join(base_dir, config["paths"]["matching_results"])
    candidate_path = os.path.join(base_dir, config["paths"]["candidate_pairs"])
    test_dir = os.path.join(base_dir, config["paths"]["test_dir"])

    if not os.path.exists(validator_path):
        logger.error(f"Validator script not found at {validator_path}")
        return 1

    if not os.path.exists(matching_path):
        logger.error(f"Matching results not found at {matching_path}")
        return 1

    cmd = [
        sys.executable, validator_path,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir
    ]
    logger.info(f"Running validator: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026: Multi-Source Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "full", "validate"],
        default="full",
        help="Pipeline execution mode: 'train', 'infer', 'full' (train + infer), or 'validate'."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional S1 sample size for fast training or testing."
    )

    args = parser.parse_args()

    if args.mode in ("train", "full"):
        logger.info(">>> Starting Training Phase <<<")
        run_training(
            config_path=args.config,
            sample_s1_size=args.sample_size
        )

    if args.mode in ("infer", "full"):
        logger.info(">>> Starting Inference Phase <<<")
        # If running the full pipeline, we ALWAYS want to infer on the full test set
        test_sample = None if args.mode == "full" else args.sample_size
        run_inference(
            config_path=args.config,
            sample_test_size=test_sample
        )

    if args.mode == "validate":
        code = validate_outputs(args.config)
        sys.exit(code)


if __name__ == "__main__":
    main()
