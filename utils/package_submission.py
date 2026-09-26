"""
Submission packager script for Amazon ML Challenge 2026.

Assembles the final archive:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import argparse
import os
import shutil
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("package_submission")


def create_submission_package(team_name: str, base_dir: str = ".") -> str:
    """Build the official submission zip archive."""
    clean_team_name = team_name.strip().replace(" ", "_")
    zip_filename = f"{clean_team_name}_submission.zip"
    zip_path = os.path.join(base_dir, zip_filename)
    staging_dir = os.path.join(base_dir, "temp_submission_staging")

    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)

    try:
        # Create folder hierarchy
        output_stage = os.path.join(staging_dir, "output")
        code_stage = os.path.join(staging_dir, "code", "business_entity_resolution")
        src_stage = os.path.join(code_stage, "src")

        os.makedirs(output_stage, exist_ok=True)
        os.makedirs(src_stage, exist_ok=True)

        # 1. Output files
        match_src = os.path.join(base_dir, "outputs", "matching_results.tsv")
        cand_src = os.path.join(base_dir, "outputs", "candidate_pairs.tsv")

        if os.path.exists(match_src):
            shutil.copy2(match_src, os.path.join(output_stage, "matching_results.tsv"))
            logger.info("Copied matching_results.tsv to output/")
        else:
            logger.warning("outputs/matching_results.tsv not found! (Will be empty in package)")

        if os.path.exists(cand_src):
            shutil.copy2(cand_src, os.path.join(output_stage, "candidate_pairs.tsv"))
            logger.info("Copied candidate_pairs.tsv to output/")
        else:
            logger.warning("outputs/candidate_pairs.tsv not found!")

        # 2. Source code
        code_dirs = ["preprocessing", "retrieval", "features", "models", "graph", "training", "inference", "evaluation", "configs", "utils"]
        for d in code_dirs:
            src_dir = os.path.join(base_dir, d)
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, os.path.join(src_stage, d), dirs_exist_ok=True)

        main_py = os.path.join(base_dir, "main.py")
        if os.path.exists(main_py):
            shutil.copy2(main_py, os.path.join(src_stage, "main.py"))

        # Explicitly ensure trained models and artifacts are packaged
        models_saved = os.path.join(base_dir, "models", "saved")
        if os.path.exists(models_saved):
            shutil.copytree(models_saved, os.path.join(src_stage, "models", "saved"), dirs_exist_ok=True)
            logger.info(f"Packaged trained model artifacts from {models_saved}")

        # 3. README and requirements
        shutil.copy2(os.path.join(base_dir, "README.md"), os.path.join(code_stage, "README.md"))
        shutil.copy2(os.path.join(base_dir, "requirements.txt"), os.path.join(code_stage, "requirements.txt"))

        # 4. Documentation_template.md
        doc_path = os.path.join(base_dir, "Documentation_template.md")
        if os.path.exists(doc_path):
            shutil.copy2(doc_path, os.path.join(staging_dir, "Documentation_template.md"))

        # 5. Create zip
        logger.info(f"Creating zip archive at {zip_path}...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(staging_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, staging_dir)
                    zipf.write(file_path, arcname)

        logger.info(f"Successfully generated submission zip: {zip_path} ({os.path.getsize(zip_path) / 1024 / 1024:.2f} MB)")
        return zip_path

    finally:
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create final submission zip package.")
    parser.add_argument("--team-name", default="AmazonML_Team", help="Your team name.")
    args = parser.parse_args()
    create_submission_package(args.team_name)
