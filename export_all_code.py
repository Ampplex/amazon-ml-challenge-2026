import os

files_to_export = [
    'configs/config.yaml',
    'preprocessing/schema.py',
    'preprocessing/normalize_names.py',
    'preprocessing/normalize_addresses.py',
    'retrieval/candidate_generation.py',
    'retrieval/embeddings.py',
    'retrieval/faiss_index.py',
    'features/feature_engineering.py',
    'models/matching_models.py',
    'graph/consistency.py',
    'training/build_pairs.py',
    'training/hard_negative_mining.py',
    'training/train_ranker.py',
    'inference/pipeline.py',
    'evaluation/f05.py',
    'utils/validate_submission.py',
    'utils/package_submission.py',
    'tests/test_scaling_verification.py',
    'tests/test_end_to_end_pipeline.py',
    'main.py',
    'requirements.txt',
    'Documentation_template.md',
]

output_file = 'all_code.md'

with open(output_file, 'w', encoding='utf-8') as out:
    out.write('# Amazon ML Challenge 2026 — Complete Pipeline Source Code\n\n')
    out.write('This document contains the complete source code for all modules in the Business Entity Resolution pipeline.\n\n')
    out.write('## Table of Contents\n\n')
    for fpath in files_to_export:
        anchor = fpath.replace('/', '').replace('.', '').replace('_', '-').lower()
        out.write(f'- [{fpath}](#file-{anchor})\n')
    out.write('\n---\n\n')

    for fpath in files_to_export:
        if os.path.exists(fpath):
            ext = fpath.split('.')[-1]
            lang = 'python' if ext == 'py' else ('yaml' if ext == 'yaml' else ('markdown' if ext == 'md' else 'text'))
            out.write(f'## File: `{fpath}`\n\n')
            out.write(f'```{lang}\n')
            with open(fpath, 'r', encoding='utf-8') as f:
                out.write(f.read())
            out.write('\n```\n\n---\n\n')

print(f'Successfully generated {output_file} ({os.path.getsize(output_file):,} bytes)')
