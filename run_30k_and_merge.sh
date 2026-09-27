#!/bin/bash
echo "Starting 30k sample pipeline..."
python3 main.py --mode full --sample-size 30000

echo "Merging 30k predictions into the 1.7M baseline file..."
python3 -c '
import pandas as pd
baseline = pd.read_csv("output/matching_results_baseline.tsv", sep="\t", dtype=str).fillna("")
preds = pd.read_csv("output/matching_results.tsv", sep="\t", dtype=str).fillna("")
preds_dict = dict(zip(preds["source1_entity_id"], preds["matched_entity_ids"]))
baseline["matched_entity_ids"] = baseline["source1_entity_id"].map(lambda x: preds_dict.get(x, ""))
baseline.to_csv("matching_results.tsv", sep="\t", index=False)
'
echo "Done! Final file saved to matching_results.tsv"
