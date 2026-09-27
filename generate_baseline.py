import pandas as pd
import os

print("Loading test entities...")
s1_test = pd.read_csv("dataset/student_resource/student_resource/dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id"])
s1_test = s1_test.rename(columns={"entity_id": "source1_entity_id"})
s1_test["matched_entity_ids"] = ""

os.makedirs("output", exist_ok=True)
output_path = "output/matching_results_baseline.tsv"
s1_test.to_csv(output_path, sep="\t", index=False)
print(f"Baseline saved to {output_path}!")
