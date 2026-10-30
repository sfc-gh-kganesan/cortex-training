# Excerpt of DARE-Bench's scripts/utils.py (Apache-2.0): the ``process``
# function nested in get_processed_data, changed into a module-level function.
# The changes are taking ``question_key`` and ``db_base_path`` as arguments,
# raising where the original simulates a missing v1 reference with the code
# executor, and naming ``question_key`` in the unsupported-version error.
import json
import os

import pandas as pd


def process(x, question_key, db_base_path):
    if x["task"] == "time_series_analysis":
        if question_key == "question_v1":
            ground_truth_path = os.path.join(db_base_path, x["file_path"], "verify", "ground_truth_v1.csv")
            assert os.path.exists(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json")), f"all_metadata file not found for {x['file_path']}"
            with open(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json"), "r") as f:
                all_metadata = json.load(f)
            needed_files = all_metadata['question']["needed_files_v1"]

        else:
            ground_truth_path = os.path.join(db_base_path, x["file_path"], "verify", "ground_truth_v2.csv")
            assert os.path.exists(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json")), f"all_metadata file not found for {x['file_path']}"
            with open(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json"), "r") as f:
                all_metadata = json.load(f)
            needed_files = all_metadata['question']["needed_files_v2"]

    elif question_key == "question_v1":
        verify_folder = os.path.join(db_base_path, x["file_path"], "verify")
        assert os.path.exists(verify_folder), f"Verify folder not found for {x['file_path']} with {verify_folder}"
        assert os.path.exists(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json")), f"all_metadata file not found for {x['file_path']}"
        with open(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json"), "r") as f:
            all_metadata = json.load(f)
        needed_files = all_metadata['question']["needed_files_v1"]
        ground_truth_path = os.path.join(db_base_path, x["file_path"], "verify", "simulated_pred_local.csv")
        if not os.path.exists(ground_truth_path):
            raise FileNotFoundError(
                f"{verify_folder} has no v1 reference file; generate the v1 reference "
                "predictions (simulated_pred_local.csv) first"
            )
    elif question_key == "question_v2":
        ground_truth_path = os.path.join(db_base_path, x["file_path"], "verify", "ground_truth.csv")
        assert os.path.exists(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json")), f"all_metadata file not found for {x['file_path']}"
        with open(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json"), "r") as f:
            all_metadata = json.load(f)
        needed_files = all_metadata['question']["needed_files_v2"]
    else:
        raise NotImplementedError(f"Question version {question_key} not implemented")

    if "metadata.txt" not in needed_files:
        needed_files.append("metadata.txt")

    # try to read the ground truth file to ensure it exists, we only save the ground_truth_path and load during evaluation
    with open(ground_truth_path) as f:
        pd.read_csv(ground_truth_path)

    return [
        {
            "id": x["file_path"],
            "original_file_path": os.path.join(db_base_path, x["file_path"], "source"), # contains the the path to the original file
            "question": [x[question_key], ], # we will use [prompt, ] to make it compatible with multi-turn structure
            "metadata": {
                "needed_files": needed_files,
                "target": all_metadata["question"]["target"],
            },
            "task": x["task"],
            "tools": x["available_tools"],
            "ground_truth": ground_truth_path, # we will use the ground truth path to load the ground truth during evaluation
            "multi_turn": True,
        }
    ]
