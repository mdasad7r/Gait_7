import os
import json
from sklearn.model_selection import train_test_split
import config

# Output file
SPLIT_JSON = "data_split.json"

# Train/Test rule: 
# Train: nm-01~04, bg-01, cl-01
# Test: nm-05~06, bg-02, cl-02
TRAIN_SEQS = ["nm-01", "nm-02", "nm-03", "nm-04", "bg-01", "cl-01"]
TEST_SEQS  = ["nm-05", "nm-06", "bg-02", "cl-02"]

def is_valid_sequence(seq_name):
    """Return True if folder name is a valid sequence like 'nm-01', 'cl-02', etc."""
    return any(seq_name.startswith(prefix[:2]) for prefix in TRAIN_SEQS + TEST_SEQS)

def create_split(casia_root=config.CASIA_ROOT):
    train_data, test_data = [], []

    for subject_id in sorted(os.listdir(casia_root)):
        subject_path = os.path.join(casia_root, subject_id)
        if not os.path.isdir(subject_path) or not subject_id.isdigit():
            continue

        for seq_name in sorted(os.listdir(subject_path)):
            seq_path = os.path.join(subject_path, seq_name)
            if not os.path.isdir(seq_path) or not is_valid_sequence(seq_name):
                continue

            # Assign to train/test list based on sequence type
            if seq_name in TRAIN_SEQS:
                train_data.append(seq_path)
            elif seq_name in TEST_SEQS:
                test_data.append(seq_path)

    # Save JSON
    split = {"train": train_data, "test": test_data}
    with open(SPLIT_JSON, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Data split saved to {SPLIT_JSON}: {len(train_data)} train, {len(test_data)} test sequences.")

if __name__ == "__main__":
    create_split()
