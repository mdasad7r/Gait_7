import os

# === Custom Split for CASIA-B ===
# Train sequences: nm-01..04, bg-01, cl-01
# Test sequences:  nm-05..06, bg-02, cl-02

TRAIN_KEYS = ["nm-01", "nm-02", "nm-03", "nm-04", "bg-01", "cl-01"]
TEST_KEYS  = ["nm-05", "nm-06", "bg-02", "cl-02"]

def get_sequences_by_condition(subject_root):
    """
    Returns two lists: train_sequence_paths, test_sequence_paths
    Each sequence path points to the folder that contains the frames for a view.
    Example:
    CASIA-B/output/001/nm-01/000/   -> this is a sequence path
    """
    train_seqs, test_seqs = [], []

    # Loop through all sequence types (nm-01, cl-01, etc.)
    for seq_type in sorted(os.listdir(subject_root)):
        seq_type_path = os.path.join(subject_root, seq_type)
        if not os.path.isdir(seq_type_path):
            continue

        # Determine if this sequence type belongs to train or test
        seq_key = seq_type.lower()

        # Loop through all views (000, 018, 036, ...)
        for view in sorted(os.listdir(seq_type_path)):
            view_path = os.path.join(seq_type_path, view)
            if not os.path.isdir(view_path):
                continue

            if any(seq_key.startswith(k) for k in TRAIN_KEYS):
                train_seqs.append(view_path)
            elif any(seq_key.startswith(k) for k in TEST_KEYS):
                test_seqs.append(view_path)

    return train_seqs, test_seqs


def get_all_sequences(casia_root):
    """
    Loops over all subjects in CASIA-B and collects train/test sequences
    according to the specified protocol.
    """
    train_paths, test_paths = [], []

    for sid in sorted(os.listdir(casia_root)):
        subject_dir = os.path.join(casia_root, sid)
        if not os.path.isdir(subject_dir):
            continue

        t_paths, tst_paths = get_sequences_by_condition(subject_dir)
        train_paths.extend(t_paths)
        test_paths.extend(tst_paths)

    return train_paths, test_paths


if __name__ == "__main__":
    # Example usage
    BASE_PATH = "CASIA-B/output"  # update if needed
    train_list, test_list = get_all_sequences(BASE_PATH)
    print(f"Total training sequences: {len(train_list)}")
    print(f"Total testing sequences: {len(test_list)}")
    print("Example train seq:", train_list[:3])
    print("Example test seq:", test_list[:3])
