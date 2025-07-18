# split_dataset.py

import os
import numpy as np

def get_all_subject_ids(dataset_root="CASIA_B/output"):
    """Returns all subject IDs found in the dataset directory."""
    ids = sorted([d for d in os.listdir(dataset_root) if d.isdigit()])
    return ids

def get_train_val_test_ids(dataset_root="CASIA_B/output"):
    """Split 001–074 for training/validation, 075–124 for testing."""
    all_ids = get_all_subject_ids(dataset_root)
    train_val_ids = [sid for sid in all_ids if 1 <= int(sid) <= 74]
    test_ids = [sid for sid in all_ids if 75 <= int(sid) <= 124]
    return train_val_ids, test_ids

def k_fold_split(subject_ids, k=5, seed=42):
    """Split subject IDs into k folds for cross-validation."""
    np.random.seed(seed)
    shuffled = np.random.permutation(subject_ids)
    fold_size = len(subject_ids) // k
    folds = [shuffled[i * fold_size:(i + 1) * fold_size] for i in range(k)]
    return folds

def get_gallery_probe_sequences(subject_root):
    """
    For a given subject, return two lists of (sequence_type, view) tuples:
    - gallery = NM01 to NM04
    - probe  = NM05–06, CL, BG
    """
    gallery, probe = [], []

    for seq_type in os.listdir(subject_root):
        if not seq_type.startswith(('nm', 'cl', 'bg')):
            continue

        seq_type_path = os.path.join(subject_root, seq_type)
        for view in os.listdir(seq_type_path):
            key = (seq_type, view)

            # NM01–04 are gallery
            if seq_type.startswith('nm'):
                nm_id = int(seq_type.split('-')[1])
                if 1 <= nm_id <= 4:
                    gallery.append(key)
                else:
                    probe.append(key)
            else:
                probe.append(key)

    return gallery, probe
