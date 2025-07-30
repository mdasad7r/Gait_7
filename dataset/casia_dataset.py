import os
import json
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import config

# === Utility Functions ===

def load_split(json_path="data_split.json"):
    """Load train/test split JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} not found. Run split_dataset.py first.")
    
    with open(json_path, "r") as f:
        return json.load(f)


def load_sequence(seq_path, seq_len=config.SEQUENCE_LEN, image_size=config.IMAGE_SIZE):
    """
    Load all frames in a sequence folder, resize, normalize and return
    fixed-length tensor of shape (seq_len, H, W, 1)
    """
    frames = sorted(os.listdir(seq_path))
    frames = [f for f in frames if f.endswith((".png", ".jpg"))]

    imgs = []
    for f in frames[:seq_len]:  # Truncate
        img = load_img(os.path.join(seq_path, f), color_mode="grayscale", target_size=image_size)
        img = img_to_array(img) / 255.0  # Normalize
        imgs.append(img)

    # Pad sequences if shorter than SEQUENCE_LEN
    while len(imgs) < seq_len:
        imgs.append(np.zeros((*image_size, 1), dtype=np.float32))

    return np.array(imgs, dtype=np.float32)


def build_dataset(split="train", json_path="data_split.json"):
    """
    Load sequences and create dataset arrays.
    Returns: X of shape (N, T, H, W, 1), y (N, NUM_CLASSES)
    """
    split_data = load_split(json_path)
    seq_paths = split_data[split]

    X, y = [], []
    for seq_path in seq_paths:
        # Subject ID is folder name like "001"
        subject_id = int(os.path.basename(os.path.dirname(seq_path))) - 1  # 0-indexed
        seq_array = load_sequence(seq_path)
        X.append(seq_array)
        y.append(subject_id)

    X = np.array(X, dtype=np.float32)
    y = to_categorical(y, num_classes=config.NUM_CLASSES)
    print(f"[{split.upper()}] Loaded {len(X)} sequences: {X.shape}")

    return X, y


if __name__ == "__main__":
    # Quick test
    X_train, y_train = build_dataset("train")
    X_test, y_test = build_dataset("test")
    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)
