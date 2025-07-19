import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
import config


def load_silhouette_sequence(seq_path, sequence_len=config.SEQUENCE_LEN, img_size=config.IMAGE_SIZE):
    """
    Load a silhouette sequence from a directory of PNG frames.
    Samples `sequence_len` frames uniformly and pads if necessary.
    """
    frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
    total = len(frames)

    if total == 0:
        return None

    step = max(1, total // sequence_len)
    selected = frames[::step][:sequence_len]

    imgs = []
    for fname in selected:
        img_path = os.path.join(seq_path, fname)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        imgs.append(img)

    # Pad with black frames if not enough
    while len(imgs) < sequence_len:
        imgs.append(tf.zeros_like(imgs[0]))

    return tf.stack(imgs)  # Shape: (T, H, W, 1)


def load_dataset(subject_ids, mode='train'):
    """
    Load all silhouette sequences and labels for given subject IDs.
    Returns two tensors: x_data (T, H, W, 1), y_data (categorical labels).
    """
    x_data, y_data = [], []

    base_path = config.TRAIN_PATH if mode in ['train', 'val'] else config.TEST_PATH

    for sid in subject_ids:
        subject_path = os.path.join(base_path, sid)
        if not os.path.exists(subject_path):
            continue

        for seq_type in os.listdir(subject_path):
            if not seq_type.startswith(('nm', 'cl', 'bg')):
                continue

            seq_type_path = os.path.join(subject_path, seq_type)

            for view in os.listdir(seq_type_path):
                seq_path = os.path.join(seq_type_path, view)
                if not os.path.isdir(seq_path):
                    continue

                sequence = load_silhouette_sequence(seq_path)
                if sequence is not None:
                    x_data.append(sequence)
                    y_data.append(int(sid) - 1)  # zero-based class indexing

    if not x_data:
        raise ValueError(f"No data found for subject_ids: {subject_ids}")

    x_data = tf.stack(x_data)
    y_data = to_categorical(y_data, config.NUM_CLASSES)

    return x_data, y_data


def get_dataset(subject_ids, mode='train'):
    """
    Wraps (x, y) tensors into a batched, prefetched tf.data.Dataset.
    """
    x, y = load_dataset(subject_ids, mode)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if mode == 'train':
        dataset = dataset.shuffle(1024)
    dataset = dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset
