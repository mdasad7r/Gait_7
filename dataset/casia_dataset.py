import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# === Dataset Loader for CASIA-B ===
class CASIABDataset(tf.keras.utils.Sequence):
    def __init__(self, sequences, sequence_len=50, image_size=(64, 64), batch_size=16, shuffle=True):
        """
        Args:
            sequences: list of folder paths (each folder contains silhouette frames)
            sequence_len: number of frames per sequence to sample
            image_size: resize (H, W)
            batch_size: number of sequences per batch
            shuffle: shuffle dataset each epoch
        """
        self.sequences = sequences
        self.sequence_len = sequence_len
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Map subject IDs to class indices
        self.subjects = sorted(list({seq.split(os.sep)[-3] for seq in sequences}))
        self.subject_to_idx = {sid: idx for idx, sid in enumerate(self.subjects)}

        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.sequences) / self.batch_size))

    def on_epoch_end(self):
        """Shuffle after each epoch"""
        self.indices = np.arange(len(self.sequences))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        """Generate one batch"""
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_sequences = [self.sequences[i] for i in batch_indices]

        X, y = self.__data_generation(batch_sequences)
        return X, y

    def __data_generation(self, batch_sequences):
        X = []
        y = []

        for seq_path in batch_sequences:
            # Label is subject ID (folder name 3 levels up: 001/nm-01/000)
            subject_id = seq_path.split(os.sep)[-3]
            label = self.subject_to_idx[subject_id]

            # Collect all frame paths
            frame_files = sorted([
                os.path.join(seq_path, f) for f in os.listdir(seq_path)
                if f.endswith(".png") or f.endswith(".jpg")
            ])

            # Handle sequences shorter than required
            if len(frame_files) < self.sequence_len:
                frame_files = (frame_files * (self.sequence_len // len(frame_files) + 1))[:self.sequence_len]

            # Uniformly sample frames
            step = len(frame_files) // self.sequence_len
            frame_files = frame_files[::step][:self.sequence_len]

            # Load frames as grayscale
            frames = []
            for f in frame_files:
                img = Image.open(f).convert("L").resize(self.image_size)
                img_array = np.array(img, dtype=np.float32) / 255.0  # normalize [0,1]
                img_array = np.expand_dims(img_array, axis=-1)  # [H,W,1]
                frames.append(img_array)

            # Stack into [T,H,W,1]
            frames_tensor = np.stack(frames, axis=0)
            X.append(frames_tensor)
            y.append(label)

        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)  # [B,T,H,W,1]
        y = np.array(y, dtype=np.int32)
        return X, y


def get_loaders(train_sequences, test_sequences, batch_size=16, sequence_len=50, image_size=(64, 64)):
    """Creates TensorFlow dataset generators for CASIA-B train/test sets."""
    train_loader = CASIABDataset(train_sequences, sequence_len, image_size, batch_size=batch_size, shuffle=True)
    test_loader = CASIABDataset(test_sequences, sequence_len, image_size, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_loader.subject_to_idx


if __name__ == "__main__":
    # Example usage with split_dataset.py
    from split_dataset import get_all_sequences

    BASE_PATH = "CASIA-B/output"
    train_list, test_list = get_all_sequences(BASE_PATH)

    print(f"Train: {len(train_list)} sequences | Test: {len(test_list)} sequences")

    train_loader, test_loader, label_map = get_loaders(train_list, test_list, batch_size=8)
    print(f"Classes: {len(label_map)}")
    for batch_x, batch_y in train_loader:
        print("Batch X:", batch_x.shape)  # [B,T,H,W,1]
        print("Batch Y:", batch_y.shape)
        break
