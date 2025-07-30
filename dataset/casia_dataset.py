import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# === Dataset Loader for CASIA-B ===
class CASIABDataset(Dataset):
    def __init__(self, sequences, sequence_len=50, image_size=(64, 64)):
        """
        Args:
            sequences: list of folder paths (each folder contains silhouette frames)
            sequence_len: number of frames per sequence to sample
            image_size: resize (H, W)
        """
        self.sequences = sequences
        self.sequence_len = sequence_len
        self.image_size = image_size

        # Preprocessing for each image
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),  # [0,1]
        ])

        # Map subject IDs to class indices
        self.subjects = sorted(list({seq.split(os.sep)[-3] for seq in sequences}))
        self.subject_to_idx = {sid: idx for idx, sid in enumerate(self.subjects)}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.sequences[idx]

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

        # Load frames
        frames = [self.transform(Image.open(f)) for f in frame_files]  # list of [C,H,W]

        # Stack into [T,C,H,W]
        frames_tensor = torch.stack(frames)

        return frames_tensor, label


def get_loaders(train_sequences, test_sequences, batch_size=16, sequence_len=50, image_size=(64, 64)):
    """Creates PyTorch DataLoaders for CASIA-B train/test sets."""
    train_dataset = CASIABDataset(train_sequences, sequence_len, image_size)
    test_dataset = CASIABDataset(test_sequences, sequence_len, image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, train_dataset.subject_to_idx


if __name__ == "__main__":
    # Example usage with split_dataset.py
    from split_dataset import get_all_sequences

    BASE_PATH = "CASIA-B/output"
    train_list, test_list = get_all_sequences(BASE_PATH)

    print(f"Train: {len(train_list)} sequences | Test: {len(test_list)} sequences")

    train_loader, test_loader, label_map = get_loaders(train_list, test_list, batch_size=8)
    print(f"Classes: {len(label_map)}")
    for batch_x, batch_y in train_loader:
        print("Batch X:", batch_x.shape)  # [B,T,C,H,W]
        print("Batch Y:", batch_y.shape)
        break
