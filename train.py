import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from model.resnet_tkan import build_model  # Import your model

import config

# ===============================
# CONFIG
# ===============================
SEQUENCE_LEN = config.SEQUENCE_LEN       # e.g., 50
IMG_SIZE = (64, 64)
BATCH_SIZE = 8
EPOCHS = 20

DATA_DIR = "/content/Gait_7/casia-b/train/output"  # Root folder


# ===============================
# 1️⃣ Load dataset from PNGs
# ===============================
def load_sequences(data_dir, sequence_len=50):
    """
    Reads CASIA-B dataset sequences from the folder structure:
    /id/condition/seq/frame.png
    Returns:
        gray_sequences: (N, T, 64, 64, 1)
        rgb_sequences: (N, T, 64, 64, 3)
        labels: (N,)
    """
    seq_paths = sorted(glob.glob(os.path.join(data_dir, "*/*/*")))  # e.g., .../001/bg-01/018

    gray_sequences = []
    rgb_sequences = []
    labels = []

    for seq_path in seq_paths:
        # Get all frames in this sequence
        frame_paths = sorted(glob.glob(os.path.join(seq_path, "*.png")))
        if len(frame_paths) < sequence_len:
            continue  # skip short sequences

        # Take first `sequence_len` frames
        frame_paths = frame_paths[:sequence_len]

        # Load frames
        gray_frames = []
        rgb_frames = []
        for f in frame_paths:
            # Grayscale
            gray_img = load_img(f, color_mode="grayscale", target_size=IMG_SIZE)
            gray_arr = img_to_array(gray_img) / 255.0  # (64, 64, 1)

            # RGB
            rgb_img = load_img(f, color_mode="rgb", target_size=IMG_SIZE)
            rgb_arr = img_to_array(rgb_img) / 255.0  # (64, 64, 3)

            gray_frames.append(gray_arr)
            rgb_frames.append(rgb_arr)

        # Stack sequence
        gray_sequences.append(np.stack(gray_frames, axis=0))
        rgb_sequences.append(np.stack(rgb_frames, axis=0))

        # Label = subject ID (folder name)
        subject_id = os.path.basename(os.path.dirname(os.path.dirname(seq_path)))  # e.g., "001"
        labels.append(int(subject_id) - 1)  # make zero-based

    # Convert to arrays
    gray_sequences = np.array(gray_sequences, dtype=np.float32)
    rgb_sequences = np.array(rgb_sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print("Loaded sequences:", gray_sequences.shape, rgb_sequences.shape, labels.shape)
    return gray_sequences, rgb_sequences, labels


print("📥 Loading CASIA-B dataset...")
X_gray, X_rgb, y = load_sequences(DATA_DIR, sequence_len=SEQUENCE_LEN)

# ===============================
# 2️⃣ Train / Test split
# ===============================
Xg_train, Xg_val, Xrgb_train, Xrgb_val, y_train, y_val = train_test_split(
    X_gray, X_rgb, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 3️⃣ Build model
# ===============================
print("🔧 Building model...")
model = build_model(
    sequence_len=SEQUENCE_LEN,
    cnn_input_shape=(64, 64, 1),
    resnet_input_shape=(64, 64, 3),
    feature_dim=config.FEATURE_DIM,
    tkan_hidden_dim=config.TKAN_HIDDEN_DIM,
    num_classes=config.NUM_CLASSES,
    dropout_rate=config.DROPOUT_RATE,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ===============================
# 4️⃣ Train model
# ===============================
print("🚀 Training...")
history = model.fit(
    [Xg_train, Xrgb_train],
    y_train,
    validation_data=([Xg_val, Xrgb_val], y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# ===============================
# 5️⃣ Save model
# ===============================
model.save("/content/Gait_7/dual_encoder_tkan_model.h5")
print("✅ Training complete. Model saved.")
