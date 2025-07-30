import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime

from config import (
    SEQUENCE_LEN, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_CLASSES,
    MODEL_SAVE_DIR, LOG_DIR
)

from split_dataset import get_all_sequences
from dataset.casia_dataset import get_loaders  # will now return tf.data.Dataset
from model.resnet_tkan import build_model  # should return a Keras Model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    In TensorFlow, we typically use model.fit(), 
    but keeping function name for compatibility.
    This function now uses a custom training loop.
    """
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = criterion(labels, outputs)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_loss += loss.numpy() * inputs.shape[0]
        preds = tf.argmax(outputs, axis=1)
        correct += tf.reduce_sum(tf.cast(preds == labels, tf.int32)).numpy()
        total += labels.shape[0]

    return total_loss / total, correct / total


def validate_one_epoch(model, loader, criterion, device):
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        outputs = model(inputs, training=False)
        loss = criterion(labels, outputs)

        total_loss += loss.numpy() * inputs.shape[0]
        preds = tf.argmax(outputs, axis=1)
        correct += tf.reduce_sum(tf.cast(preds == labels, tf.int32)).numpy()
        total += labels.shape[0]

    return total_loss / total, correct / total


def main():
    # === Device Setup ===
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        device = "GPU"
    else:
        device = "CPU"
    print(f"Using device: {device}")

    # === Load Dataset ===
    base_path = "CASIA-B/output"
    train_sequences, test_sequences = get_all_sequences(base_path)
    train_loader, test_loader, label_map = get_loaders(
        train_sequences,
        test_sequences,
        batch_size=BATCH_SIZE,
        sequence_len=SEQUENCE_LEN,
        image_size=IMAGE_SIZE
    )

    print(f"Train sequences: {len(train_sequences)}, Test sequences: {len(test_sequences)}")
    print(f"Number of classes: {len(label_map)}")

    # === Model ===
    model = build_model(num_classes=len(label_map))
    optimizer = Adam(learning_rate=LEARNING_RATE)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # TensorBoard
    log_dir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = TensorBoard(log_dir=log_dir)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # TensorBoard logging
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar("Loss/train", train_loss, step=epoch)
            tf.summary.scalar("Loss/val", val_loss, step=epoch)
            tf.summary.scalar("Accuracy/train", train_acc, step=epoch)
            tf.summary.scalar("Accuracy/val", val_acc, step=epoch)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model.save(os.path.join(MODEL_SAVE_DIR, f"best_model_epoch{epoch+1}.h5"))
            print(f"Saved best model with accuracy {best_acc:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    main()
