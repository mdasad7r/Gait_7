import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from datetime import datetime

import config
from model.resnet_tkan import build
from dataset.casia_dataset import build_dataset

# === GPU Config (Optional) ===
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)


# === 1. Load Dataset ===
print("[INFO] Loading CASIA-B dataset...")
X_train, y_train = build_dataset("train")
X_test, y_test = build_dataset("test")

print(f"[INFO] Training data: {X_train.shape}, {y_train.shape}")
print(f"[INFO] Validation data: {X_test.shape}, {y_test.shape}")


# === 2. Build Model ===
print("[INFO] Building model...")
model = build()
model.summary()

# === 3. Compile Model ===
optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])


# === 4. Callbacks ===
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, f"model_{timestamp}.h5")
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    TensorBoard(log_dir=os.path.join(config.LOG_DIR, timestamp))
]

# === 5. Train Model ===
print("[INFO] Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS,
    callbacks=callbacks,
    shuffle=True
)

# === 6. Save Final Model ===
final_model_path = os.path.join(config.MODEL_SAVE_DIR, f"final_model_{timestamp}.h5")
model.save(final_model_path)
print(f"[INFO] Training complete. Final model saved to {final_model_path}")
