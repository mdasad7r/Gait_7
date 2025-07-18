import os
import tensorflow as tf
from model.resnet_tkan import build
from dataset.casia_dataset import get_dataset
from split_dataset import get_train_val_test_ids, k_fold_split
import config


def train_one_fold(fold_index, train_ids, val_ids):
    print(f"\n--- Training Fold {fold_index + 1} ---")

    # Load datasets
    train_ds = get_dataset(train_ids, mode='train')
    val_ds = get_dataset(val_ids, mode='val')

    # Build and compile model
    model = build()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        verbose=1
    )

    # Save model
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    model.save(os.path.join(config.MODEL_SAVE_DIR, f"model_fold_{fold_index + 1}.h5"))


def train_cross_validation():
    train_val_ids, _ = get_train_val_test_ids()
    folds = k_fold_split(train_val_ids, k=config.NUM_FOLDS, seed=config.RANDOM_SEED)

    for i in range(config.NUM_FOLDS):
        val_ids = folds[i]
        train_ids = [sid for j, fold in enumerate(folds) if j != i for sid in fold]
        train_one_fold(i, train_ids, val_ids)


def train_final_and_test():
    _, test_ids = get_train_val_test_ids()
    train_ids = [f"{i:03d}" for i in range(1, 75)]  # 001–074

    print("\n--- Training Final Model on All Train+Val Subjects ---")
    train_ds = get_dataset(train_ids, mode='train')
    test_ds = get_dataset(test_ids, mode='test')

    model = build()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        epochs=config.EPOCHS,
        verbose=1
    )

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    model.save(os.path.join(config.MODEL_SAVE_DIR, "final_model.h5"))

    print("\n--- Evaluating on Test Set ---")
    model.evaluate(test_ds, verbose=1)


if __name__ == "__main__":
    train_cross_validation()
    train_final_and_test()
