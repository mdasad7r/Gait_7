# config.py

# === Dataset & Paths ===
CASIA_ROOT = "CASIA_B/output"         # Path to unzipped silhouettes (e.g., CASIA_B/output/001/...)
MODEL_SAVE_DIR = "saved_models"       # Directory for saving trained models

# === Input Shapes ===
SEQUENCE_LEN = 50                     # Number of frames per sequence
IMG_HEIGHT = 64                        # Silhouette image height
IMG_WIDTH = 64                         # Silhouette image width
IMG_CHANNELS = 1                      # Grayscale input

# === Feature Dimensions ===
FEATURE_DIM = 256                    # Projected feature size after fusion
TKAN_HIDDEN_DIM = 128                # Hidden dimension for TKAN

# === Training Parameters ===
NUM_CLASSES = 124                     # Only subjects 001–074 are used for training
DROPOUT_RATE = 0.3
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

# === ResNet Parameters ===
FREEZE_RESNET_LAYERS = 100           # Number of layers to freeze in ResNet50
RESNET_INPUT_CHANNELS = 3            # ResNet expects 3-channel input

# === Cross-Validation ===
NUM_FOLDS = 5
RANDOM_SEED = 42
