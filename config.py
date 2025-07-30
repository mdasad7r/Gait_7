import os

# === Dataset & Paths ===
BASE_PATH = "CASIA-B"  # Base dataset folder (from Kaggle)
CASIA_ROOT = os.path.join(BASE_PATH, "output")  # Full path to silhouettes: CASIA-B/output/001/...

TRAIN_PATH = CASIA_ROOT  # Data loading is split by subject ID, not folder
TEST_PATH = CASIA_ROOT

MODEL_SAVE_DIR = "saved_models"       # Directory for saving trained models
LOG_DIR = "logs"                      # Optional: TensorBoard logs, etc.

# Create output directories if they don’t exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Input Shapes ===
SEQUENCE_LEN = 50        # Number of frames per sequence
IMG_HEIGHT = 64          # Silhouette image height
IMG_WIDTH = 64           # Silhouette image width
IMG_CHANNELS = 1         # Grayscale input
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# === Feature Dimensions ===
FEATURE_DIM = 256        # Projected feature size after fusion
TKAN_HIDDEN_DIM = 128    # Hidden dimension for TKAN

# === Training Parameters ===
NUM_CLASSES = 124        # Total subjects in CASIA-B (001–124)
DROPOUT_RATE = 0.3
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

# === ResNet Parameters ===
FREEZE_RESNET_LAYERS = 100   # Number of layers to freeze in ResNet50
RESNET_INPUT_CHANNELS = 3    # ResNet expects 3-channel input


