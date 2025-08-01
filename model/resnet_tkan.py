from keras.models import Model
from keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D, GlobalAveragePooling1D,
    Concatenate, TimeDistributed, Lambda
)
from keras.applications import ResNet50
from keras import layers
import tensorflow as tf
from tkan import TKAN
import config


# === Custom CNN encoder ===
def build_custom_cnn_encoder(input_shape=(64, 64, 1), feature_dim=config.FEATURE_DIM):
    inp = Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(feature_dim, activation='relu')(x)

    return Model(inp, x, name="custom_cnn_encoder")


# === ResNet50-based encoder with partial layer freezing ===
def build_resnet50_encoder(input_shape=(64, 64, 3), projection_dim=config.FEATURE_DIM):
    inp = Input(shape=input_shape)
    x = Lambda(lambda x: tf.image.resize(x, (32, 32), method='bilinear'))(inp)

    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3))

    # Freeze first N layers
    frozen_layers = getattr(config, "FREEZE_RESNET_LAYERS", 100)
    for layer in base_model.layers[:frozen_layers]:
        layer.trainable = False
    for layer in base_model.layers[frozen_layers:]:
        layer.trainable = True

    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(projection_dim)(x)

    return Model(inp, x, name="resnet50_encoder")


# === Full Model ===
def build_model(sequence_len=config.SEQUENCE_LEN,
                cnn_input_shape=(64, 64, 1),
                resnet_input_shape=(64, 64, 3),
                feature_dim=config.FEATURE_DIM,
                tkan_hidden_dim=config.TKAN_HIDDEN_DIM,
                num_classes=config.NUM_CLASSES,
                dropout_rate=config.DROPOUT_RATE):
    """
    Dual-stream model: Custom CNN + ResNet50 → TKAN → Classification
    """
    # === Dual Inputs ===
    seq_gray_input = Input(shape=(sequence_len, *cnn_input_shape), name="seq_gray")
    seq_rgb_input  = Input(shape=(sequence_len, *resnet_input_shape), name="seq_rgb")

    # === Encoders ===
    cnn_encoder = build_custom_cnn_encoder(input_shape=cnn_input_shape)
    resnet_encoder = build_resnet50_encoder(input_shape=resnet_input_shape)

    cnn_features = TimeDistributed(cnn_encoder)(seq_gray_input)  # (T, feature_dim)
    resnet_features = TimeDistributed(resnet_encoder)(seq_rgb_input)  # (T, feature_dim)

    # === Feature Fusion ===
    fused = Concatenate()([cnn_features, resnet_features])  # (T, 2*feature_dim)
    fused_proj = Dense(feature_dim, activation='relu')(fused)  # Project to (T, feature_dim)

    # === TKAN Temporal Modeling ===
    x = TKAN(
        units=tkan_hidden_dim,
        sub_kan_configs=[0, 1, 2, 3, 4],
        return_sequences=True,
        use_bias=True
    )(fused_proj)

    x = Dropout(dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax', name="identity_output")(x)

    return Model([seq_gray_input, seq_rgb_input], out, name="dual_encoder_tkan_model")
