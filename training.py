#!/usr/bin/env python
"""
train_model.py

This script loads the processed NPZ data from the folder structure created by json_process.py.
Each subfolder in DATA_FOLDER corresponds to a gesture class. The script builds a dataset,
maps each gesture (folder name) to a label index, and trains a Conv3D network.
Data augmentation is applied on the fly to help with very limited examples per class.
Callbacks (ModelCheckpoint and EarlyStopping) are used to help the model generalize.

Usage:
    python train_model.py
"""
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# CONFIGURATION
# -------------------------------
IMG_SIZE = (112, 112)
NUM_FRAMES = 5
DATA_FOLDER = "data"  # Must be the same folder as created by prepare_data.py
BATCH_SIZE = 8
EPOCHS = 50

# -------------------------------
# DATA LOADING FUNCTION
# -------------------------------
def load_npz_data(data_folder=DATA_FOLDER):
    """
    Walk through DATA_FOLDER and load each NPZ file.
    Returns:
      X: list of processed clips, each of shape (NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 1)
      y: list of gesture labels (folder names)
    """
    X = []
    y = []
    gesture_folders = [os.path.join(data_folder, d) for d in os.listdir(data_folder)
                       if os.path.isdir(os.path.join(data_folder, d))]
    for folder in gesture_folders:
        # Use the folder name as label
        label = os.path.basename(folder)
        npz_files = glob.glob(os.path.join(folder, "*.npz"))
        for file in npz_files:
            try:
                data = np.load(file, allow_pickle=True)
                clip = data["clip"]
                if clip.shape != (NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 1):
                    print(f"[WARNING] Unexpected shape in {file}: {clip.shape}")
                    continue
                X.append(clip)
                y.append(label)
            except Exception as e:
                print(f"[ERROR] Failed to load {file}: {e}")
    X = np.array(X)
    return X, y

# -------------------------------
# DATA AUGMENTATION & GENERATOR
# -------------------------------
# Create an ImageDataGenerator for augmentation on individual frames.
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False
)

def augment_video_clip(clip):
    """
    Apply the same random transformation to each frame in a video clip.
    clip: NumPy array of shape (NUM_FRAMES, height, width, channels)
    Returns the augmented clip.
    """
    transform_params = datagen.get_random_transform(clip[0].shape)
    augmented = np.empty_like(clip)
    for i in range(clip.shape[0]):
        augmented[i] = datagen.apply_transform(clip[i], transform_params)
    return augmented

def data_generator(X, y, batch_size=BATCH_SIZE, augment=True):
    """
    Generator that yields batches of video clips and labels.
    """
    num_samples = len(X)
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_X = []
            batch_y = []
            for idx in batch_idx:
                clip = X[idx]
                if augment:
                    clip = augment_video_clip(clip)
                batch_X.append(clip)
                batch_y.append(y[idx])
            yield np.array(batch_X), np.array(batch_y)

# -------------------------------
# MAIN TRAINING FUNCTION
# -------------------------------
def main():
    # Load data from processed NPZ files
    X, y = load_npz_data(DATA_FOLDER)
    print(f"[INFO] Loaded {len(X)} samples.")

    if len(X) == 0:
        print("[ERROR] No data loaded. Exiting.")
        return

    # Build a label-to-index mapping based on the gesture folder names
    unique_labels = sorted(list(set(y)))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"[INFO] Found {len(unique_labels)} unique gesture classes: {unique_labels}")

    # Convert string labels to integer indices and then to one-hot vectors
    y_indices = np.array([label_to_index[label] for label in y])
    num_classes = len(unique_labels)
    y_cat = to_categorical(y_indices, num_classes)

    # Build a simple Conv3D model
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation="relu", input_shape=(NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 1)),
        MaxPooling3D(pool_size=(1, 2, 2)),
        Conv3D(64, (3, 3, 3), activation="relu"),
        MaxPooling3D(pool_size=(1, 2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    steps_per_epoch = max(1, len(X) // BATCH_SIZE)

    # Set up callbacks: checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(
        filepath="model_checkpoint.h5",
        monitor="loss",
        save_best_only=True,
        verbose=1
    )
    early_stopping_callback = EarlyStopping(
        monitor="loss",
        patience=5,
        verbose=1
    )

    # Train the model using the data generator (with augmentation)
    model.fit(
        data_generator(X, y_cat, batch_size=BATCH_SIZE, augment=True),
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    model.save("asl_model.h5")
    print("[INFO] Model saved as asl_model.h5")

if __name__ == "__main__":
    main()
