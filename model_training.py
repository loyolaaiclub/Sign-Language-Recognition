import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Rescaling
from tensorflow.keras.utils import to_categorical

def load_collected_data(data_dir):
    """
    Loads training data from npz files saved in a folder structure:
      data/
         ├── label1/
         │     ├── label1_0.npz
         │     └── label1_1.npz
         └── label2/
               ├── label2_0.npz
               └── label2_1.npz
    Each npz file contains keys 'frames' (and optionally 'landmarks').
    """
    X, y = [], []
    labels = sorted(os.listdir(data_dir))  # Ensure consistent label ordering
    label_map = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        label_path = os.path.join(data_dir, label)
        for file in os.listdir(label_path):
            if file.endswith('.npz'):
                npz_file = np.load(os.path.join(label_path, file))
                sample_frames = npz_file['frames']
                X.append(sample_frames)
                # Assign the same label to all frames in the sample
                y.append(np.full(len(sample_frames), label_map[label]))
    X = np.vstack(X)  # Combine all frames into one array
    y = np.hstack(y)  # Combine all labels into one array
    return X, y, label_map

def preprocess_collected_data(X, y):
    """
    Preprocesses the image data and converts labels to one-hot encoding.
    """
    X = X.astype('float32')
    X = X[..., np.newaxis]  # Add channel dimension for grayscale images
    y = to_categorical(y)
    return X, y

def build_classification_model(input_shape, num_classes):
    """
    Builds and compiles a simple CNN model for gesture classification.
    """
    model = Sequential([
        Rescaling(1./255, input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Set the path to the directory containing the collected data
    data_directory = "data"

    # Load and preprocess the data
    X, y, label_mapping = load_collected_data(data_directory)
    X, y = preprocess_collected_data(X, y)

    print(f"Loaded dataset with {X.shape[0]} samples and {len(label_mapping)} classes.")
    print(f"Label mapping: {label_mapping}")

    # Build the model
    input_shape = X.shape[1:]  # e.g., (112, 112, 1)
    num_classes = len(label_mapping)
    model = build_classification_model(input_shape, num_classes)

    # Define callbacks for early stopping and model checkpointing
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('collected_data_model.h5', save_best_only=True)
    ]

    # Train the model
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=callbacks)

    # Save the trained model
    model.save("collected_data_model_final.h5")
    print("Model training complete. Saved as 'collected_data_model_final.h5'.")
