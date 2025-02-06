import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Rescaling
from tensorflow.keras.utils import to_categorical

def load_collected_data(data_dir):
    X, y = [], []
    labels = sorted(os.listdir(data_dir))  # Ensure consistent label ordering
    label_map = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        label_path = os.path.join(data_dir, label)
        for file in os.listdir(label_path):
            if file.endswith('.npz'):
                data = np.load(os.path.join(label_path, file))['arr_0']
                X.append(data)
                y.append([label_map[label]] * len(data))

    X = np.vstack(X)  # Combine all frames
    y = np.hstack(y)  # Combine all labels
    return X, y, label_map

def preprocess_collected_data(X, y):
    # Instead of normalizing here, we include normalization in the model with Rescaling.
    X = X.astype('float32')
    # Add channel dimension for grayscale images
    X = X[..., np.newaxis]
    # One-hot encode labels
    y = to_categorical(y)
    return X, y

def build_classification_model(input_shape, num_classes):
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
    # Set the path to the directory where the collected data is stored
    data_directory = "data"

    # Load and preprocess the data
    X, y, label_mapping = load_collected_data(data_directory)
    X, y = preprocess_collected_data(X, y)

    # Print dataset details
    print(f"Loaded dataset with {X.shape[0]} samples and {len(label_mapping)} classes.")
    print(f"Label mapping: {label_mapping}")

    # Build the model
    input_shape = X.shape[1:]  # Example: (112, 112, 1)
    num_classes = len(label_mapping)
    model = build_classification_model(input_shape, num_classes)

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('collected_data_model.h5', save_best_only=True)
    ]

    # Train the model
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=callbacks)

    # Save the trained model (if not already saved via ModelCheckpoint)
    model.save("collected_data_model_final.h5")
    print("Model training complete. Saved as 'collected_data_model_final.h5'.")
