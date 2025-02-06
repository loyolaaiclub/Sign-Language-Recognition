import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def load_collected_data(data_dir):
    """
    Loads keypoint data from CSV files organized in subdirectories (one per label).
    Each CSV file represents one sample clip (a sequence of frames), and each row in the CSV
    corresponds to the flattened keypoints for one frame.
    """
    X, y = [], []
    # Get only subdirectories (each corresponding to a label)
    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    label_map = {label: idx for idx, label in enumerate(labels)}
    
    for label in labels:
        label_path = os.path.join(data_dir, label)
        for file in os.listdir(label_path):
            if file.endswith('.csv'):
                csv_path = os.path.join(label_path, file)
                df = pd.read_csv(csv_path)
                # Drop the 'sample' and 'frame' columns if they exist, leaving only keypoint features.
                for col in ['sample', 'frame']:
                    if col in df.columns:
                        df = df.drop(columns=[col])
                # Each row represents the keypoint features for one frame.
                data = df.values  # shape: (num_frames, num_features)
                X.append(data)
                # Create a label for each frame in the clip
                y.append([label_map[label]] * data.shape[0])
                
    # Combine all frames from all CSV files into a single dataset
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y, label_map

def preprocess_collected_data(X, y):
    """
    Preprocess the keypoint data. In this example, we simply cast to float and one-hot encode the labels.
    Depending on your keypoint scale, you might wish to add normalization.
    """
    X = X.astype('float32')
    y = to_categorical(y)
    return X, y

def build_classification_model(input_shape, num_classes):
    """
    Build a simple fully-connected network for keypoint classification.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Set the path to your data directory (organized by label)
    data_directory = "data"
    
    # Load and preprocess the keypoint data from CSV files
    X, y, label_mapping = load_collected_data(data_directory)
    X, y = preprocess_collected_data(X, y)
    
    print(f"Loaded dataset with {X.shape[0]} frames and {len(label_mapping)} classes.")
    print(f"Label mapping: {label_mapping}")
    
    # Build and train the model. The input shape now equals the number of keypoint features.
    input_shape = (X.shape[1],)
    num_classes = len(label_mapping)
    model = build_classification_model(input_shape, num_classes)
    
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    
    # Save the trained model
    model.save("collected_data_model.keras")
    print("Model training complete. Saved as 'collected_data_model.keras'.")
