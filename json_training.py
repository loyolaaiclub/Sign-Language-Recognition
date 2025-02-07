import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def load_json_data(json_file):
    """
    Loads JSON data from a file and extracts a feature vector and label for each record.
    Feature vector includes:
      - start_time
      - end_time
      - fps
      - width
      - height
      - box values (expects four values)
    Label is taken from the "clean_text" field.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    features = []
    labels = []

    for record in data:
        # Extract numerical features (using default values if key is missing)
        start_time = record.get("start_time", 0.0)
        end_time = record.get("end_time", 0.0)
        fps = record.get("fps", 0.0)
        width = record.get("width", 0.0)
        height = record.get("height", 0.0)
        box = record.get("box", [0.0, 0.0, 0.0, 0.0])  # Expecting four values
        
        # Create a feature vector (total 9 features)
        feature_vector = [start_time, end_time, fps, width, height] + box
        features.append(feature_vector)

        # Use the cleaned text as the label
        labels.append(record.get("clean_text", "unknown"))

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels)
    return features, labels

def preprocess_data(features, labels):
    """
    Encodes string labels to integers and converts to one-hot encoding.
    """
    le = LabelEncoder()
    integer_labels = le.fit_transform(labels)
    categorical_labels = to_categorical(integer_labels)
    return features, categorical_labels, le.classes_

def build_model(input_dim, num_classes):
    """
    Builds a simple feed-forward neural network for classification.
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Path to your JSON dataset (adjust as needed)
    json_file = "./MS-ASL/MSASL_train.json"
    features, labels = load_json_data(json_file)
    
    features, categorical_labels, classes = preprocess_data(features, labels)
    
    print(f"Loaded {features.shape[0]} samples with {features.shape[1]} features each.")
    print(f"Number of classes: {len(classes)}; Classes: {classes}")
    
    # Build and train the model
    model = build_model(features.shape[1], len(classes))
    model.summary()
    
    model.fit(features, categorical_labels, epochs=20, batch_size=16, validation_split=0.2)
    
    # Save the trained model
    model.save("json_training_model.h5")
    print("Model training complete. Saved as 'json_training_model.h5'.")
