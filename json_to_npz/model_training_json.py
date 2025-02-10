import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def load_dataset(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    features = data['features']
    labels = data['labels']
    return features, labels

def preprocess_data(features, labels):
    # Encode string labels to integer indices
    le = LabelEncoder()
    integer_labels = le.fit_transform(labels)
    # Convert to one-hot encoding for classification
    categorical_labels = to_categorical(integer_labels)
    return features, categorical_labels, le.classes_

def build_model(input_dim, num_classes):
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
    npz_file = "dataset.npz"
    features, labels = load_dataset(npz_file)
    features, categorical_labels, classes = preprocess_data(features, labels)
    
    print(f"Loaded {features.shape[0]} samples with {features.shape[1]} features each.")
    print(f"Number of classes: {len(classes)}; Classes: {classes}")
    
    model = build_model(features.shape[1], len(classes))
    model.summary()
    
    model.fit(features, categorical_labels, epochs=20, batch_size=16, validation_split=0.2)
    model.save("json_dataset_model.h5")
    print("Model training complete. Saved as 'json_dataset_model.h5'.")
