import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def load_dataset(npz_file):
    data = np.load(npz_file, allow_pickle=True) # Load the dataset
    features = data['features'] # Extract features
    labels = data['labels']     # Extract labels
    return features, labels     # Return features and labels

def preprocess_data(features, labels):
    # Encode string labels to integer indices
    le = LabelEncoder() # Create a LabelEncoder
    integer_labels = le.fit_transform(labels) # Fit and transform labels
    # Convert to one-hot encoding for classification
    categorical_labels = to_categorical(integer_labels) # Convert to one-hot encoding
    return features, categorical_labels, le.classes_ # Return classes for reference

def build_model(input_dim, num_classes): 
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim), # input_dim is the number of features
        BatchNormalization(), # Normalize and scale input
        Dropout(0.3), # Randomly set 30% of the input units to 0
        Dense(32, activation='relu'), # Rectified Linear Unit Activation Function
        BatchNormalization(), # Normalize and scale input
        Dropout(0.3), # Randomly set 30% of the input units to 0
        Dense(num_classes, activation='softmax') # Output layer with softmax activation
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    npz_file = "dataset.npz" # Path to the NPZ file
    features, labels = load_dataset(npz_file) # Load the dataset
    features, categorical_labels, classes = preprocess_data(features, labels) # Preprocess the data
    
    print(f"Loaded {features.shape[0]} samples with {features.shape[1]} features each.") # Print the number of samples and features
    print(f"Number of classes: {len(classes)}; Classes: {classes}") # Print the number of classes and their names
    
    model = build_model(features.shape[1], len(classes)) # Build the model
    model.summary() # Print a summary of the model
    
    model.fit(features, categorical_labels, epochs=20, batch_size=16, validation_split=0.2) # Train the model
    model.save("json_dataset_model.h5") # Save the model
    print("Model training complete. Saved as 'json_dataset_model.h5'.") # Print a completion message
