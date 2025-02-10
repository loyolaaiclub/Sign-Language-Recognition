import json
import numpy as np
import os

def json_to_npz(json_file, output_npz):
    # Load JSON data from file
    with open("../MS-ASL/MSASL_val.json", 'r') as f:
        data = json.load(f)
        
    
    # file path ../MS-ASL/MSASL_train.json
    
    features = []
    labels = []
    
    # For each record, extract features and label.
    # Here we use: start_time, end_time, fps, width, height, and the four box values.
    for record in data:
        start_time = record.get("start_time", 0.0)
        end_time = record.get("end_time", 0.0)
        fps = record.get("fps", 0.0)
        width = record.get("width", 0.0)
        height = record.get("height", 0.0)
        box = record.get("box", [0.0, 0.0, 0.0, 0.0])
        
        # Create a feature vector (total of 9 numerical features)
        feature_vector = [start_time, end_time, fps, width, height] + box
        features.append(feature_vector)
        
        # Use the cleaned text as the label
        labels.append(record.get("clean_text", ""))
    
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels)
    
    # Save both features and labels into an NPZ file
    np.savez_compressed(output_npz, features=features, labels=labels)
    print(f"Saved NPZ file to {output_npz}")

if __name__ == "__main__":
    json_file = "dataset.json"  # Path to your JSON file
    output_npz = "dataset.npz"
    json_to_npz(json_file, output_npz)
