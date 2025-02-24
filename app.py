#!/usr/bin/env python
"""
app.py

This script uses a webcam to perform live prediction for ASL gestures using a trained
Conv3D model. It collects a fixed number of frames to form a clip, preprocesses the clip,
and then passes it to the model to obtain a prediction. The predicted label is then
overlaid on the video feed.
"""

import cv2
import numpy as np
import tensorflow as tf
import os

# -------------------------------
# CONFIGURATION
# -------------------------------
IMG_SIZE = (112, 112)         # Target size for each frame
NUM_FRAMES = 5                # Number of frames per clip (temporal depth)
DATA_FOLDER = "data"          # Folder that contains labels.txt
MODEL_PATH = "asl_model.h5"   # Path to the trained model file
include_not_speaking = False  # Set to True if you appended a "not speaking" label during training

# -------------------------------
# LOAD MODEL & LABELS
# -------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# Load labels from labels.txt
labels_file_path = os.path.join(DATA_FOLDER, "labels.txt")
with open(labels_file_path, "r", encoding="utf-8") as file:
    labels = [line.strip() for line in file.readlines()]

# Append 'not speaking' label if needed
if include_not_speaking:
    labels.append("not speaking")
print("Loaded labels:", labels)

# -------------------------------
# PREPROCESSING FUNCTION FOR A CLIP
# -------------------------------
def preprocess_clip(frames):
    """
    Processes a list of frames and returns a 5D tensor for prediction:
      - Converts each frame to grayscale.
      - Resizes each frame to IMG_SIZE.
      - Normalizes pixel values to [0, 1].
      - Stacks frames to produce shape (NUM_FRAMES, height, width, 1).
      - Adds a batch dimension, resulting in shape (1, NUM_FRAMES, height, width, 1).
    """
    processed_frames = []
    for frame in frames:
        # Convert BGR (from cv2) to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize the frame
        resized = cv2.resize(gray, IMG_SIZE)
        # Normalize pixel values
        normalized = resized.astype("float32") / 255.0
        processed_frames.append(normalized)
    
    # Stack frames along the time dimension: (NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1])
    clip = np.stack(processed_frames, axis=0)
    # Add channel dimension: (NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 1)
    clip = np.expand_dims(clip, axis=-1)
    # Add batch dimension: (1, NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 1)
    clip = np.expand_dims(clip, axis=0)
    return clip

# -------------------------------
# MAIN APPLICATION LOOP
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("Starting live ASL prediction. Press 'q' to exit.")
    
    frame_buffer = []      # Buffer to hold NUM_FRAMES frames
    last_prediction = "None"  # To store the most recent prediction

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Append current frame to buffer
        frame_buffer.append(frame)

        # When we have collected enough frames, process the clip
        if len(frame_buffer) == NUM_FRAMES:
            clip = preprocess_clip(frame_buffer)
            predictions = model.predict(clip, verbose=0)
            confidence = np.max(predictions)
            predicted_index = np.argmax(predictions)
            last_prediction = labels[predicted_index]
            
            # (Optional) Print the confidence for debugging:
            # print(f"Predicted: {last_prediction} with confidence {confidence:.2f}")

            # Clear the buffer for the next clip. Alternatively, you could implement a sliding window.
            frame_buffer = []

        # Overlay the last prediction on the current frame
        cv2.putText(frame, f"Prediction: {last_prediction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ASL Live Prediction", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
