#!/usr/bin/env python
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the trained model (from the CSV-based training pipeline)
model = tf.keras.models.load_model('collected_data_model.h5')

# Dynamically load labels from the data directory (each subfolder corresponds to a label)
data_dir = "data"
labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print("Labels:", labels)

# Initialize MediaPipe Holistic and Drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """
    Extracts and concatenates keypoints from the holistic results.
    If a particular landmark is not detected, returns an array of zeros.
    Expected dimensions:
      - Pose: 33 landmarks x 4 values = 132
      - Face: 468 landmarks x 3 values = 1404
      - Left hand: 21 landmarks x 3 values = 63
      - Right hand: 21 landmarks x 3 values = 63
    Total expected features: 132 + 1404 + 63 + 63 = 1662.
    """
    # Pose
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)
    
    # Face
    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3)
    
    # Left Hand
    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(21 * 3)
    
    # Right Hand
    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(21 * 3)
    
    # Concatenate all keypoints into one feature vector
    return np.concatenate([pose, face, left_hand, right_hand])

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    # Initialize holistic with reasonable detection/tracking confidence thresholds
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break
            
            # Flip the frame horizontally for a mirror-effect
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            # Draw landmarks on the frame for visual feedback
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Extract keypoints and create input for the model
            keypoints = extract_keypoints(results)
            input_data = keypoints.reshape(1, -1)  # Shape must match model's input shape (1, 1662)

            # Predict gesture from the keypoints
            predictions = model.predict(input_data, verbose=0)
            confidence = np.max(predictions)
            predicted_label = labels[np.argmax(predictions)]

            # Overlay prediction text on the frame
            text = f"Prediction: {predicted_label} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("ASL Recognition", frame)
            
            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
