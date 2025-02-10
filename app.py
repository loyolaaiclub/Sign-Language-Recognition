#!/usr/bin/env python
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model (saved as a Keras file, e.g. "collected_data_model.keras")
model = load_model('collected_data_model.keras')

# Define labels (ensure the order matches training)
labels = ['hello', 'bye', 'goodbye']
include_not_speaking = True
if include_not_speaking:
    labels.append('not speaking')
confidence_threshold = 0.8 if include_not_speaking else 0.0

# Initialize MediaPipe Hands for real-time hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def preprocess_frame(frame):
    """
    Preprocess the input frame for prediction.
    Steps:
      - Convert to grayscale.
      - Resize to (112, 112).
      - Normalize pixel values.
      - Reshape to add batch and channel dimensions.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (112, 112))
    normalized = resized.astype('float32') / 255.0
    return normalized.reshape(1, 112, 112, 1)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    print("Press 'q' to exit")
    previous_prediction = None
    prediction_sequence = []

    with open("predictions.txt", "w") as file:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Process frame with MediaPipe for hand detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                cv2.putText(frame, "No hand detected", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Preprocess frame and perform prediction
            preprocessed = preprocess_frame(frame)
            predictions = model.predict(preprocessed, verbose=0)
            confidence = np.max(predictions)
            predicted_label = labels[np.argmax(predictions)]

            # Apply "not speaking" logic if below threshold
            if include_not_speaking and confidence < confidence_threshold:
                predicted_label = 'not speaking'

            # Log change in prediction
            if predicted_label != previous_prediction:
                prediction_sequence.append(predicted_label)
                file.write(predicted_label + '\n')
                file.flush()
                previous_prediction = predicted_label

            # Overlay prediction and sequence on frame
            text = f"Prediction: {predicted_label} ({confidence:.2f})"
            sequence_text = " ".join(prediction_sequence)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Sequence: " + sequence_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Sign Language Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()
    print("\nPrediction sequence saved to 'predictions.txt'.")

if __name__ == "__main__":
    main()
