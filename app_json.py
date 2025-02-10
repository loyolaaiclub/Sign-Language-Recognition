import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

# -------------------------------
# LOAD MODEL & SETUP LABELS
# -------------------------------
model = load_model('collected_data_model.h5')

# IMPORTANT: Ensure that the order here matches the order of the labels used during training.
# For example, if your training labels (sorted) were ['absent', 'come on', 'help'] then:
labels = ['absent', 'come on', 'help']

# Optionally include a "not speaking" category if the prediction confidence is low.
include_not_speaking = True
if include_not_speaking:
    labels.append('not speaking')
confidence_threshold = 0.8 if include_not_speaking else 0.0

# -------------------------------
# SETUP MEDIAPIPE HANDS
# -------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------
# PREPROCESSING FUNCTION
# -------------------------------
def preprocess_frame(frame):
    """
    Preprocesses the input frame:
      - Converts to grayscale.
      - Resizes to 112x112.
      - Normalizes pixel values.
      - Reshapes to add the batch and channel dimensions.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (112, 112))
    normalized = resized.astype('float32') / 255.0
    return normalized.reshape(1, 112, 112, 1)

# -------------------------------
# MAIN APPLICATION
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access the webcam.")
        return

    print("Starting webcam. Press 'q' to exit.")
    previous_prediction = None
    prediction_sequence = []

    # Open a text file to log predictions
    with open("predictions.txt", "w") as file:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read a frame from the webcam.")
                break

            # Flip frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)

            # Process the frame with MediaPipe for hand detection
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

            # If confidence is below the threshold, set label to "not speaking"
            if include_not_speaking and confidence < confidence_threshold:
                predicted_label = 'not speaking'

            # Log the prediction if it has changed
            if predicted_label != previous_prediction:
                prediction_sequence.append(predicted_label)
                file.write(predicted_label + '\n')
                file.flush()
                previous_prediction = predicted_label

            # Overlay prediction and sequence on the frame
            text = f"Prediction: {predicted_label} ({confidence:.2f})"
            sequence_text = " ".join(prediction_sequence)
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Sequence: " + sequence_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Sign Language Detector", frame)

            # Break loop on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()
    print("\nPrediction sequence saved to 'predictions.txt'.")

if __name__ == "__main__":
    main()
