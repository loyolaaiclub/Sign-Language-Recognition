import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('collected_data_model.h5')

# Define the labels (make sure these match the order during training)
labels = ['hello', 'bye', 'goodbye']  # Add more based on your dataset

def preprocess_frame(frame):
    """
    Preprocess the input frame for prediction.
    - Convert to grayscale
    - Resize to (112, 112)
    - Normalize pixel values
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (112, 112))         # Resize to (112, 112)
    normalized = resized.astype('float32') / 255.0  # Normalize pixel values
    return normalized.reshape(1, 112, 112, 1)      # Add batch and channel dimensions

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return

    print("Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam")
            break

        # Flip frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Preprocess the frame
        preprocessed = preprocess_frame(frame)

        # Predict the gesture
        predictions = model.predict(preprocessed, verbose=0)
        predicted_label = labels[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Display prediction on the frame
        text = f"Prediction: {predicted_label} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the webcam feed
        cv2.imshow("Sign Language Detector", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
