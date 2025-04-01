# app.py
import cv2
import mediapipe as mp
import pyautogui
import pickle
import numpy as np
import pandas as pd  # Imported to convert features into a DataFrame

MODEL_FILE = "gesture_model.pkl"

# Load the trained model.
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)
print("DEBUG: Loaded model from", MODEL_FILE)

# Create the feature names list (pixel1 ... pixel784)
feature_names = [f"pixel{i}" for i in range(1, 785)]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# Get screen size for cursor mapping.
screen_width, screen_height = pyautogui.size()
print(f"DEBUG: Screen size: {screen_width}x{screen_height}")

print("App running. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("DEBUG: Failed to grab frame.")
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    print(f"DEBUG: Processing frame with shape: {frame.shape}")
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    predicted_label = None

    if results.multi_hand_landmarks:
        print("DEBUG: Detected hand landmarks.")
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            print(f"DEBUG: Hand {idx} x_coords: {x_coords}")
            print(f"DEBUG: Hand {idx} y_coords: {y_coords}")
            
            x_min = max(min(x_coords) - 20, 0)
            x_max = min(max(x_coords) + 20, w)
            y_min = max(min(y_coords) - 20, 0)
            y_max = min(max(y_coords) + 20, h)
            print(f"DEBUG: Hand {idx} bounding box: ({x_min}, {y_min}), ({x_max}, {y_max})")
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                print("DEBUG: Hand image is empty, skipping this hand.")
                continue

            # Convert to grayscale and resize to 28x28.
            hand_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            print(f"DEBUG: Hand gray image shape: {hand_gray.shape}")
            hand_resized = cv2.resize(hand_gray, (28, 28))
            print(f"DEBUG: Hand resized to: {hand_resized.shape}")
            flattened = hand_resized.flatten()
            print(f"DEBUG: Flattened feature vector shape: {flattened.shape}")
            features = flattened.reshape(1, -1)
            # Convert the NumPy array to a DataFrame with the correct column names.
            features_df = pd.DataFrame(features, columns=feature_names)

            try:
                predicted_label = model.predict(features_df)[0]
                print(f"DEBUG: Predicted label: {predicted_label}")
            except Exception as e:
                print("DEBUG: Prediction error:", e)
                continue

            cv2.putText(frame, f"Gesture: {predicted_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- Action Mapping ---
            # Here we assume:
            #    "3" -> move cursor based on the center of the hand bounding box
            #    "6" -> trigger a click
            #    (Any other label performs no action.)
            if str(predicted_label) == "3":
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                # Map frame coordinates to screen coordinates.
                cursor_x = int(center_x / w * screen_width)
                cursor_y = int(center_y / h * screen_height)
                print(f"DEBUG: Moving cursor to: ({cursor_x}, {cursor_y})")
                pyautogui.moveTo(cursor_x, cursor_y)
            elif str(predicted_label) == "6":
                print("DEBUG: Triggering mouse click")
                pyautogui.click()
            else:
                print("DEBUG: No action mapped for this label.")

    else:
        print("DEBUG: No hand landmarks detected.")

    cv2.imshow("App - Press 'q' to exit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("DEBUG: Exiting app loop.")
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("DEBUG: Cleaned up and closed all resources.")
