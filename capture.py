# capture.py
import cv2
import mediapipe as mp
import csv
import os

CSV_FILE = "hand_gesture_data.csv"

# Create CSV file with header if it doesn't exist.
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"]
        # 28x28 = 784 pixels
        for i in range(1, 785):
            header.append(f"pixel{i}")
        writer.writerow(header)

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
print("Press 'c' to capture gesture sample, 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # If a hand is detected, draw landmarks and a bounding box.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min = max(min(x_coords) - 20, 0)
            x_max = min(max(x_coords) + 20, w)
            y_min = max(min(y_coords) - 20, 0)
            y_max = min(max(y_coords) + 20, h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow("Capture - Press 'c' to capture, 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min = max(min(x_coords) - 20, 0)
            x_max = min(max(x_coords) + 20, w)
            y_min = max(min(y_coords) - 20, 0)
            y_max = min(max(y_coords) + 20, h)
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                print("Hand region is empty. Try again.")
                continue
            # Convert the hand image to grayscale.
            hand_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            # Resize to 28x28.
            hand_resized = cv2.resize(hand_gray, (28, 28))
            # Flatten the image into a 1D array.
            flattened = hand_resized.flatten()
            # Prompt for label (e.g., "3", "6", etc.).
            label = input("Enter label for captured gesture: ")
            # Append the data to the CSV.
            with open(CSV_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([label] + flattened.tolist())
            print("Sample captured.")
        else:
            print("No hand detected. Try again.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
