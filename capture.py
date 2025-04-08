# capture.py
import cv2
import mediapipe as mp
import csv
import os
import numpy as np

CSV_FILE = "hand_gesture_data.csv"

# Create CSV file with header if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"]
        # 28x28 = 784 pixels
        for i in range(1, 785):
            header.append(f"pixel{i}")
        writer.writerow(header)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def apply_augmentation(image):
    """Apply random augmentations to the image."""
    augmented_images = []
    
    # Original image
    augmented_images.append(image)
    
    # Rotation
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        augmented_images.append(rotated)
    
    # Brightness variation
    for alpha in [0.8, 1.2]:
        bright = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        augmented_images.append(bright)
    
    # Add noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    augmented_images.append(noisy)
    
    return augmented_images

def preprocess_hand_image(hand_img):
    """Preprocess the hand image for better recognition."""
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hand_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    hand_eq = clahe.apply(hand_gray)
    
    # Apply Gaussian blur to reduce noise
    hand_blur = cv2.GaussianBlur(hand_eq, (5,5), 0)
    
    # Resize to target size
    hand_resized = cv2.resize(hand_blur, (28, 28))
    return hand_resized

cap = cv2.VideoCapture(0)
print("Press 'c' to capture gesture sample, 'q' to quit.")

samples_per_gesture = 0
MAX_SAMPLES_PER_GESTURE = 50  # Limit samples per gesture to avoid imbalance

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

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

    # Display sample count
    cv2.putText(frame, f"Samples: {samples_per_gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Capture - Press 'c' to capture, 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        if results.multi_hand_landmarks and samples_per_gesture < MAX_SAMPLES_PER_GESTURE:
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

            # Get label from user
            label = input("Enter label for captured gesture: ")
            
            # Process and augment the image
            processed_img = preprocess_hand_image(hand_img)
            augmented_images = apply_augmentation(processed_img)
            
            # Save original and augmented images
            for aug_img in augmented_images:
                flattened = aug_img.flatten()
                with open(CSV_FILE, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([label] + flattened.tolist())
            
            samples_per_gesture += len(augmented_images)
            print(f"Saved {len(augmented_images)} samples. Total samples: {samples_per_gesture}")
            
            if samples_per_gesture >= MAX_SAMPLES_PER_GESTURE:
                print(f"Maximum samples ({MAX_SAMPLES_PER_GESTURE}) reached for this gesture.")
                samples_per_gesture = 0  # Reset for next gesture
        else:
            print("No hand detected. Try again.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
