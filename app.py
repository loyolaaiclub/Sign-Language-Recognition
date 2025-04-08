# app.py
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import tensorflow as tf
import os

# Map numeric labels to letters (1-26 to A-Z)
LABEL_MAP = {i: chr(ord('A') + i) for i in range(26)}

MODEL_FILE = "asl_model.h5"

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_FILE)
print("Model loaded successfully")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# Get screen size for cursor mapping
screen_width, screen_height = pyautogui.size()
print(f"Screen size: {screen_width}x{screen_height}")

# Initialize gesture smoothing
SMOOTHING_WINDOW = 5
recent_predictions = []

def preprocess_hand_image(hand_img):
    """Preprocess the hand image for the model"""
    if hand_img.size == 0:
        return None
        
    # Convert to grayscale and resize
    hand_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hand_eq = clahe.apply(hand_gray)
    
    # Resize to model input size
    hand_resized = cv2.resize(hand_eq, (28, 28))
    
    # Normalize and reshape for model input
    processed = hand_resized.astype('float32') / 255.0
    processed = np.expand_dims(processed, axis=(0, -1))
    return processed, hand_resized

def get_smoothed_prediction(pred):
    """Apply smoothing to predictions"""
    recent_predictions.append(pred)
    if len(recent_predictions) > SMOOTHING_WINDOW:
        recent_predictions.pop(0)
    # Return most common prediction in window
    return max(set(recent_predictions), key=recent_predictions.count)

def draw_prediction_info(frame, letter, confidence, x, y):
    """Draw prediction information on the frame"""
    # Draw background rectangle
    cv2.rectangle(frame, (x, y), (x + 150, y + 60), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, y), (x + 150, y + 60), (0, 255, 0), 2)
    
    # Draw letter and confidence
    cv2.putText(frame, f"Letter: {letter}", (x + 10, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Conf: {confidence:.2f}", (x + 10, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def display_current_sign(frame, letter, processed_img, confidence):
    """Display the current sign more prominently"""
    # Create a larger display area on the right side of the frame
    height, width = frame.shape[:2]
    display_width = min(300, width // 3)
    display_height = min(300, height // 2)
    
    # Create a background for the sign display
    sign_area = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
    # Display the processed hand image (upscaled)
    if processed_img is not None:
        # Convert grayscale to BGR for display
        hand_display = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        # Resize to fit in the display area
        display_size = min(display_width, display_height) - 40
        hand_display = cv2.resize(hand_display, (display_size, display_size))
        
        # Calculate position to center the hand image
        x_offset = (display_width - display_size) // 2
        y_offset = 20
        
        # Place the hand image in the sign area
        sign_area[y_offset:y_offset+display_size, x_offset:x_offset+display_size] = hand_display * 255
    
    # Add the letter and confidence
    letter_font_size = 2.0
    conf_font_size = 0.7
    letter_text_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, letter_font_size, 3)[0]
    letter_x = (display_width - letter_text_size[0]) // 2
    letter_y = display_height - 60
    
    cv2.putText(sign_area, letter, (letter_x, letter_y),
                cv2.FONT_HERSHEY_SIMPLEX, letter_font_size, (0, 255, 0), 3)
    
    cv2.putText(sign_area, f"Confidence: {confidence:.2f}", 
                (10, display_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, conf_font_size, (0, 255, 0), 2)
    
    # Add the sign area to the main frame
    frame[10:10+display_height, width-display_width-10:width-10] = sign_area
    
    # Add a border
    cv2.rectangle(frame, 
                  (width-display_width-10, 10), 
                  (width-10, 10+display_height), 
                  (0, 255, 0), 2)

print("App running. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks with connections
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Get hand bounding box
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min = max(min(x_coords) - 20, 0)
            x_max = min(max(x_coords) + 20, w)
            y_min = max(min(y_coords) - 20, 0)
            y_max = min(max(y_coords) + 20, h)
            
            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Extract and preprocess hand image
            hand_img = frame[y_min:y_max, x_min:x_max]
            processed_data = preprocess_hand_image(hand_img)
            
            if processed_data is not None:
                processed_img, hand_resized = processed_data
                
                # Get model prediction
                predictions = model.predict(processed_img, verbose=0)
                predicted_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_idx]
                
                # Convert numeric label to letter
                predicted_letter = LABEL_MAP[predicted_idx]
                
                # Apply smoothing
                smoothed_idx = get_smoothed_prediction(predicted_idx)
                smoothed_letter = LABEL_MAP[smoothed_idx]
                
                # Only show high confidence predictions
                if confidence > 0.6:
                    draw_prediction_info(frame, smoothed_letter, confidence, 10, 10)
                    display_current_sign(frame, smoothed_letter, hand_resized, confidence)

                    # Special actions for certain letters
                    if smoothed_letter == 'M':  # Move cursor
                        center_x = (x_min + x_max) // 2
                        center_y = (y_min + y_max) // 2
                        cursor_x = int(center_x / w * screen_width)
                        cursor_y = int(center_y / h * screen_height)
                        print(cursor_x, cursor_y)
                        pyautogui.moveTo(cursor_x, cursor_y)
                    elif smoothed_letter == 'C':  # Click
                        pyautogui.click()

    # Add help text
    cv2.putText(frame, "M: Move cursor, C: Click", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition - Press 'q' to exit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
