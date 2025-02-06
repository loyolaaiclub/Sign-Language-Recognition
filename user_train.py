import cv2
import os
import numpy as np
from datetime import datetime
import time  # For warmup delay
import platform  # To detect OS type
import mediapipe as mp

def collect_data():
    # ---------------------------- User Setup ----------------------------------
    label = input("Enter the sign label to record (e.g., 'hello', 'bye'): ").lower()
    samples_needed = 30
    clip_duration = 2  # seconds

    # Create directories for saving data
    data_path = os.path.join('data', label)
    os.makedirs(data_path, exist_ok=True)

    # Determine the starting index for new files
    existing_files = [f for f in os.listdir(data_path) if f.startswith(label) and f.endswith('.npz')]
    if existing_files:
        existing_indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
        start_index = max(existing_indices) + 1
    else:
        start_index = 0

    # ---------------------------- Camera Setup ----------------------------------
    system_platform = platform.system().lower()
    if system_platform == 'darwin':  # macOS
        backend = cv2.CAP_AVFOUNDATION
    elif system_platform == 'windows':
        backend = cv2.CAP_DSHOW
    else:  # Linux and others
        backend = cv2.CAP_V4L2

    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        raise IOError(
            "Cannot open webcam. Ensure:\n"
            "1. Camera permissions are granted (check your OS settings)\n"
            "2. No other apps are using the camera\n"
            "3. Your camera is properly connected or built-in"
        )

    # Warmup the camera
    print("\nInitializing camera...")
    for _ in range(5):
        cap.read()
    time.sleep(1)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    if system_platform != 'darwin':  # Skip codec setting for macOS
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    print(f"\n=== Collecting {samples_needed} samples for '{label}' ===")
    print("Press 's' to start recording, 'q' to quit early")

    # ----------------------- Initialize MediaPipe Hands -----------------------
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # -------------------------- Data Collection Loop --------------------------
    sample_count = 0
    recording = False
    frames = []         # To store low-res grayscale frames for the sample
    landmarks_data = [] # To store corresponding hand landmarks data
    window_open = True

    while sample_count < samples_needed and window_open:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture error - trying to recover...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0, backend)
            continue

        # Mirror the frame for a more natural interaction
        frame = cv2.flip(frame, 1)

        # Create a copy for display (with mediapipe drawings)
        display_frame = frame.copy()

        # Convert the frame from BGR to RGB as required by MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw detected hand landmarks on the display copy
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display instructions on the preview window
        text = f"Sample {sample_count + 1}/{samples_needed} | Press 's' to start"
        cv2.putText(display_frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        try:
            cv2.imshow('Data Collector', display_frame)
        except cv2.error as e:
            print(f"Display error: {str(e)}")
            window_open = False
            break

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s') and not recording:
            recording = True
            frames = []
            landmarks_data = []
            start_time = datetime.now()
            print(f"Recording sample {sample_count + 1}...")

        if recording:
            # --------------------- Save Frame Data ---------------------
            # For storage, use the original frame (without drawings)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (112, 112))  # Resize for compact storage
            frames.append(small)

            # --------------------- Extract Landmarks ---------------------
            # If hands are detected, store each hand's normalized landmark coordinates
            if results.multi_hand_landmarks:
                frame_landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_points = []
                    for lm in hand_landmarks.landmark:
                        hand_points.append([lm.x, lm.y, lm.z])
                    frame_landmarks.append(hand_points)
            else:
                frame_landmarks = []
            landmarks_data.append(frame_landmarks)

            # Check if the recording duration has been reached
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= clip_duration:
                output_path = os.path.join(data_path, f"{label}_{start_index + sample_count}.npz")
                # Save both frames and landmarks. The landmarks are stored as an object array.
                np.savez_compressed(output_path,
                                    frames=np.array(frames),
                                    landmarks=np.array(landmarks_data, dtype=object))
                sample_count += 1
                recording = False
                print(f"Saved sample {sample_count}")

    # ------------------------ Cleanup Resources ------------------------------
    cap.release()
    cv2.destroyAllWindows()
    hands.close()  # Properly close the MediaPipe Hands object
    time.sleep(0.5)  # Allow windows to close

    print(f"\nData collection complete! {sample_count} samples saved to {data_path}")

if __name__ == "__main__":
    collect_data()
