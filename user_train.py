import cv2
import os
import numpy as np
from datetime import datetime
import time
import platform
import mediapipe as mp

def collect_data():
    # ---------------------------- User Setup ----------------------------------
    label = input("Enter the sign label to record (e.g., 'hello', 'bye'): ").lower()
    samples_needed = 30
    clip_duration = 2  # seconds

    # Create directory for saving data
    data_path = os.path.join('data', label)
    os.makedirs(data_path, exist_ok=True)

    # Determine starting index based on existing files
    existing_files = [f for f in os.listdir(data_path) if f.startswith(label) and f.endswith('.npz')]
    start_index = max([int(f.split('_')[-1].split('.')[0]) for f in existing_files], default=-1) + 1

    # ---------------------------- Camera Setup ----------------------------------
    system_platform = platform.system().lower()
    if system_platform == 'darwin':
        backend = cv2.CAP_AVFOUNDATION
    elif system_platform == 'windows':
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_V4L2

    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        raise IOError("Cannot open webcam. Check camera permissions and connection.")

    print("\nInitializing camera...")
    for _ in range(5):
        cap.read()
    time.sleep(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    if system_platform != 'darwin':
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

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

    sample_count = 0
    recording = False
    frames = []         # Stores resized grayscale frames
    landmarks_data = [] # Stores corresponding hand landmarks
    window_open = True

    while sample_count < samples_needed and window_open:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture error, retrying...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0, backend)
            continue

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # Process frame with MediaPipe for hand detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(display_frame, "Hand detected", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No hand detected", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        text = f"Sample {sample_count + 1}/{samples_needed} | Press 's' to start"
        cv2.putText(display_frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

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
            # Countdown before starting recording (3-second countdown)
            for i in range(3, 0, -1):
                ret, countdown_frame = cap.read()
                countdown_frame = cv2.flip(countdown_frame, 1)
                cv2.putText(countdown_frame, f"Recording in {i}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.imshow('Data Collector', countdown_frame)
                cv2.waitKey(1000)
            recording = True
            frames = []
            landmarks_data = []
            start_time = datetime.now()
            print(f"Recording sample {sample_count + 1}...")

        if recording:
            # --------------------- Capture and Process Frame ---------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (112, 112))
            frames.append(small)

            # --------------------- Extract Hand Landmarks ---------------------
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

            # Check if recording duration has been reached
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= clip_duration:
                output_path = os.path.join(data_path, f"{label}_{start_index + sample_count}.npz")
                np.savez_compressed(output_path,
                                    frames=np.array(frames),
                                    landmarks=np.array(landmarks_data, dtype=object))
                sample_count += 1
                recording = False
                print(f"Saved sample {sample_count}")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    time.sleep(0.5)
    print(f"\nData collection complete! {sample_count} samples saved to {data_path}")

if __name__ == "__main__":
    collect_data()
