#!/usr/bin/env python
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time
import platform
import argparse

import mediapipe as mp

# --- Helper Functions ---

def parse_args():
    parser = argparse.ArgumentParser(description="Collect ASL gesture keypoint data using MediaPipe Holistic.")
    parser.add_argument('--label', type=str, required=True,
                        help="Label for the sign (e.g., 'hello', 'bye').")
    parser.add_argument('--samples', type=int, default=30,
                        help="Number of samples (clips) to record.")
    parser.add_argument('--duration', type=float, default=2,
                        help="Duration (in seconds) of each sample/clip.")
    return parser.parse_args()

def flatten_landmarks(landmarks, prefix):
    """
    Convert a flat list/array of landmark coordinates into a dictionary
    with keys like 'prefix_0', 'prefix_1', ..., so that each coordinate becomes a separate column.
    """
    flat_dict = {}
    for i, val in enumerate(landmarks):
        flat_dict[f'{prefix}_{i}'] = val
    return flat_dict

def extract_keypoints(results):
    """
    Extract and flatten landmarks from the MediaPipe holistic results.
    Returns flattened arrays for pose, face, left hand, and right hand.
    If a particular set of landmarks is not detected, return an array of zeros.
    """
    # Pose: 33 landmarks, each with (x, y, z, visibility)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)
    
    # Face: 468 landmarks, each with (x, y, z)
    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3)
    
    # Left hand: 21 landmarks, each with (x, y, z)
    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(21 * 3)
    
    # Right hand: 21 landmarks, each with (x, y, z)
    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(21 * 3)
    
    return pose, face, left_hand, right_hand

# --- Main Data Collection Function ---

def collect_data(label, samples_needed, clip_duration):
    # Create output directory if needed
    data_path = os.path.join('data', label)
    os.makedirs(data_path, exist_ok=True)
    
    # Determine starting index based on existing files
    existing_files = [f for f in os.listdir(data_path) if f.startswith(label) and f.endswith('.csv')]
    if existing_files:
        existing_indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.split('_')[-1].split('.')[0].isdigit()]
        start_index = max(existing_indices) + 1 if existing_indices else 0
    else:
        start_index = 0

    # Set up camera with platform–specific backend
    system_platform = platform.system().lower()
    if system_platform == 'darwin':  # macOS
        backend = cv2.CAP_AVFOUNDATION
    elif system_platform == 'windows':
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_V4L2

    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        raise IOError("Cannot open webcam. Check permissions or if another application is using the camera.")

    # Camera warmup
    print("\nInitializing camera...")
    for _ in range(5):
        cap.read()
    time.sleep(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    if system_platform != 'darwin':
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # Initialize MediaPipe Holistic and Drawing modules
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    sample_count = 0
    recording = False
    frame_data = []  # To store data for the current sample clip
    frame_idx = 0

    print(f"\n=== Collecting {samples_needed} samples for '{label}' ===")
    print("Press 's' to start recording a sample, 'q' to quit early.")

    # Use holistic in a context manager
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while sample_count < samples_needed:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture error - trying to recover...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(0, backend)
                continue

            # Flip frame horizontally for a mirror effect.
            frame = cv2.flip(frame, 1)
            orig_frame = frame.copy()

            # Convert the BGR frame to RGB for MediaPipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # Improve performance
            results = holistic.process(image)
            image.flags.writeable = True

            # Draw landmarks for visual feedback
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Overlay instructions and status
            if not recording:
                cv2.putText(frame, f"Press 's' to record sample {sample_count + 1}/{samples_needed}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                elapsed = (datetime.now() - start_time).total_seconds()
                cv2.putText(frame, f"Recording sample {sample_count + 1} | Frame: {frame_idx} | Elapsed: {elapsed:.1f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Press 's' to restart sample, 'q' to quit", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow('Data Collector', frame)
            key = cv2.waitKey(1)

            if key == ord('q'):
                print("Quitting early.")
                break

            # Start recording if 's' is pressed and we are not already recording.
            if key == ord('s') and not recording:
                recording = True
                frame_data = []  # Reset current sample data
                start_time = datetime.now()
                frame_idx = 0
                print(f"Recording sample {sample_count + 1}...")
                continue  # Wait for the next frame

            # If recording, extract and store keypoints from this frame.
            if recording:
                # Extract keypoints from the current frame.
                pose_kp, face_kp, left_hand_kp, right_hand_kp = extract_keypoints(results)

                # Create a row dictionary for this frame.
                row = {'frame': frame_idx}
                row.update(flatten_landmarks(pose_kp, 'pose'))
                row.update(flatten_landmarks(face_kp, 'face'))
                row.update(flatten_landmarks(left_hand_kp, 'left_hand'))
                row.update(flatten_landmarks(right_hand_kp, 'right_hand'))
                frame_data.append(row)
                frame_idx += 1

                # Check if the clip duration has elapsed
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= clip_duration:
                    # Stop recording the current sample
                    recording = False
                    sample_filename = os.path.join(data_path, f"{label}_{start_index + sample_count}.csv")
                    df = pd.DataFrame(frame_data)
                    df.insert(0, 'sample', sample_count)  # add sample index column
                    df.to_csv(sample_filename, index=False)
                    print(f"Saved sample {sample_count + 1} with {frame_idx} frames to {sample_filename}")
                    sample_count += 1
                    time.sleep(0.5)  # brief pause before next sample

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nData collection complete! {sample_count} samples saved to {data_path}")

# --- Main Execution ---

if __name__ == "__main__":
    args = parse_args()
    # Use lowercase label for consistency
    collect_data(args.label.lower(), args.samples, args.duration)
