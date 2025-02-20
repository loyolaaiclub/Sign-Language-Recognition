#!/usr/bin/env python
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time
import platform
import mediapipe as mp

def collect_data():
    label = input("Enter the sign label to record (e.g., 'hello', 'bye'): ").lower()
    samples_needed = 2
    clip_duration = 2  # seconds

    data_path = os.path.join('data', label)
    os.makedirs(data_path, exist_ok=True)

    existing_files = [f for f in os.listdir(data_path) if f.startswith(label) and f.endswith('.npz')]
    start_index = max([int(f.split('_')[-1].split('.')[0]) for f in existing_files], default=-1) + 1

    system_platform = platform.system().lower()
    backend = cv2.CAP_AVFOUNDATION if system_platform == 'darwin' else cv2.CAP_DSHOW if system_platform == 'windows' else cv2.CAP_V4L2
    
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

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_face = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    sample_count = 0
    recording = False
    frames, hand_data, pose_data, face_data = [], [], [], []
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)
        face_results = face.process(frame_rgb)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(display_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(display_frame, face_landmarks, mp_face.FACEMESH_TESSELATION)
        
        cv2.rectangle(display_frame, (5, 5), (635, 60), (50, 50, 50), -1)
        cv2.putText(display_frame, f"Collecting Samples for '{label}'", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 's' to start, 'q' to quit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('Data Collector', display_frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s') and not recording:
            recording = True
            frames, hand_data, pose_data, face_data = [], [], [], []
            start_time = datetime.now()
            print(f"Recording sample {sample_count + 1}...")

        if recording:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (112, 112))
            frames.append(small)

            hand_landmarks_list = [[lm.x, lm.y, lm.z] for lm in hand_results.multi_hand_landmarks[0].landmark] if hand_results.multi_hand_landmarks else []
            pose_landmarks_list = [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark] if pose_results.pose_landmarks else []
            face_landmarks_list = [[lm.x, lm.y, lm.z] for lm in face_results.multi_face_landmarks[0].landmark] if face_results.multi_face_landmarks else []
            
            hand_data.append(hand_landmarks_list)
            pose_data.append(pose_landmarks_list)
            face_data.append(face_landmarks_list)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= clip_duration:
                output_path = os.path.join(data_path, f"{label}_{start_index + sample_count}.npz")
                np.savez_compressed(output_path,
                                    frames=np.array(frames),
                                    hand_landmarks=np.array(hand_data, dtype=object),
                                    pose_landmarks=np.array(pose_data, dtype=object),
                                    face_landmarks=np.array(face_data, dtype=object))
                sample_count += 1
                recording = False
                print(f"Saved sample {sample_count}")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()
    face.close()
    print(f"\nData collection complete! {sample_count} samples saved to {data_path}")

if __name__ == "__main__":
    args = parse_args()
    # Use lowercase label for consistency
    collect_data(args.label.lower(), args.samples, args.duration)
