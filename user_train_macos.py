import cv2
import os
import numpy as np
from datetime import datetime
import time  # Added for warmup delay

def collect_data():
    # User input
    label = input("Enter the sign label to record (e.g., 'hello', 'bye'): ").lower()
    samples_needed = 30
    clip_duration = 2  # seconds

    # Create directories
    data_path = os.path.join('data', label)
    os.makedirs(data_path, exist_ok=True)

    # Determine the starting index for new files
    existing_files = [f for f in os.listdir(data_path) if f.startswith(label) and f.endswith('.npz')]
    if existing_files:
        existing_indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
        start_index = max(existing_indices) + 1
    else:
        start_index = 0

    # Camera setup with macOS fixes
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macOS-specific backend
    if not cap.isOpened():
        raise IOError("""
        Cannot open webcam. Ensure:
        1. Camera permissions are granted (System Preferences > Security & Privacy)
        2. No other apps are using the camera
        3. You're on a physical Mac (some virtual machines have camera issues)
        """)

    # Camera warmup sequence
    print("\nInitializing camera...")
    for _ in range(5):  # Dummy reads to stabilize
        cap.read()
    time.sleep(1)

    # Set camera properties with validation
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))  # macOS-compatible codec

    print(f"\n=== Collecting {samples_needed} samples for '{label}' ===")
    print("Press 's' to start recording, 'q' to quit early")

    sample_count = 0
    recording = False
    frames = []
    window_open = True

    while sample_count < samples_needed and window_open:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture error - trying to recover...")
            # Attempt camera reinitialization
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            continue

        # Mirror preview
        frame = cv2.flip(frame, 1)

        # Display instructions
        text = f"Sample {sample_count + 1}/{samples_needed} | Press 's' to start"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        try:
            cv2.imshow('Data Collector', frame)
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
            start_time = datetime.now()
            print(f"Recording sample {sample_count + 1}...")

        if recording:
            # Store low-res grayscale frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (112, 112))  # Resize for compact storage
            frames.append(small)

            # Check recording duration
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= clip_duration:
                # Save as compressed numpy array
                output_path = os.path.join(data_path, f"{label}_{start_index + sample_count}.npz")
                np.savez_compressed(output_path, np.array(frames))

                sample_count += 1
                recording = False
                print(f"Saved sample {sample_count}")

    # Cleanup resources properly
    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.5)  # Allow windows to close

    print(f"\nData collection complete! {sample_count} samples saved to {data_path}")


if __name__ == "__main__":
    collect_data()
