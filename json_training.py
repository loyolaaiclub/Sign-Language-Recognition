import ssl
import certifi
import os
import json
import cv2
import numpy as np
import urllib.parse
import subprocess
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import concurrent.futures

# Use Certifi SSL Certificates to fix SSL errors
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# -------------------------------
# CONFIGURATION
# -------------------------------
TRAIN_JSON_PATH = "MS-ASL/MSASL_train.json"
TEST_JSON_PATH  = "MS-ASL/MSASL_test.json"
VIDEOS_FOLDER = "videos"  # Temporary folder for video downloads
IMG_SIZE = (112, 112)

# -------------------------------
# YOUTUBE VIDEO HANDLING
# -------------------------------
def sanitize_youtube_url(url):
    """Ensure the URL starts with http:// or https://."""
    if not url.startswith("http"):
        return "https://" + url
    return url

def download_and_process(record):
    url = record.get("url", None)
    if not url:
        return None
    
    video_path = download_youtube_video(url)
    if not video_path:
        return None

    frame = get_frame_from_video(video_path, record.get("start_time", 0.0))
    delete_video(video_path)  # Remove after processing
    return (frame, record.get("clean_text", "unknown"))

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(download_and_process, data))


def get_video_id(url):
    """Extract the YouTube video ID from the URL."""
    url = sanitize_youtube_url(url)
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    video_id = query.get("v", [None])[0]
    if video_id is None:
        video_id = parsed.path.split("/")[-1]
    return video_id

def download_youtube_video(url, videos_folder="videos"):
    """Downloads a YouTube video using yt-dlp and returns the local file path."""
    video_id = get_video_id(url)
    filename = f"{video_id}.mp4"
    video_path = os.path.join(videos_folder, filename)

    os.makedirs(videos_folder, exist_ok=True)

    try:
        print(f"[INFO] Downloading video {url} using yt-dlp ...")
        command = [
            "yt-dlp",
            "--no-check-certificate",  # Bypass SSL errors
            "-f", "mp4",
            "-o", video_path,
            url
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[ERROR] yt-dlp failed: {result.stderr}")
            return None

        print(f"[INFO] Video saved as {video_path}")
        return video_path

    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        return None

def delete_video(video_path):
    """Deletes the video file after processing."""
    try:
        os.remove(video_path)
        print(f"[INFO] Deleted video: {video_path}")
    except Exception as e:
        print(f"[WARNING] Could not delete {video_path}: {e}")

def get_frame_from_video(video_path, time_sec):
    """Extracts a frame from the video at the specified time (in seconds)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[WARNING] Could not read frame at {time_sec} sec in {video_path}")
        return None

    return frame

def get_multiple_frames(video_path, start_time, end_time, num_frames=5):
    """Extract multiple frames between start_time and end_time."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    frames = []
    for i in range(num_frames):
        time_sec = start_time + i * (end_time - start_time) / (num_frames - 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames

# -------------------------------
# DATA LOADING & PREPROCESSING
# -------------------------------
def load_data(json_path):
    """
    Downloads YouTube videos, extracts a frame, preprocesses it, and deletes the video.
    Returns:
      - images: NumPy array of shape (N, 112, 112, 1)
      - labels: list of text labels (from "clean_text")
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = []
    labels = []

    for idx, record in enumerate(data):
        url = record.get("url", None)
        if url is None:
            print(f"[WARNING] No URL for record {idx}, skipping.")
            continue

        video_path = download_youtube_video(url)
        if video_path is None:
            print(f"[WARNING] Skipping record {idx} because video could not be downloaded.")
            continue

        start_time = record.get("start_time", 0.0)
        end_time = record.get("end_time", start_time + 1.0)
        mid_time = (start_time + end_time) / 2.0

        frame = get_frame_from_video(video_path, mid_time)
        if frame is None:
            print(f"[WARNING] No frame extracted for record {idx}, skipping.")
            delete_video(video_path)  # Delete video if processing failed
            continue

        # Get actual frame dimensions
        frame_height, frame_width = frame.shape[:2]
        box = record.get("box", None)
        if box is None or len(box) != 4:
            print(f"[WARNING] Invalid bounding box for record {idx}, skipping.")
            delete_video(video_path)
            continue

        # Convert normalized coordinates to pixel coordinates
        x_min = int(box[0] * frame_width)
        y_min = int(box[1] * frame_height)
        x_max = int(box[2] * frame_width)
        y_max = int(box[3] * frame_height)

        x_min = max(0, min(x_min, frame_width - 1))
        x_max = max(1, min(x_max, frame_width))
        y_min = max(0, min(y_min, frame_height - 1))
        y_max = max(1, min(y_max, frame_height))

        if x_max <= x_min or y_max <= y_min:
            print(f"[WARNING] Invalid crop dimensions for record {idx}, skipping.")
            delete_video(video_path)
            continue

        cropped = frame[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            print(f"[WARNING] Empty crop for record {idx}, skipping.")
            delete_video(video_path)
            continue

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMG_SIZE)
        normalized = resized.astype('float32') / 255.0

        images.append(normalized)
        labels.append(record.get("clean_text", "unknown"))

        # Delete video file after processing
        delete_video(video_path)

    images = np.array(images).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
    return images, labels

# -------------------------------
# MAIN TRAINING SCRIPT
# -------------------------------
if __name__ == "__main__":
    print("Loading training data...")
    X_train, y_train_text = load_data(TRAIN_JSON_PATH)
    print(f"Loaded {len(X_train)} training samples.")

    print("Loading test data...")
    X_test, y_test_text = load_data(TEST_JSON_PATH)
    print(f"Loaded {len(X_test)} test samples.")

    if len(X_train) == 0 or len(X_test) == 0:
        print("[ERROR] No data loaded. Please check your JSON files and YouTube URL access.")
        exit(1)

    unique_labels = sorted(list(set(y_train_text)))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    print("Unique labels found:", unique_labels)

    y_train = np.array([label_to_index[label] for label in y_train_text])
    y_test  = np.array([label_to_index[label] for label in y_test_text])

    num_classes = len(unique_labels)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat  = to_categorical(y_test, num_classes)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train_cat, epochs=10, batch_size=32, validation_data=(X_test, y_test_cat))
    model.save('collected_data_model.h5')
    print("Model saved as collected_data_model.h5")
