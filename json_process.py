#!/usr/bin/env python
"""
prepare_data.py

This script reads a dataset JSON file (one record per line) and for each record:
  - Downloads the YouTube video (if not already downloaded)
  - Extracts NUM_FRAMES uniformly between the provided start and end times
  - Crops each frame using the given bounding box, converts to grayscale,
    resizes to IMG_SIZE, and normalizes the pixel values.
  - Saves the processed clip as an NPZ file in a folder corresponding to its gesture.
  
Usage:
    python prepare_data.py --json MSASL_train.json
"""
import os
import json
import ssl
import certifi
import cv2
import numpy as np
import urllib.parse
import subprocess
import concurrent.futures
import argparse

# -------------------------------
# CONFIGURATION
# -------------------------------
IMG_SIZE = (112, 112)           # Target image size for each frame
NUM_FRAMES = 5                  # Number of frames per clip
VIDEOS_FOLDER = "videos"        # Folder to temporarily store downloaded videos
DATA_FOLDER = "data"            # Main folder for processed NPZ files

os.makedirs(VIDEOS_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Set up SSL certificates using certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def sanitize_youtube_url(url):
    """Ensure the URL starts with http:// or https://."""
    if not url.startswith("http"):
        return "https://" + url
    return url

def get_video_id(url):
    """Extract the YouTube video ID from the URL."""
    url = sanitize_youtube_url(url)
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    video_id = query.get("v", [None])[0]
    if video_id is None:
        video_id = parsed.path.split("/")[-1]
    return video_id

def download_youtube_video(url, videos_folder=VIDEOS_FOLDER):
    """Download a YouTube video using yt-dlp and return the local file path."""
    video_id = get_video_id(url)
    filename = f"{video_id}.mp4"
    video_path = os.path.join(videos_folder, filename)
    if os.path.exists(video_path):
        print(f"[INFO] Video already exists: {video_path}")
        return video_path
    try:
        print(f"[INFO] Downloading video {url} ...")
        command = [
            "yt-dlp",
            "--no-check-certificate",
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
    """Delete the temporary video file."""
    try:
        os.remove(video_path)
        print(f"[INFO] Deleted video: {video_path}")
    except Exception as e:
        print(f"[WARNING] Could not delete {video_path}: {e}")

def get_multiple_frames(video_path, start_time, end_time, num_frames=NUM_FRAMES):
    """
    Extract a list of frames uniformly spaced between start_time and end_time.
    Returns a list of frames (as NumPy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video {video_path}")
        return []
    frames = []
    if end_time <= start_time:
        end_time = start_time + 1.0
    for i in range(num_frames):
        time_sec = start_time + i * (end_time - start_time) / (num_frames - 1) if num_frames > 1 else start_time
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"[WARNING] Could not read frame at {time_sec:.2f} sec in {video_path}")
    cap.release()
    return frames

def process_record(record):
    """
    Process a single record:
      - Download the video (if needed)
      - Extract and process frames
      - Save the processed clip as an NPZ file in DATA_FOLDER/gesture/
    """
    url = record.get("url", None)
    if not url:
        print("[WARNING] No URL found; skipping record.")
        return

    video_id = get_video_id(url)
    video_path = download_youtube_video(url)
    
    if video_path is None:
        print("[WARNING] Download failed; skipping record.")
        return  # Stop processing this record

    # Move folder creation **AFTER** successful download
    gesture = record.get("clean_text", "unknown").strip().lower()
    gesture_folder = "".join(ch for ch in gesture if ch.isalnum())  # Remove spaces & special chars
    gesture_path = os.path.join(DATA_FOLDER, gesture_folder)
    os.makedirs(gesture_path, exist_ok=True)

    # Create a unique filename for saving
    record_id = f"{video_id}_{str(record.get('start_time')).replace('.','_')}"
    npz_file = os.path.join(gesture_path, record_id + ".npz")
    if os.path.exists(npz_file):
        print(f"[INFO] Processed data already exists: {npz_file}")
        return

    start_time = record.get("start_time", 0.0)
    end_time = record.get("end_time", start_time + 1.0)
    frames = get_multiple_frames(video_path, start_time, end_time, num_frames=NUM_FRAMES)
    delete_video(video_path)
    if len(frames) != NUM_FRAMES:
        print("[WARNING] Not enough frames; skipping record.")
        return

    processed_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        box = record.get("box", None)
        if not box or len(box) != 4:
            print("[WARNING] Invalid bounding box; skipping record.")
            return
        x_min = int(box[0] * w)
        y_min = int(box[1] * h)
        x_max = int(box[2] * w)
        y_max = int(box[3] * h)
        if x_max <= x_min or y_max <= y_min or (x_max - x_min) < 10 or (y_max - y_min) < 10:
            print("[WARNING] Invalid crop dimensions; skipping record.")
            return
        cropped = frame[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            print("[WARNING] Empty crop; skipping record.")
            return
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMG_SIZE)
        normalized = resized.astype("float32") / 255.0
        normalized = np.expand_dims(normalized, axis=-1)  # (112, 112, 1)
        processed_frames.append(normalized)

    clip = np.stack(processed_frames, axis=0)  # (NUM_FRAMES, 112, 112, 1)
    try:
        np.savez_compressed(npz_file, clip=clip, label=gesture)
        print(f"[INFO] Saved processed data: {npz_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save {npz_file}: {e}")

import json

def main(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)  # Load the entire file as a JSON array
    print(f"[INFO] Found {len(data)} records in {json_file}")

    # Now process each record normally
    for record in data:
        process_record(record)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(process_record, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ASL data into NPZ files organized by gesture.")
    parser.add_argument("--json", type=str, required=True, help="Path to input JSON file (e.g. MSASL_train.json)")
    args = parser.parse_args()
    main(args.json)
