import ssl
import certifi
import os
import json
import cv2
import numpy as np
import urllib.parse
import subprocess
import concurrent.futures
import threading
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -------------------------------
# CONFIGURATION
# -------------------------------
TRAIN_JSON_PATH = "MS-ASL/MSASL_train.json"
TEST_JSON_PATH  = "MS-ASL/MSASL_test.json"
VIDEOS_FOLDER = "videos"          # Temporary folder for video downloads
CACHE_FOLDER = "processed_data"   # Folder to cache processed clips (to save WiFi/data)
CHECKPOINT_FOLDER = "checkpoints" # Folder for model checkpoints
IMG_SIZE = (112, 112)             # Target image size for each frame
NUM_FRAMES = 5                    # Number of frames to extract per video clip

# Limit new downloads to prevent too much data usage over WiFi.
# Only videos that are not yet cached will be downloaded.
NEW_DOWNLOADS = 0
MAX_NEW_DOWNLOADS = 50  # Adjust this limit as needed
new_downloads_lock = threading.Lock()

# Ensure required directories exist.
os.makedirs(VIDEOS_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

# Set up SSL certificates using Certifi.
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# -------------------------------
# YOUTUBE VIDEO HANDLING
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
        print(f"[INFO] Downloading video {url} using yt-dlp ...")
        command = [
            "yt-dlp",
            "--no-check-certificate",  # bypass SSL errors
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
    """Delete the video file after processing."""
    try:
        os.remove(video_path)
        print(f"[INFO] Deleted video: {video_path}")
    except Exception as e:
        print(f"[WARNING] Could not delete {video_path}: {e}")

# -------------------------------
# FRAME EXTRACTION
# -------------------------------
def get_multiple_frames(video_path, start_time, end_time, num_frames=NUM_FRAMES):
    """
    Extracts a list of frames uniformly spaced between start_time and end_time.
    Returns a list of frames (as NumPy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video {video_path}")
        return []
    frames = []
    # Ensure a nonzero interval.
    if end_time <= start_time:
        end_time = start_time + 1.0
    for i in range(num_frames):
        # Uniformly sample frames within the interval.
        time_sec = start_time + i * (end_time - start_time) / (num_frames - 1) if num_frames > 1 else start_time
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"[WARNING] Could not read frame at {time_sec:.2f} sec in {video_path}")
    cap.release()
    return frames

# -------------------------------
# DATA AUGMENTATION
# -------------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False
)

def augment_video_clip(clip, datagen):
    """
    Apply the same random transformation to every frame in the clip.
    clip: NumPy array of shape (NUM_FRAMES, height, width, channels)
    Returns the augmented clip.
    """
    transform_params = datagen.get_random_transform(clip[0].shape)
    augmented_clip = np.empty_like(clip)
    for i in range(clip.shape[0]):
        augmented_clip[i] = datagen.apply_transform(clip[i], transform_params)
    return augmented_clip

def video_generator(X, y, batch_size, datagen, augment=True):
    """
    Generator that yields batches of video clips and labels.
    If augment==True, applies data augmentation to each clip.
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_X = X[batch_idx]
            batch_y = y[batch_idx]
            if augment:
                augmented_batch = np.empty_like(batch_X)
                for j in range(batch_X.shape[0]):
                    augmented_batch[j] = augment_video_clip(batch_X[j], datagen)
                yield augmented_batch, batch_y
            else:
                yield batch_X, batch_y

# -------------------------------
# DATA LOADING & PREPROCESSING WITH CACHING
# -------------------------------
def process_record(record, videos_folder=VIDEOS_FOLDER, cache_folder=CACHE_FOLDER):
    """
    Process a single record:
      - If a cached processed clip exists, load and return it.
      - Otherwise, if within download limits, download the video,
        extract NUM_FRAMES, crop using the provided bounding box,
        convert to grayscale, resize, normalize, and then cache the result.
    Returns (clip, label) or None on failure.
    """
    url = record.get("url", None)
    if not url:
        print("[WARNING] No URL found in record; skipping.")
        return None

    video_id = get_video_id(url)
    os.makedirs(cache_folder, exist_ok=True)
    cache_file = os.path.join(cache_folder, f"{video_id}.npz")
    
    # Load from cache if available.
    if os.path.exists(cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True)
            clip = data["clip"]
            # If label is stored as an array, extract the single element.
            label = data["label"].item() if isinstance(data["label"], np.ndarray) else data["label"]
            print(f"[INFO] Loaded cached data for video {video_id}")
            return clip, label
        except Exception as e:
            print(f"[WARNING] Failed to load cache for {video_id}: {e}")

    # Check if we are within our new download limits.
    global NEW_DOWNLOADS
    with new_downloads_lock:
        if NEW_DOWNLOADS >= MAX_NEW_DOWNLOADS:
            print(f"[INFO] Maximum new downloads reached. Skipping video {video_id}.")
            return None
        NEW_DOWNLOADS += 1

    video_path = download_youtube_video(url, videos_folder)
    if video_path is None:
        print("[WARNING] Video could not be downloaded; skipping record.")
        return None

    start_time = record.get("start_time", 0.0)
    end_time = record.get("end_time", start_time + 1.0)
    frames = get_multiple_frames(video_path, start_time, end_time, num_frames=NUM_FRAMES)
    delete_video(video_path)  # Clean up downloaded file.

    if len(frames) != NUM_FRAMES:
        print("[WARNING] Not enough frames extracted; skipping record.")
        return None

    processed_frames = []
    for frame in frames:
        frame_height, frame_width = frame.shape[:2]
        box = record.get("box", None)
        if not box or len(box) != 4:
            print("[WARNING] Invalid bounding box; skipping record.")
            return None

        # Convert normalized box coordinates to pixels.
        x_min = int(box[0] * frame_width)
        y_min = int(box[1] * frame_height)
        x_max = int(box[2] * frame_width)
        y_max = int(box[3] * frame_height)

        # Validate crop dimensions.
        if x_max <= x_min or y_max <= y_min or (x_max - x_min) < 10 or (y_max - y_min) < 10:
            print("[WARNING] Invalid crop dimensions; skipping record.")
            return None

        cropped = frame[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            print("[WARNING] Empty crop; skipping record.")
            return None

        # Convert to grayscale, resize, and normalize.
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMG_SIZE)
        normalized = resized.astype("float32") / 255.0
        normalized = np.expand_dims(normalized, axis=-1)  # Shape: (112, 112, 1)
        processed_frames.append(normalized)

    clip = np.stack(processed_frames, axis=0)  # Shape: (NUM_FRAMES, 112, 112, 1)
    label = record.get("clean_text", "unknown")

    # Save processed clip to cache.
    try:
        np.savez_compressed(cache_file, clip=clip, label=label)
        print(f"[INFO] Cached processed data for video {video_id}")
    except Exception as e:
        print(f"[WARNING] Could not save cache for {video_id}: {e}")

    return clip, label

def load_data(json_path, videos_folder=VIDEOS_FOLDER, cache_folder=CACHE_FOLDER):
    """
    Load JSON records, process each record in parallel (using caching), and return:
      - clips: NumPy array of shape (N, NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 1)
      - labels: list of text labels.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    clips = []
    labels = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_record, record, videos_folder, cache_folder) for record in data]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                clip, label = result
                clips.append(clip)
                labels.append(label)
    if clips:
        clips = np.array(clips)
    return clips, labels

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

    # Map labels to indices.
    unique_labels = sorted(list(set(y_train_text)))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    print("Unique labels found:", unique_labels)

    y_train = np.array([label_to_index[label] for label in y_train_text])
    y_test  = np.array([label_to_index[label] for label in y_test_text])

    num_classes = len(unique_labels)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat  = to_categorical(y_test, num_classes)

    # Build a simple Conv3D model.
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation="relu", input_shape=(NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 1)),
        MaxPooling3D(pool_size=(1, 2, 2)),
        Conv3D(64, (3, 3, 3), activation="relu"),
        MaxPooling3D(pool_size=(1, 2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Training parameters.
    batch_size = 8
    epochs = 10
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_test) // batch_size

    # Create data generators.
    train_gen = video_generator(X_train, y_train_cat, batch_size, datagen, augment=True)
    val_gen = video_generator(X_test, y_test_cat, batch_size, datagen, augment=False)

    # Set up callbacks.
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_FOLDER, "model_epoch_{epoch:02d}.h5"),
        save_weights_only=False,
        save_freq="epoch",
        verbose=1
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=1,
        restore_best_weights=True
    )

    # Train the model.
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    model.save("collected_data_model.h5")
    print("Model saved as collected_data_model.h5")
