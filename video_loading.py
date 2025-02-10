import cv2
import numpy as np
import os

def extract_frames_from_video(video_path, frame_interval=30):
    """
    Extract frames from a video file.
    Parameters:
      video_path (str): Path to the video file.
      frame_interval (int): Extract one frame every `frame_interval` frames.
    Returns:
      List of frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

def save_frames_to_npz(frames, output_path):
    """
    Save a list of frames to an npz file using key 'frames'.
    """
    np.savez_compressed(output_path, frames=np.array(frames))

def load_frames_from_npz(npz_path):
    """
    Load frames from an npz file saved with key 'frames'.
    """
    data = np.load(npz_path)
    return data['frames']

def process_video_folder(video_folder, output_folder, frame_interval=30):
    """
    Process all video files in the specified video_folder.
    Extract frames from each video and save them as a compressed npz file in output_folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file in os.listdir(video_folder):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_folder, file)
            print(f"Processing video: {video_path}")
            frames = extract_frames_from_video(video_path, frame_interval=frame_interval)
            output_file = os.path.splitext(file)[0] + ".npz"
            output_path = os.path.join(output_folder, output_file)
            save_frames_to_npz(frames, output_path)
            print(f"Saved extracted frames to: {output_path}")

if __name__ == "__main__":
    # Default directories—adjust these paths as needed
    video_folder = os.path.join("data", "videos")
    output_folder = os.path.join("data", "video_frames")
    frame_interval = 30  # Change as needed
    process_video_folder(video_folder, output_folder, frame_interval)
