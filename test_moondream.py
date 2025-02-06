import cv2
import moondream as md
import numpy as np
from PIL import Image
import os
import requests
from concurrent.futures import ThreadPoolExecutor

def download_video(video_url, video_path):
    """Download the video file if it does not exist."""
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    if os.path.exists(video_path):
        print(f"Video already exists: {video_path}")
        return
    
    print("Downloading test video...")
    response = requests.get(video_url, stream=True)
    if response.status_code == 200:
        with open(video_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Video saved: {video_path}")
    else:
        raise Exception("Failed to download video. Check the URL.")

def extract_frames(video_path, frame_interval=1):
    """
    Extract frames at a specified interval directly from memory without saving them to disk.
    Returns a list of (timestamp, image) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_rate * frame_interval)

    frames = []
    frame_number = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_number % interval == 0:
            timestamp = frame_number / frame_rate
            frames.append((timestamp, frame))  # Store frame in memory

        frame_number += 1

    cap.release()
    print(f"Extracted {len(frames)} frames from the video.")
    return frames

def process_frame(model, frame_info):
    """
    Process a single frame and generate captions.
    """
    timestamp, frame = frame_info
    try:
        # Convert frame from OpenCV to PIL Image for Moondream processing
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Query the image
        caption = model.query(image, question="what is in the image?")['answer']

        print(f"[{timestamp:.2f}s] {caption}")
        return timestamp, caption
    except Exception as e:
        print(f"Error processing frame at {timestamp:.2f}s: {e}")
        return timestamp, "Error processing frame"

import os

def process_frames_multithreaded(model, frames):
    """
    Process all extracted frames using multithreading to speed up inference.
    Dynamically determine the number of threads based on system capacity.
    """
    # Determine the number of available CPU cores
    max_workers = os.cpu_count() or 4  # Default to 4 if cpu_count() returns None
    print("Max workers used in the process",max_workers)
    captions = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda frame_info: process_frame(model, frame_info), frames)
    
    captions.extend(results)
    return captions

def save_captions(captions, output_file="captions.srt"):
    """
    Save captions in SRT subtitle format.
    """
    with open(output_file, "w") as f:
        for i, (timestamp, caption) in enumerate(captions, start=1):
            f.write(f"{i}\n")
            f.write(f"{timestamp:.2f} --> {timestamp + 2:.2f}\n")  # Approximate duration
            f.write(f"{caption}\n\n")

    print(f"Captions saved to {output_file}")

def main():
    # Setup paths and URLs
    video_dir = "test_videos"
    video_filename = "sample.mp4"
    video_path = os.path.join(video_dir, video_filename)
    video_url = "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"

    # Download video if needed
    download_video(video_url, video_path)

    # Initialize Moondream model
    model = md.vl(model="moondream-0_5b-int8.mf")

    # Extract frames
    print("Extracting frames...")
    frames = extract_frames(video_path, frame_interval=1)

    # Process frames with multithreading
    print("Processing frames...")
    captions = process_frames_multithreaded(model, frames)

    # Save results
    save_captions(captions)

    print("Processing complete.")

if __name__ == "__main__":
    main()
