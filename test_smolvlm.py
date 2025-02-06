import cv2
import torch
import os
import requests
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup (CPU/GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize SmolVLM model & processor
def initialize_model():
    """Load the SmolVLM processor and model."""
    try:
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager"
        ).to(DEVICE)
        return processor, model
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

# Video downloading method (Same as previous approach)
def download_video(video_url: str, video_path: str) -> None:
    """Download video if it doesn't exist."""
    if not os.path.exists(video_path):
        logger.info("Downloading test video...")
        try:
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            with open(video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            logger.info(f"Video saved: {video_path}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download video: {str(e)}")
            raise

# Process individual video frames
def process_frame(frame, processor, model):
    """Process a video frame and generate a caption."""
    try:
        # Convert OpenCV frame (BGR) to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Prepare input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe what is happening in this image."}
                ]
            }
        ]

        # Process input
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)

        # Generate caption
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

        caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return caption
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return "Error generating caption"

# Process video & generate captions
def process_video(video_path, processor, model):
    """Extract frames from a video and generate captions."""
    captions = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Error opening video file")

    try:
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(frame_rate * 2)  # Capture every 2 seconds
        frame_number = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_number % frame_interval == 0:
                # Generate caption
                caption = process_frame(frame, processor, model)
                timestamp = frame_number / frame_rate
                captions.append((timestamp, caption))

                logger.info(f"[{timestamp:.2f}s] {caption}")

                # Overlay caption onto the frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                text_color = (0, 255, 0)  # Green text
                bg_color = (0, 0, 0)  # Black background

                text_size, _ = cv2.getTextSize(caption, font, font_scale, font_thickness)
                text_x = 10
                text_y = frame.shape[0] - 20  # Bottom of frame

                # Draw text background rectangle
                cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5), bg_color, -1)

                # Put caption text
                cv2.putText(frame, caption, (text_x, text_y), font, font_scale, text_color, font_thickness)

                # Resize frame for display
                scale_percent = 50  # Reduce size to 50%
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                # Show only the current frame
                cv2.imshow("Analyzing Frame", resized_frame)

            frame_number += 1

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captions

# Save captions as subtitles
def save_captions(captions, output_path="captions.srt"):
    """Save captions in SRT format."""
    try:
        with open(output_path, "w") as f:
            for i, (timestamp, caption) in enumerate(captions, start=1):
                f.write(f"{i}\n")
                f.write(f"{timestamp:.2f} --> {timestamp+2:.2f}\n")
                f.write(f"{caption}\n\n")
        logger.info(f"Captions saved to {output_path}")
    except IOError as e:
        logger.error(f"Error saving captions: {str(e)}")
        raise

# Main function to run the pipeline
def main():
    """Main function to run the SmolVLM video captioning pipeline."""
    model_path = "HuggingFaceTB/SmolVLM-Instruct"
    video_dir = "test_videos"
    video_filename = "sample.mp4"
    video_path = os.path.join(video_dir, video_filename)
    video_url = "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"

    try:
        # Ensure directory exists
        os.makedirs(video_dir, exist_ok=True)

        # Initialize SmolVLM
        processor, model = initialize_model()

        # Download video if needed
        download_video(video_url, video_path)

        # Process video
        captions = process_video(video_path, processor, model)

        # Save captions
        save_captions(captions)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
