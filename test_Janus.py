import os
import requests
import cv2
import torch
from transformers import AutoModelForCausalLM
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
from PIL import Image
from io import BytesIO
import os

# Disable FlashAttention2 if it's causing issues
os.environ["TRANSFORMERS_NO_FLASHATTN"] = "1"
# Step 1: Download the Video from the URL
def download_video(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Video downloaded successfully: {output_path}")
    else:
        print(f"Failed to download video. Status code: {response.status_code}")

# Step 2: Extract Frames from the Video
def extract_frames(video_path, frames_dir, interval=30):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_path = os.path.join(frames_dir, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_count += 1
        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_count} frames from the video.")

# Step 3: Load the Janus-1.3B Model
def load_janus_model(model_path="deepseek-ai/Janus-1.3B"):
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    return vl_chat_processor, vl_gpt

# Step 4: Analyze Frames with Janus-1.3B
def analyze_frame(frame_path, vl_chat_processor, vl_gpt):
    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>\nDescribe the content of this frame.",
            "images": [frame_path],
        },
        {"role": "Assistant", "content": ""},
    ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    output = vl_gpt.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.95,
        top_k=60,
    )

    response = vl_chat_processor.batch_decode(output, skip_special_tokens=True)[0]
    return response

# Step 5: Display or Log the Results
def main():
    video_url = "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"  # Replace with your video URL
    video_path = "downloaded_video.mp4"
    frames_dir = "extracted_frames"

    download_video(video_url, video_path)
    extract_frames(video_path, frames_dir, interval=30)

    vl_chat_processor, vl_gpt = load_janus_model()

    for frame_file in sorted(os.listdir(frames_dir)):
        frame_path = os.path.join(frames_dir, frame_file)
        description = analyze_frame(frame_path, vl_chat_processor, vl_gpt)
        print(f"Frame {frame_file}: {description}")

if __name__ == "__main__":
    main()
