import streamlit as st
import cv2
import moondream as md
import numpy as np
from PIL import Image
import os
import tempfile
import json
import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import shutil
from pathlib import Path
logging.basicConfig(level=logging.INFO)
@dataclass
class ProcessedFrame:
    timestamp: float
    frame_number: int
    detections: dict
    image_path: Optional[str] = None

class VideoProcessor:
    def __init__(self, model, temp_dir: str = "temp_processed_frames"):
        self.model = model
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
    def __del__(self):
        self.cleanup()
        
    def cleanup(self):
        """Remove temporary directory and its contents"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @staticmethod
    def get_contrast_color(img, x, y, box_size=20):
        try:
            y1 = max(0, y - box_size)
            y2 = min(img.shape[0], y + box_size)
            x1 = max(0, x - box_size)
            x2 = min(img.shape[1], x + box_size)
            region = img[y1:y2, x1:x2]
            
            if region.size > 0:
                avg_color = np.mean(region)
                return (0, 255, 255) if avg_color < 128 else (255, 0, 0)
        except:
            pass
        return (0, 255, 0)

    def draw_detection_boxes(self, image: Image.Image, detections: Dict, object_name: str) -> Image.Image:
        img = np.array(image)
        height, width = img.shape[:2]
        
        if isinstance(detections, dict) and 'objects' in detections:
            for i, obj in enumerate(detections['objects']):
                try:
                    x_min = max(0, int(float(obj['x_min']) * width))
                    y_min = max(0, int(float(obj['y_min']) * height))
                    x_max = min(width, int(float(obj['x_max']) * width))
                    y_max = min(height, int(float(obj['y_max']) * height))
                    
                    if x_min < x_max and y_min < y_max:
                        color = self.get_contrast_color(img, x_min, y_min)
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                        
                        label = f"{object_name} {i+1}"
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )
                        
                        text_bg_x1 = x_min
                        text_bg_y1 = max(0, y_min - text_height - 10)
                        text_bg_x2 = x_min + text_width + 10
                        text_bg_y2 = y_min
                        
                        cv2.rectangle(img, 
                                    (text_bg_x1, text_bg_y1),
                                    (text_bg_x2, text_bg_y2),
                                    color, -1)
                        
                        cv2.putText(img, label,
                                  (x_min + 5, max(y_min - 8, text_height)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        
                except Exception as e:
                    logging.warning(f"Error drawing box {i+1}: {str(e)}")
        
        return Image.fromarray(img)

    def process_single_frame(self, frame_data: Tuple[int, np.ndarray, float], object_to_detect: str) -> ProcessedFrame:
        frame_number, frame, timestamp = frame_data
        
        # Convert BGR to RGB
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Perform detection
        detections = self.model.detect(image, object_to_detect)
        
        # Draw boxes
        annotated_frame = self.draw_detection_boxes(image, detections, object_to_detect)
        
        # Save frame
        frame_path = self.temp_dir / f"frame_{frame_number:06d}.jpg"
        annotated_frame.save(frame_path)
        
        return ProcessedFrame(
            timestamp=timestamp,
            frame_number=frame_number,
            detections=detections,
            image_path=str(frame_path)
        )

    def process_video(self, video_path: str, frame_interval: int, object_to_detect: str, 
                     progress_bar, status_text) -> List[ProcessedFrame]:
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = int(frame_rate * frame_interval)
        
        frames_to_process = []
        frame_number = 0
        
        # First pass: collect frames
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            if frame_number % interval == 0:
                timestamp = frame_number / frame_rate
                frames_to_process.append((frame_number, frame, timestamp))
            
            frame_number += 1
            
        cap.release()
        
        processed_frames = []
        
        # Process frames using thread pool
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_frame = {
                executor.submit(self.process_single_frame, frame_data, object_to_detect): frame_data
                for frame_data in frames_to_process
            }
            
            for i, future in enumerate(as_completed(future_to_frame)):
                frame_data = future_to_frame[future]
                try:
                    processed_frame = future.result()
                    processed_frames.append(processed_frame)
                    
                    # Update progress
                    progress = int((i + 1) / len(frames_to_process) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {i + 1}/{len(frames_to_process)}")
                    
                except Exception as e:
                    logging.error(f"Error processing frame {frame_data[0]}: {str(e)}")
        
        # Sort frames by frame number
        processed_frames.sort(key=lambda x: x.frame_number)
        return processed_frames
    def process_video_for_description(self, video_path: str, frame_interval: int,
                                    progress_bar, status_text) -> List[dict]:
        """Process video frames for general description."""
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = int(frame_rate * frame_interval)
        
        frames_to_process = []
        frame_number = 0
        
        logging.info(f"Processing video for description with interval {interval} frames")
        
        # Collect frames
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            if frame_number % interval == 0:
                timestamp = frame_number / frame_rate
                frames_to_process.append((frame_number, frame, timestamp))
            
            frame_number += 1
            
        cap.release()
        
        descriptions = []
        
        # Process frames using thread pool
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_frame = {
                executor.submit(self.describe_single_frame, frame_data): frame_data
                for frame_data in frames_to_process
            }
            
            for i, future in enumerate(as_completed(future_to_frame)):
                frame_data = future_to_frame[future]
                try:
                    frame_desc = future.result()
                    descriptions.append(frame_desc)
                    
                    # Update progress
                    progress = int((i + 1) / len(frames_to_process) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {i + 1}/{len(frames_to_process)}")
                    
                except Exception as e:
                    logging.error(f"Error processing frame {frame_data[0]}: {str(e)}")
        
        # Sort by frame number
        descriptions.sort(key=lambda x: x['frame_number'])
        return descriptions

    def describe_single_frame(self, frame_data: Tuple[int, np.ndarray, float]) -> dict:
        """Generate description for a single frame."""
        frame_number, frame, timestamp = frame_data
        
        # Convert BGR to RGB
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Get description
        description = self.model.query(image, "Describe this image in detail.")
        
        # Save frame
        frame_path = self.temp_dir / f"frame_{frame_number:06d}.jpg"
        image.save(frame_path)
        
        return {
            'timestamp': timestamp,
            'frame_number': frame_number,
            'description': description['answer'],
            'image_path': str(frame_path)
        }
def initialize_model():
    if 'model' not in st.session_state:
        st.session_state.model = md.vl(model="moondream-0_5b-int8.mf")
    return st.session_state.model

def main():
    st.title("Advanced Video Analysis with Moondream")
    
    model = initialize_model()
    processor = VideoProcessor(model)
    
    st.header("Upload Media")
    source_type = st.radio("Select source type:", ["Upload Video", "Upload Image", "Use System Path"])
    
    file_path = None
    
    if source_type == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            file_path = tfile.name
            
    elif source_type == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            file_path = uploaded_file
            
    else:
        file_path = st.text_input("Enter file path:")
    
    st.header("Analysis Options")
    analysis_type = st.selectbox("Select Analysis Type:", 
                                ["General Description", "Custom Question", "Object Detection"])
    
    if file_path:
        if source_type == "Upload Video":
            st.header("Video Processing")
            frame_interval = st.slider("Frame Interval (seconds)", 1, 10, 1)
            
            if analysis_type == "Object Detection":
                object_to_detect = st.text_input("Enter object to detect in video:", "person")
                save_results = st.checkbox("Save results after processing", value=False)
                
                if st.button("Process Video"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Process video for object detection
                        processed_frames = processor.process_video(
                            file_path, frame_interval, object_to_detect,
                            progress_bar, status_text
                        )
                        
                        # Display results
                        st.header("Processing Results")
                        for frame in processed_frames:
                            if frame.image_path and Path(frame.image_path).exists():
                                st.image(frame.image_path, 
                                       caption=f"Frame at {frame.timestamp:.2f}s")
                                st.write(f"Detections at {frame.timestamp:.2f}s:", 
                                       frame.detections)
                        
                        # Save results if requested
                        if save_results:
                            # [existing save_results code remains the same]
                            pass
                            
                    finally:
                        processor.cleanup()
                        if Path(file_path).exists():
                            os.unlink(file_path)
                        gc.collect()
                        
            elif analysis_type == "General Description":
                if st.button("Generate Description"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Process video for description
                        descriptions = processor.process_video_for_description(
                            file_path, frame_interval,
                            progress_bar, status_text
                        )
                        
                        # Display results
                        st.header("Video Description Results")
                        for desc in descriptions:
                            if Path(desc['image_path']).exists():
                                st.image(desc['image_path'],
                                       caption=f"Frame at {desc['timestamp']:.2f}s")
                                st.write(f"Description at {desc['timestamp']:.2f}s:")
                                st.write(desc['description'])
                                st.markdown("---")
                                
                    finally:
                        processor.cleanup()
                        if Path(file_path).exists():
                            os.unlink(file_path)
                        gc.collect()
            elif analysis_type == "Custom Question":
                custom_question = st.text_input("Enter your question about the image:")
                if st.button("Ask Question"):
                    if source_type in ["Upload Image", "Use System Path"]:
                        image = Image.open(file_path)
                        st.image(image, caption="Uploaded Image")
                        answer = model.query(image, question=custom_question)
                        st.write("Answer:", answer['answer'])
                        
            elif analysis_type == "Object Detection":
                object_to_detect = st.text_input("Enter object to detect:", "person")
                if st.button("Detect Objects"):
                    if source_type in ["Upload Image", "Use System Path"]:
                        image = Image.open(file_path)
                        detections = model.detect(image, object_to_detect)
                        annotated_image = processor.draw_detection_boxes(
                            image, detections, object_to_detect
                        )
                        st.image(annotated_image, caption="Detection Results")
                        st.write("Detections:", detections)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()