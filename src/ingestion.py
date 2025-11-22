import cv2
import numpy as np
import easyocr
import os
from . import config

class FrameExtractor:
    def __init__(self, video_path, use_gpu=True):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
        # Initialize OCR reader (English is standard for dates usually, but can add others)
        self.reader = easyocr.Reader(['en'], gpu=use_gpu) 

    def release(self):
        self.cap.release()

    def get_frame_at_time(self, time_sec):
        """Extracts a frame at a specific timestamp."""
        frame_no = int(time_sec * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def extract_frames_sampling(self, interval_sec=config.FRAME_SAMPLING_INTERVAL_SEC):
        """Generator that yields frames at fixed intervals."""
        current_time = 0
        while current_time < self.duration:
            frame = self.get_frame_at_time(current_time)
            if frame is not None:
                yield current_time, frame
            current_time += interval_sec

    def extract_frames_sequential(self, step=1):
        """
        Generator that yields frames sequentially without seeking.
        Much faster for processing every frame.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_idx % step == 0:
                time_sec = frame_idx / self.fps
                yield time_sec, frame
            
            frame_idx += 1

    def extract_frames_scene_change(self, threshold=config.SCENE_CHANGE_THRESHOLD):
        """
        Generator that yields frames when a scene change is detected.
        Uses simple pixel difference.
        """
        prev_frame = None
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if prev_frame is None:
                prev_frame = frame
                yield frame_idx / self.fps, frame
                frame_idx += 1
                continue

            # Calculate difference
            # Resize for speed
            small_curr = cv2.resize(frame, (64, 64))
            small_prev = cv2.resize(prev_frame, (64, 64))
            
            diff = cv2.absdiff(small_curr, small_prev)
            non_zero_count = np.count_nonzero(diff)
            percentage_diff = (non_zero_count / diff.size) * 100

            if percentage_diff > threshold:
                yield frame_idx / self.fps, frame
                prev_frame = frame
            
            frame_idx += 1

def extract_year(frame, reader):
    """
    Extracts year from the frame using OCR.
    Uses config.YEAR_ROI to crop the image first.
    """
    h, w = frame.shape[:2]
    rx, ry, rw, rh = config.YEAR_ROI
    
    # Convert normalized coordinates to pixel coordinates
    x = int(rx * w)
    y = int(ry * h)
    width = int(rw * w)
    height = int(rh * h)
    
    # Crop
    roi = frame[y:y+height, x:x+width]
    
    # Preprocess for OCR (grayscale, threshold)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Simple thresholding might help, or adaptive
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Run OCR
    # detail=0 returns just the text
    results = reader.readtext(thresh, detail=0)
    
    # Parse results to find a year (4 digits)
    # This is a heuristic
    for text in results:
        clean_text = ''.join(filter(str.isdigit, text))
        if len(clean_text) == 4:
            return int(clean_text)
            
    return None
