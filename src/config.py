import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DEBUG_DIR = os.path.join(OUTPUT_DIR, 'debug')
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
BORDERS_DIR = os.path.join(OUTPUT_DIR, 'borders')

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
os.makedirs(BORDERS_DIR, exist_ok=True)

# Video Processing
FRAME_SAMPLING_INTERVAL_SEC = 0.2  # Sample at 5 fps to catch rapid year changes
SCENE_CHANGE_THRESHOLD = 30.0    # Percentage or absolute value diff for scene change

# Image Processing
ROI_MASK_PATH = os.path.join(INPUT_DIR, 'roi_mask.png') # Optional mask file
COLOR_QUANTIZATION_K = 32 # Increased to 32 for better color separation
MORPH_KERNEL_SIZE = (3, 3)
MORPH_CLOSING_KERNEL_SIZE = (7, 7) # Larger kernel for healing text/border gaps

# Text Removal (Legacy - kept for reference but unused in new strategy)
TEXT_WHITE_THRESHOLD = 200 
TEXT_BLACK_THRESHOLD = 50  
MAX_TEXT_COMPONENT_AREA = 800 
MIN_TEXT_SOLIDITY = 0.4 
MIN_TEXT_ASPECT_RATIO = 0.2 
MAX_TEXT_ASPECT_RATIO = 5.0

# OCR
# Coordinates for Year ROI (x, y, w, h) - needs calibration based on video
# Example: Top-right corner
YEAR_ROI = (0.4208, 0.8870, 0.0766, 0.0750) # Normalized coordinates (x_start, y_start, width, height)

# Performance Optimization
LABEL_OCR_INTERVAL_FRAMES = 10  # Run label OCR only every N frames (labels change rarely)
INCREMENTAL_SAVE_INTERVAL = 5   # Save results to CSV every N frames

# State Detection
MIN_STATE_AREA_PX = 500 # Minimum area to be considered a state
MIN_ISLAND_AREA_PX = 50 # Minimum area to be considered a valid island (vs noise)
SMOOTHING_WINDOW_SIZE = 5
