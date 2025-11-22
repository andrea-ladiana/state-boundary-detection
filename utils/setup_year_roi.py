import cv2
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import FrameExtractor

class ROISelector:
    def __init__(self, frame):
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.roi = None
        
    def onselect(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        
    def select_roi(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(self.frame)
        ax.set_title("Click and drag to select the YEAR region, then close the window")
        
        rs = RectangleSelector(ax, self.onselect,
                              useblit=True,
                              button=[1],
                              minspanx=5, minspany=5,
                              spancoords='pixels',
                              interactive=True)
        
        plt.show()
        return self.roi

def setup_year_roi(video_path):
    """
    Interactive tool to select the Year ROI from a sample frame.
    """
    print(f"Loading video: {video_path}")
    extractor = FrameExtractor(video_path)
    
    # Get a frame from the middle of the video
    mid_time = extractor.duration / 2
    frame = extractor.get_frame_at_time(mid_time)
    
    if frame is None:
        print("ERROR: Could not extract frame from video.")
        extractor.release()
        return
    
    print("\nInstructions:")
    print("1. A matplotlib window will open showing a frame from the middle of the video")
    print("2. Click and drag to select the region containing the YEAR")
    print("3. You can adjust the selection by dragging the corners/edges")
    print("4. Close the window when satisfied")
    print("5. The normalized coordinates will be saved to config.py\n")
    
    # Select ROI
    selector = ROISelector(frame)
    roi = selector.select_roi()
    
    if roi is None or roi[2] == 0 or roi[3] == 0:
        print("No ROI selected. Exiting.")
        extractor.release()
        return
    
    x, y, w, h = roi
    
    # Get frame dimensions
    frame_h, frame_w = frame.shape[:2]
    
    # Normalize coordinates
    x_norm = x / frame_w
    y_norm = y / frame_h
    w_norm = w / frame_w
    h_norm = h / frame_h
    
    print(f"\nSelected ROI (pixels): x={x}, y={y}, w={w}, h={h}")
    print(f"Normalized ROI: x={x_norm:.4f}, y={y_norm:.4f}, w={w_norm:.4f}, h={h_norm:.4f}")
    
    # Update config.py
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'config.py')
    
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Find and replace YEAR_ROI line
    updated = False
    for i, line in enumerate(lines):
        if line.startswith('YEAR_ROI'):
            lines[i] = f"YEAR_ROI = ({x_norm:.4f}, {y_norm:.4f}, {w_norm:.4f}, {h_norm:.4f}) # Normalized coordinates (x_start, y_start, width, height)\n"
            updated = True
            break
    
    if updated:
        with open(config_path, 'w') as f:
            f.writelines(lines)
        print(f"\nSuccessfully updated config.py with new YEAR_ROI coordinates!")
        print("You can now run the pipeline with: python src/main.py --input input/test.mp4")
    else:
        print("\nERROR: Could not find YEAR_ROI in config.py")
    
    extractor.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utils/setup_year_roi.py <path_to_video>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    setup_year_roi(video_path)
