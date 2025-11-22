import cv2
import numpy as np
from . import config

def find_connected_components(image):
    """
    Finds connected components in the quantized image.
    Returns:
        num_labels: number of labels
        labels: image with label per pixel
        stats: stats matrix (x, y, w, h, area)
        centroids: centroids of components
    """
    # Convert to grayscale if not already (though quantized might be colored)
    # If quantized is colored, we might need to process each color separately or 
    # convert to grayscale. However, different states might have different colors 
    # but same grayscale value.
    # Better approach: Iterate over unique colors (except background) and find components for each.
    # For simplicity in this pipeline, we assume the quantized image has distinct colors for states.
    # We can convert to grayscale for connected components if we assume neighbors have different colors.
    # BUT, if two states share a color but are disjoint, they are different components.
    # If they touch and share a color, they merge (which is a limitation of the map itself).
    
    # Let's try converting to grayscale and finding components. 
    # Note: cv2.connectedComponents expects a binary image or single channel.
    # If we just convert to gray, different colors might map to same gray.
    # A robust way is to treat each unique color as a mask.
    
    # Optimization: Just use Canny edges or gradient to separate, then fill?
    # Or just run connected components on the packed 32-bit integer representation of RGB.
    
    # Pack RGB into single int32 for labeling
    if len(image.shape) == 3:
        # (H, W, 3)
        b, g, r = cv2.split(image)
        # We can't easily use connectedComponents on 32-bit int directly in OpenCV python wrapper 
        # in a way that distinguishes colors perfectly without thresholding.
        # Alternative: Use skimage.measure.label which supports multidimensional
        
        # For OpenCV speed:
        # We can iterate over unique colors.
        unique_colors = np.unique(image.reshape(-1, 3), axis=0)
        
        all_labels = np.zeros(image.shape[:2], dtype=np.int32)
        current_label_offset = 1
        all_stats = []
        all_centroids = []
        
        # Background color assumption: Black or specific color? 
        # We'll assume the largest area color is background or we process all.
        
        for color in unique_colors:
            # Create mask for this color
            mask = cv2.inRange(image, color, color)
            
            # Find components for this color
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
            
            # Skip background (label 0) of the mask
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < config.MIN_STATE_AREA_PX:
                    continue
                    
                # Update global labels
                # We need to assign a unique ID
                new_id = current_label_offset
                all_labels[labels == i] = new_id
                
                all_stats.append(stats[i])
                all_centroids.append(centroids[i])
                
                current_label_offset += 1
                
        return all_labels, all_stats, all_centroids
        
    else:
        # Already single channel
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
        # Filter small areas?
        return labels, stats, centroids

def associate_labels(frame, labels_map, reader):
    """
    Runs OCR on the frame to find text labels and associates them with blobs in labels_map.
    
    Args:
        frame: Original frame (or preprocessed) for OCR.
        labels_map: 2D array where each pixel has a component ID.
        reader: EasyOCR reader instance.
        
    Returns:
        mapping: Dict { component_id: "Label Text" }
    """
    # Run OCR on the full frame
    # detail=1 returns (bbox, text, prob)
    results = reader.readtext(frame, detail=1)
    
    mapping = {}
    
    for (bbox, text, prob) in results:
        # bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        # Calculate centroid of the text box
        pts = np.array(bbox, dtype=np.int32)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        
        # Check bounds
        h, w = labels_map.shape
        if 0 <= cx < w and 0 <= cy < h:
            # Get component ID at centroid
            comp_id = labels_map[cy, cx]
            
            if comp_id > 0:
                # We found a match
                # If multiple texts map to same blob, maybe concat or take longest?
                if comp_id in mapping:
                    mapping[comp_id] += " " + text
                else:
                    mapping[comp_id] = text
                    
    return mapping
