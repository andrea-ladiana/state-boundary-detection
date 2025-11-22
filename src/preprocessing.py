import os
import cv2
import numpy as np
from . import config

def apply_roi_mask(frame, mask_path=config.ROI_MASK_PATH):
    """
    Applies a static mask to the frame to exclude non-map elements.
    If mask_path doesn't exist, returns frame as is (or creates a dummy mask).
    """
    if not mask_path or not os.path.exists(mask_path):
        # Warning: No mask found
        return frame
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return frame
        
    # Resize mask to match frame if necessary
import os
import cv2
import numpy as np
from . import config

def apply_roi_mask(frame, mask_path=config.ROI_MASK_PATH):
    """
    Applies a static mask to the frame to exclude non-map elements.
    If mask_path doesn't exist, returns frame as is (or creates a dummy mask).
    """
    if not mask_path or not os.path.exists(mask_path):
        # Warning: No mask found
        return frame
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return frame
        
    # Resize mask to match frame if necessary
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_frame

def quantize_colors_lab(image, k=config.COLOR_QUANTIZATION_K):
    """
    Quantizes image colors using K-Means in LAB color space.
    Better for perceptual color grouping.
    """
    # Convert to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Reshape to a list of pixels
    pixel_values = lab_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria = ( type, max_iter = 100, epsilon = 0.2 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Perform k-means clustering
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8 bit values
    centers = np.uint8(centers)
    
    # Map labels to center values
    segmented_lab = centers[labels.flatten()]
    
    # Reshape back to original image dimension
    segmented_lab = segmented_lab.reshape(lab_image.shape)
    
    # Convert back to BGR
    segmented_bgr = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2BGR)
    
    return segmented_bgr

def get_sea_color(image):
    """
    Identifies the background/sea color from the top-left pixel.
    """
    # Assume (0,0) is sea
    sea_pixel = image[0, 0]
    return sea_pixel

def create_solid_state_mask(image, target_color, closing_kernel_size=config.MORPH_CLOSING_KERNEL_SIZE):
    """
    Creates a solid binary mask for a specific color.
    1. Thresholds by color.
    2. Applies morphological closing.
    3. Fills internal holes using RETR_EXTERNAL contours.
    4. Removes small noise islands.
    """
    # Create mask for the target color
    lower_bound = np.array([max(c - 2, 0) for c in target_color])
    upper_bound = np.array([min(c + 2, 255) for c in target_color])
    
    mask = cv2.inRange(image, lower_bound, upper_bound)
    
    # Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, closing_kernel_size)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Solid Hull (Refinement A: Internal Topology Healing)
    # Find external contours only (ignores internal holes like rivers/text)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    solid_mask = np.zeros_like(mask)
    cv2.drawContours(solid_mask, contours, -1, 255, thickness=cv2.FILLED)
    
    # Small Island Removal (Refinement C: Despeckling)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(solid_mask, connectivity=8)
    
    final_mask = np.zeros_like(solid_mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= config.MIN_ISLAND_AREA_PX:
            final_mask[labels == i] = 255
            
    return final_mask

def perform_watershed_expansion(quantized_image, state_masks, sea_color):
    """
    Refinement B: Domain Expansion via Watershed.
    Expands state regions to fill black borders and touch each other.
    
    Args:
        quantized_image: The quantized BGR image.
        state_masks: Dict {state_id_hex: binary_mask}
        sea_color: BGR color of the sea.
        
    Returns:
        labeled_map: Image where pixel value = state_index (0 for boundaries/unknown).
        index_to_hex: Dict mapping state_index to state_id_hex.
    """
    h, w = quantized_image.shape[:2]
    markers = np.zeros((h, w), dtype=np.int32)
    
    # 1. Define Markers
    # ID 1: Sea (Background)
    sea_mask = create_solid_state_mask(quantized_image, sea_color)
    # Erode sea slightly to be safe
    sea_mask = cv2.erode(sea_mask, np.ones((3,3), np.uint8), iterations=1)
    markers[sea_mask > 0] = 1
    
    # ID 2...N: States
    index_to_hex = {}
    current_id = 2
    
    for hex_id, mask in state_masks.items():
        # Erode state mask to be a "sure foreground" seed
        # This ensures we don't start watershed from the fuzzy edges
        sure_fg = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)
        
        # If erosion kills the state (too small), use the original mask
        if cv2.countNonZero(sure_fg) == 0:
            sure_fg = mask
            
        markers[sure_fg > 0] = current_id
        index_to_hex[current_id] = hex_id
        current_id += 1
        
    # 2. Run Watershed
    # Watershed needs a 3-channel image. We use the quantized one.
    # It modifies markers in-place. -1 will be boundaries.
    cv2.watershed(quantized_image, markers)
    
    # 3. Post-process: Assign boundaries (-1) to nearest neighbor?
    # For physics, boundaries are fine as 0 or -1 (domain walls).
    # But for area calculation, we might want to split them.
    # For now, we leave them as boundaries (0 in our logic, -1 in OpenCV).
    
    return markers, index_to_hex

def quantize_colors(image, k=config.COLOR_QUANTIZATION_K):
    """
    Reduces the number of colors in the image using K-Means clustering.
    """
    # Reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria = ( type, max_iter, epsilon )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # K-Means
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8 bit values
    centers = np.uint8(centers)

    # Flatten the labels array
    labels = labels.flatten()

    # Convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    
    return segmented_image

def denoise_morphological(image, kernel_size=config.MORPH_KERNEL_SIZE):
    """
    Applies morphological operations to remove noise.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Opening (Erosion followed by Dilation) removes small noise
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Closing (Dilation followed by Erosion) fills small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed

def clean_map_artifacts(image):
    """
    Advanced Dual-Phase Text Removal & Artifact Cleaning.
    Removes white and black text while preserving state borders.
    """
    # 1. White Masking (Titles, Dates, Logos)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, config.TEXT_WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Dilate white mask to capture anti-aliasing
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.dilate(white_mask, kernel, iterations=1)
    
    # 2. Inpaint Pass 1 (Remove White Text)
    # Use Telea inpainting
    inpainted_1 = cv2.inpaint(image, white_mask, 3, cv2.INPAINT_TELEA)
    
    # 3. Black Analysis (State Names vs Borders)
    # Work on the inpainted image
    gray_2 = cv2.cvtColor(inpainted_1, cv2.COLOR_BGR2GRAY)
    _, black_thresh = cv2.threshold(gray_2, config.TEXT_BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # 4. Connected Components Filter with Edge Protection
    # We want to keep small components (text) and ignore large ones (borders)
    # CRITICAL FIX: Protect borders using Canny Edge Detection
    
    # Create a "Safety Mask" from edges
    # Use Canny on the original gray image to find structural lines
    edges = cv2.Canny(gray_2, 50, 150)
    # Dilate edges slightly to create a safety zone
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_thresh, connectivity=8)
    
    text_mask = np.zeros_like(black_thresh)
    
    for i in range(1, num_labels): # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Initial Area Check - if it's huge, it's definitely a border
        if area >= config.MAX_TEXT_COMPONENT_AREA:
            continue 
            
        # Create a mask for just this component
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # CHECK 1: Edge Overlap (Safety Check)
        # Calculate overlap with the edge map
        overlap = cv2.bitwise_and(component_mask, edges_dilated)
        overlap_pixels = cv2.countNonZero(overlap)
        component_pixels = area
        
        # If significant overlap (>50%), it's part of the structure/border -> PROTECT IT
        if component_pixels > 0 and (overlap_pixels / component_pixels) > 0.5:
            continue # It's a border segment, skip removal
            
        # CHECK 2: Geometric Analysis (Secondary Check for Text)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        cnt = contours[0]
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Decision Logic: Is it Text?
        # Text is: Small Area AND High Solidity AND Normal Aspect Ratio AND NOT on an edge
        is_text = (
            area < config.MAX_TEXT_COMPONENT_AREA and
            solidity > config.MIN_TEXT_SOLIDITY and
            config.MIN_TEXT_ASPECT_RATIO < aspect_ratio < config.MAX_TEXT_ASPECT_RATIO
        )
        
        if is_text:
            text_mask[labels == i] = 255
            
    # 5. Black Masking & Dilation
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)
    
    # 6. Inpaint Pass 2 (Remove Black Text)
    inpainted_2 = cv2.inpaint(inpainted_1, text_mask, 3, cv2.INPAINT_TELEA)
    
    # 7. Denoise / Quantize
    # Re-apply quantization to smooth out inpainting artifacts
    final_clean = quantize_colors(inpainted_2, k=config.COLOR_QUANTIZATION_K)
    
    return final_clean
