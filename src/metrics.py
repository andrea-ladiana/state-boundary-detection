import cv2
import numpy as np

def get_contours(component_mask):
    """
    Extracts contours from a binary component mask using Suzuki's algorithm (cv2.findContours).
    """
    # Ensure mask is uint8
    mask = component_mask.astype(np.uint8)
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8) * 255
        
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calculate_area(contour):
    """
    Calculates area using Green's theorem (cv2.contourArea).
    """
    return cv2.contourArea(contour)

def calculate_perimeter(contour, epsilon_factor=0.001):
    """
    Calculates perimeter using Arc Length after Douglas-Peucker approximation.
    
    Args:
        contour: The contour points.
        epsilon_factor: Factor to multiply by arc length for epsilon in approxPolyDP.
                        Lower means more detailed.
    """
    # Calculate initial arc length
    perimeter_raw = cv2.arcLength(contour, True)
    
    # Apply Douglas-Peucker approximation
    epsilon = epsilon_factor * perimeter_raw
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Calculate perimeter of the approximated polygon
    perimeter_approx = cv2.arcLength(approx, True)
    
    return perimeter_approx
