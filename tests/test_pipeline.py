import unittest
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import calculate_area, calculate_perimeter, get_contours
from src.preprocessing import quantize_colors
from src.detection import find_connected_components

class TestPipeline(unittest.TestCase):
    def test_metrics_square(self):
        # Create a 100x100 image with a 50x50 square
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (25, 25), (75, 75), 255, -1)
        
        contours = get_contours(image)
        self.assertEqual(len(contours), 1)
        
        area = calculate_area(contours[0])
        perimeter = calculate_perimeter(contours[0])
        
        # Area should be roughly 50*50 = 2500
        self.assertAlmostEqual(area, 2500, delta=50)
        
        # Perimeter should be roughly 50*4 = 200
        self.assertAlmostEqual(perimeter, 200, delta=10)

    def test_quantization(self):
        # Create image with 3 distinct colors
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[0:50, :] = [255, 0, 0] # Blue
        image[50:100, :] = [0, 255, 0] # Green
        
        quantized = quantize_colors(image, k=2)
        unique_colors = np.unique(quantized.reshape(-1, 3), axis=0)
        self.assertTrue(len(unique_colors) <= 2)

    def test_connected_components(self):
        # Create image with 2 disjoint blobs
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (30, 30), 255, -1)
        cv2.rectangle(image, (60, 60), (80, 80), 255, -1)
        
        labels, stats, centroids = find_connected_components(image)
        
        # Should have background + 2 components
        # Note: find_connected_components returns labels map, stats list, centroids list
        # The length of stats includes background if we didn't filter it?
        # In our implementation we return all_stats which excludes background for each color pass?
        # Wait, let's check implementation of find_connected_components
        
        # In detection.py:
        # for i in range(1, num_labels): ... all_stats.append(stats[i])
        # So it returns only components, not background.
        
        # The implementation processes all unique colors, including the background (0).
        # So we expect: 1 component for background (black) + 2 components for rectangles (white).
        self.assertEqual(len(stats), 3)

if __name__ == '__main__':
    unittest.main()
