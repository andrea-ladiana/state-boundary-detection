import argparse
import cv2
import pandas as pd
import os
import numpy as np
import sys
import time
import torch
import json
from tqdm import tqdm

# Add src to path if running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.ingestion import FrameExtractor, extract_year
from src.preprocessing import apply_roi_mask, quantize_colors_lab, get_sea_color, create_solid_state_mask, perform_watershed_expansion
from src.metrics import get_contours, calculate_area, calculate_perimeter
from src.validation import smooth_series, check_consistency

def process_video(video_path, output_path, debug=False):
    # Check GPU availability
    use_gpu = torch.cuda.is_available()
    print(f"GPU Available: {use_gpu}")
    if use_gpu:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (Expect slower performance)")

    print(f"Processing video: {video_path}")
    
    # Initialize Extractor (EasyOCR handles GPU internally if available)
    extractor = FrameExtractor(video_path, use_gpu=use_gpu)
    
    results = []
    
    # Year tracking
    prev_year_roi = None
    current_year = None
    ocr_count = 0
    frames_since_last_ocr = 0
    
    # Profiling stats
    timers = {
        'read': 0.0,
        'ocr': 0.0,
        'preprocess': 0.0,
        'watershed': 0.0,
        'metrics': 0.0,
        'total': 0.0
    }
    
    # Frame generator
    # Use sequential extraction to catch every single frame (no skipping)
    # This ensures we don't miss years in fast-paced videos.
    frame_generator = extractor.extract_frames_sequential(step=1)
    
    # Total frames is exact
    pbar = tqdm(total=extractor.total_frames)
    
    try:
        for time_sec, frame in frame_generator:
            t_start_frame = time.time()
            pbar.update(1)
            
            # 1. Smart Year Recognition
            t0 = time.time()
            h, w = frame.shape[:2]
            rx, ry, rw, rh = config.YEAR_ROI
            x = int(rx * w)
            y = int(ry * h)
            width = int(rw * w)
            height = int(rh * h)
            year_roi = frame[y:y+height, x:x+width]
            year_roi_gray = cv2.cvtColor(year_roi, cv2.COLOR_BGR2GRAY)
            
            # Check if year ROI changed significantly
            year_changed = False
            frames_since_last_ocr += 1
            
            if prev_year_roi is not None:
                diff = cv2.absdiff(year_roi_gray, prev_year_roi)
                mean_diff = np.mean(diff)
                # Force OCR if changed OR if we haven't checked in a while (fallback)
                if mean_diff > 10 or frames_since_last_ocr >= 5: 
                    year_changed = True
            else:
                year_changed = True  # First frame
            
            if year_changed:
                # Run OCR
                year = extract_year(frame, extractor.reader)
                if year is not None:
                    current_year = year
                    ocr_count += 1
                    frames_since_last_ocr = 0 # Reset counter
                prev_year_roi = year_roi_gray.copy()
            else:
                # Reuse last known year
                year = current_year
            
            timers['ocr'] += time.time() - t0
            
            if year is None:
                continue
            
            # Skip if this is the same year as the last saved data
            if results and results[-1]['Year'] == year:
                continue
                
            # 2. Preprocessing & Quantization
            t1 = time.time()
            masked = apply_roi_mask(frame)
            
            # Optimization: Downscale for K-Means AND Watershed speedup
            # Resize to 25% size (e.g. 1080p -> 270p)
            scale_factor = 0.25
            h_small, w_small = int(h * scale_factor), int(w * scale_factor)
            small_masked = cv2.resize(masked, (w_small, h_small))
            
            small_quantized = quantize_colors_lab(small_masked, k=config.COLOR_QUANTIZATION_K)
            timers['preprocess'] += time.time() - t1
            
            # 3. Sea Identification (on small image)
            sea_color = get_sea_color(small_quantized)
            
            if debug:
                # Save debug frame (upscaled for visibility)
                debug_fname = os.path.join(config.DEBUG_DIR, f"frame_{year}_{int(time_sec)}.png")
                cv2.imwrite(debug_fname, small_quantized)
                
            # 4. State Extraction (Topology Refinement)
            t2 = time.time()
            
            # Step 1: Collect Solid Masks for all candidate states (on small image)
            pixels = small_quantized.reshape(-1, 3)
            unique_colors = np.unique(pixels, axis=0)
            
            small_state_masks = {}
            
            # Adjust kernel size for small image
            small_kernel_size = (max(1, int(config.MORPH_CLOSING_KERNEL_SIZE[0] * scale_factor)), 
                                 max(1, int(config.MORPH_CLOSING_KERNEL_SIZE[1] * scale_factor)))
            
            for color in unique_colors:
                # Skip Sea Color
                if np.array_equal(color, sea_color):
                    continue
                    
                # Skip Pure Black (Borders/Text)
                if np.mean(color) < config.TEXT_BLACK_THRESHOLD:
                    continue
                    
                # Create Solid Mask (Refinement A & C)
                color_hex = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
                # Use small kernel
                mask = create_solid_state_mask(small_quantized, color, closing_kernel_size=small_kernel_size)
                
                # Only keep if it has significant area (scaled down threshold)
                min_area_small = config.MIN_STATE_AREA_PX * (scale_factor ** 2)
                if cv2.countNonZero(mask) >= min_area_small:
                    small_state_masks[color_hex] = mask
                    
            # Step 2: Watershed Expansion (Refinement B) on small image
            small_labeled_map, index_to_hex = perform_watershed_expansion(small_quantized, small_state_masks, sea_color)
            timers['watershed'] += time.time() - t2
            
            # Step 3: Metrics from Watershed Result
            # Upscale the labeled map to full resolution for accurate metrics
            t3 = time.time()
            labeled_map = cv2.resize(small_labeled_map, (w, h), interpolation=cv2.INTER_NEAREST)
            
            year_contours = []
            
            unique_indices = np.unique(labeled_map)
            
            for idx in unique_indices:
                if idx <= 1: continue # Skip Boundary (-1), Unknown (0), Sea (1)
                
                state_id = index_to_hex.get(idx)
                if not state_id: continue
                
                # Create mask from label map
                state_mask = (labeled_map == idx).astype(np.uint8) * 255
                
                # Area
                area = cv2.countNonZero(state_mask)
                
                # Perimeter
                contours = get_contours(state_mask)
                perimeter = 0
                for cnt in contours:
                    perimeter += calculate_perimeter(cnt)
                
                # Centroid
                cX, cY = 0, 0
                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest_cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                
                results.append({
                    'Year': year,
                    'State_ID': state_id,
                    'Centroid_X': cX,
                    'Centroid_Y': cY,
                    'Area_px': area,
                    'Perimeter_px': perimeter,
                    'Color': state_id
                })
                
                # Store contours for visualization
                r = int(state_id[1:3], 16)
                g = int(state_id[3:5], 16)
                b = int(state_id[5:7], 16)
                color_bgr = (b, g, r)
                
                for cnt in contours:
                    year_contours.append((cnt, color_bgr))

                # Export Borders to JSON
                contours_list = []
                for cnt in contours:
                    # cnt is (N, 1, 2) -> (N, 2) -> list of lists
                    contours_list.append(cnt.reshape(-1, 2).tolist())
                
                border_data = {
                    "year": year,
                    "state_id": state_id,
                    "contours": contours_list,
                    "dimensions": {"width": w, "height": h}
                }
                
                year_dir = os.path.join(config.BORDERS_DIR, str(year))
                os.makedirs(year_dir, exist_ok=True)
                
                # Remove # from filename
                safe_state_id = state_id.replace("#", "")
                border_fname = os.path.join(year_dir, f"{safe_state_id}.json")
                
                with open(border_fname, 'w') as f:
                    json.dump(border_data, f)
            
            timers['metrics'] += time.time() - t3
            
            # Create visualization
            if year_contours:
                vis_frame = frame.copy()
                for contour, color_bgr in year_contours:
                    bright_color = tuple([min(int(c * 1.4), 255) for c in color_bgr])
                    cv2.drawContours(vis_frame, [contour], -1, bright_color, 2)
                
                vis_fname = os.path.join(config.VISUALIZATION_DIR, f"year_{year:04d}.png")
                cv2.imwrite(vis_fname, vis_frame)
            
            timers['total'] += time.time() - t_start_frame
            
            # 5. Incremental save
            if len(results) % config.INCREMENTAL_SAVE_INTERVAL == 0 and results:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(output_path, index=False)
                unique_years = df_temp['Year'].nunique()
                avg_time = timers['total'] / (pbar.n if pbar.n > 0 else 1)
                pbar.set_description(f"Year {year} | {unique_years} yrs | {avg_time:.2f}s/it")
                
    except KeyboardInterrupt:
        print("Processing interrupted by user.")
        
    extractor.release()
    pbar.close()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No data extracted.")
        return

    # 6. Validation & Smoothing
    if 'State_ID' in df.columns:
        df['Area_px_Smooth'] = df.groupby('State_ID')['Area_px'].transform(lambda x: smooth_series(x))
        df['Perimeter_px_Smooth'] = df.groupby('State_ID')['Perimeter_px'].transform(lambda x: smooth_series(x))
    
    # Consistency Check
    df = check_consistency(df)
    
    # Save final
    df.to_csv(output_path, index=False)
    print(f"\nData saved to {output_path}")
    print(f"Total entries: {len(df)}")
    print(f"OCR executions: {ocr_count}")
    
    print("\nPerformance Profile:")
    print(f"Total Time: {timers['total']:.2f}s")
    print(f"OCR Time: {timers['ocr']:.2f}s")
    print(f"Preprocess (K-Means) Time: {timers['preprocess']:.2f}s")
    print(f"Watershed Time: {timers['watershed']:.2f}s")
    print(f"Metrics Time: {timers['metrics']:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical State Boundaries Extraction Pipeline")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", default="output/extracted_metrics.csv", help="Path to output CSV file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (save frames)")
    
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.debug)
