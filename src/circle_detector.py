#!/usr/bin/env python3
"""
Circle Detection and YOLO Annotation Generator

This script detects black circles with white centers in images and generates
YOLO format annotation files for computer vision model training.

Requirements:
    pip install opencv-python numpy

Usage:
    python circle_detector.py [--input_dir data/raw/] [--output_dir annotations/] [--debug]
"""

import cv2
import numpy as np
import os
import glob
import argparse
import sys
from pathlib import Path
from itertools import combinations




def find_bright_rectangular_area(img, debug=False):
    """
    Find the largest area that is predominantly bright (the target).
    
    Args:
        img: Input image
        debug: Whether to show debug visualization
    
    Returns:
        tuple: (x, y, w, h) of the target area, or None if not found
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    # Use a lower threshold to capture more areas, then analyze brightness
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Analyze each contour for brightness ratio and size
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000:  # Higher minimum area threshold
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check aspect ratio (should be roughly rectangular)
        aspect_ratio = w / h if h > 0 else 0
        if 0.2 < aspect_ratio < 3.0:  # More permissive aspect ratio
            # Skip contours that start at (0,0) coordinates
            if x == 0 and y == 0:
                continue
            
            # Skip if this contour occupies almost the entire image
            image_area = img.shape[0] * img.shape[1]
            bounding_area = w * h  # Use bounding rectangle area instead of contour area
            area_ratio = bounding_area / image_area
            if area_ratio > 0.8:  # Skip if more than 80% of image (more aggressive)
                            # if debug:
            #     print(f"Skipping contour at ({x},{y}) size ({w},{h}) - covers {area_ratio*100:.1f}% of image")
                continue
            
            # Calculate brightness ratio within this area
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Get pixels within the contour
            pixels = gray[mask == 255]
            if len(pixels) == 0:
                continue
            
            # Calculate brightness statistics
            mean_brightness = np.mean(pixels)
            bright_pixels = np.sum(pixels > 150)  # Count bright pixels
            total_pixels = len(pixels)
            bright_ratio = bright_pixels / total_pixels if total_pixels > 0 else 0
            
            # Score based on area and brightness ratio
            score = area * bright_ratio
            
            valid_contours.append((contour, x, y, w, h, area, mean_brightness, bright_ratio, score))
    
    if not valid_contours:
        return None
    
    # Sort by score (area * brightness ratio) - largest bright area wins
    valid_contours.sort(key=lambda x: x[8], reverse=True)
    
    # Take the best contour
    contour, x, y, w, h, area, mean_brightness, bright_ratio, score = valid_contours[0]
    
    # Ensure it's reasonably large (at least 5% of image dimensions)
    min_area = img.shape[0] * img.shape[1] * 0.05
    if area >= min_area:
                        # if debug:
                #     # Show all contours found
                #     debug_img = img.copy()
                #     for i, (cont, cx, cy, cw, ch, carea, cmean, cbright, cscore) in enumerate(valid_contours[:10]):  # Show top 10
                #         # Draw red rectangle for each contour
                #         cv2.rectangle(debug_img, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
                #         # Add text with area and brightness info
                #         text = f"Area: {carea}, Bright: {cbright:.2f}, Score: {cscore:.0f}"
                #         cv2.putText(debug_img, text, (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                #     
                #     # Highlight the selected contour in green
                #     cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                #     cv2.putText(debug_img, f"SELECTED: Area {area}, Bright {bright_ratio:.2f}, Score {score:.0f}", 
                #                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #     
                #     # Resize for display
                #     height, width = debug_img.shape[:2]
                #     max_width, max_height = 1200, 800
                #     if width > max_width or height > max_height:
                #         scale = min(max_width / width, max_height / height)
                #         new_width = int(width * scale)
                #         new_height = int(height * scale)
                #         debug_img = cv2.resize(debug_img, (new_width, new_height))
                #     
                #     cv2.namedWindow('Contours (brightness analysis)', cv2.WINDOW_NORMAL)
                #     cv2.imshow('Contours (brightness analysis)', debug_img)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
        
        return (x, y, w, h)
    
    return None


def validate_target_circle(img, x, y, r):
    """
    Validate if a circle matches the target circle criteria.
    
    Args:
        img: Input image
        x, y, r: Circle center and radius
    
    Returns:
        tuple: (is_valid, contrast_ratio, ring_width_ratio, ring_uniformity)
    """
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create masks for different regions
    outer_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(outer_mask, (x, y), r, 255, -1)
    
    # Estimate ring width (target circles have very thin rings)
    ring_width = r // 4  # Adjusted based on actual top-left ring ratio of 4.03
    inner_radius = r - ring_width
    
    # Create ring mask (outer circle minus inner circle)
    ring_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(ring_mask, (x, y), r, 255, -1)
    cv2.circle(ring_mask, (x, y), inner_radius, 0, -1)
    
    # Create center mask
    center_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(center_mask, (x, y), inner_radius, 255, -1)
    
    # Get pixel values
    ring_pixels = gray[ring_mask == 255]
    center_pixels = gray[center_mask == 255]
    
    if len(ring_pixels) == 0 or len(center_pixels) == 0:
        return False, 0, 0, 0
    
    ring_mean = np.mean(ring_pixels)
    center_mean = np.mean(center_pixels)
    ring_std = np.std(ring_pixels)  # Measure ring uniformity
    
    contrast_ratio = center_mean / (ring_mean + 1)
    
    # Calculate ring width ratio (white center diameter / black ring width)
    white_center_diameter = 2 * inner_radius
    black_ring_width = 2 * ring_width
    ring_width_ratio = white_center_diameter / black_ring_width
    
    # Ring uniformity (lower is better - more uniform)
    ring_uniformity = ring_std
    
    # Target circle validation criteria:
    # 1. Center should be brighter than ring (contrast_ratio > 0.4) - relaxed
    # 2. Ring width ratio should be around 3.0 (2.8 to 3.5) - based on actual detected circles
    # 3. Ring should be dark (< 250) - relaxed
    # 4. Ring should be uniform (low std deviation)
    is_valid = (contrast_ratio > 0.4 and 
                ring_width_ratio > 2.8 and ring_width_ratio < 3.5 and
                ring_mean < 250 and
                ring_uniformity < 60)  # Ring should be uniform
    
    return is_valid, contrast_ratio, ring_width_ratio, ring_uniformity


def validate_circle_properties_relaxed(img, x, y, r):
    """
    Relaxed validation for circle properties (for finding the 4th circle).
    
    Args:
        img: Input image
        x, y, r: Circle center and radius
    
    Returns:
        tuple: (is_valid, contrast_ratio, ring_width_ratio)
    """
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create masks for different regions
    outer_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(outer_mask, (x, y), r, 255, -1)
    
    # Estimate ring width (typically 1/4 to 1/3 of radius)
    ring_width = r // 3
    inner_radius = r - ring_width
    
    # Create ring mask (outer circle minus inner circle)
    ring_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(ring_mask, (x, y), r, 255, -1)
    cv2.circle(ring_mask, (x, y), inner_radius, 0, -1)
    
    # Create center mask
    center_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(center_mask, (x, y), inner_radius, 255, -1)
    
    # Get pixel values
    ring_pixels = gray[ring_mask == 255]
    center_pixels = gray[center_mask == 255]
    
    if len(ring_pixels) == 0 or len(center_pixels) == 0:
        return False, 0, 0
    
    ring_mean = np.mean(ring_pixels)
    center_mean = np.mean(center_pixels)
    contrast_ratio = center_mean / (ring_mean + 1)
    
    # Calculate ring width ratio (white center diameter / black ring width)
    white_center_diameter = 2 * inner_radius
    black_ring_width = 2 * ring_width
    ring_width_ratio = white_center_diameter / black_ring_width
    
    # Very relaxed validation criteria for the 4th circle:
    # 1. Center should be brighter than ring (contrast_ratio > 1.05) - very relaxed
    # 2. Ring width ratio should be reasonable (0.8 to 4.5) - very relaxed
    # 3. Ring should be reasonably dark (< 200) - very relaxed
    is_valid = (contrast_ratio > 1.05 and 
                ring_width_ratio > 0.8 and ring_width_ratio < 4.5 and
                ring_mean < 200)
    
    return is_valid, contrast_ratio, ring_width_ratio


def find_rectangle_circles(circles, tolerance=0.3, img_shape=None, target_area_size=None, img=None):
    """
    Find 4 circles that form a rectangle with specific criteria.
    
    Args:
        circles: List of (x, y, r) tuples
        tolerance: Size similarity tolerance
        img_shape: Image shape (height, width) for center calculation
        target_area_size: Size of target area (w, h) for area ratio calculation
        img: Image for property validation
    
    Returns:
        list: 4 circles forming rectangle, or empty list if not found
    """
    if len(circles) < 4:
        return []
    
    # Group circles by similar size
    radii = [r for x, y, r in circles]
    median_radius = np.median(radii)
    
    # Filter circles by size similarity
    size_similar_circles = []
    for x, y, r in circles:
        if abs(r - median_radius) / median_radius <= tolerance:
            size_similar_circles.append((x, y, r))
    
    if len(size_similar_circles) < 4:
        return []
    
    # Calculate target area size if not provided
    if target_area_size is None:
        target_area_size = (img_shape[1], img_shape[0]) if img_shape else (1920, 1080)
    
    target_area = target_area_size[0] * target_area_size[1]
    min_rectangle_area = target_area * 0.5  # 50% of target area
    
    best_combo = None
    best_score = 0
    
    # Try all combinations of 4 circles
    for circle_combo in combinations(size_similar_circles, 4):
        # Check if they have similar characteristics
        if img is not None:
            properties = []
            for x, y, r in circle_combo:
                is_valid, contrast_ratio, ring_width_ratio, ring_uniformity = validate_target_circle(img, x, y, r)
                if not is_valid:
                    break
                properties.append((contrast_ratio, ring_width_ratio, ring_uniformity))
            else:
                # All circles have valid properties, check similarity
                contrast_ratios = [prop[0] for prop in properties]
                ring_ratios = [prop[1] for prop in properties]
                uniformities = [prop[2] for prop in properties]
                
                # Check if properties are similar (within 20% tolerance)
                avg_contrast = sum(contrast_ratios) / len(contrast_ratios)
                avg_ring_ratio = sum(ring_ratios) / len(ring_ratios)
                avg_uniformity = sum(uniformities) / len(uniformities)
                
                contrast_similar = all(abs(c - avg_contrast) / avg_contrast <= 0.2 for c in contrast_ratios)
                ring_similar = all(abs(r - avg_ring_ratio) / avg_ring_ratio <= 0.2 for r in ring_ratios)
                uniformity_similar = all(abs(u - avg_uniformity) / avg_uniformity <= 0.2 for u in uniformities)
                
                if not (contrast_similar and ring_similar and uniformity_similar):
                    continue
        
        # Check if they form a rectangle with sufficient area
        is_valid, rectangle_area, _ = is_rectangle(circle_combo, non_uniformness_tolerance=0.4)
        
        if is_valid and rectangle_area >= min_rectangle_area:
            # Calculate score based on area ratio (higher is better)
            area_ratio = rectangle_area / target_area
            score = area_ratio
            
            if score > best_score:
                best_score = score
                best_combo = list(circle_combo)
    
    return best_combo if best_combo else []
    
    return best_combo if best_combo else []

def is_rectangle(circles, angle_tolerance=15, aspect_ratio_tolerance=0.3, non_uniformness_tolerance=0.4):
    """
    Check if 4 circles form a rectangle with additional criteria.
    
    Args:
        circles: List of 4 (x, y, r) tuples
        angle_tolerance: Tolerance for right angles (degrees)
        aspect_ratio_tolerance: Tolerance for rectangle aspect ratio
        non_uniformness_tolerance: Tolerance for non-uniform rectangle (0.0 = perfect, 1.0 = very distorted)
    
    Returns:
        tuple: (is_rectangle, rectangle_area, target_area_ratio)
    """
    if len(circles) != 4:
        return False, 0, 0
    
    # Sort circles by position (top-left, top-right, bottom-left, bottom-right)
    centers = [(x, y) for x, y, r in circles]
    centers.sort(key=lambda p: (p[1], p[0]))  # Sort by y, then x
    
    # Check if we have 2 rows and 2 columns
    y_coords = [y for x, y in centers]
    x_coords = [x for x, y in centers]
    
    # Group by y-coordinate (should have 2 groups)
    y_groups = []
    current_group = [centers[0]]
    for i in range(1, len(centers)):
        if abs(centers[i][1] - centers[i-1][1]) < 50:  # Same row
            current_group.append(centers[i])
        else:
            y_groups.append(current_group)
            current_group = [centers[i]]
    y_groups.append(current_group)
    
    if len(y_groups) != 2 or len(y_groups[0]) != 2 or len(y_groups[1]) != 2:
        return False, 0, 0
    
    # Sort each row by x-coordinate
    top_row = sorted(y_groups[0], key=lambda p: p[0])
    bottom_row = sorted(y_groups[1], key=lambda p: p[0])
    
    # Calculate rectangle properties
    width = abs(top_row[1][0] - top_row[0][0])
    height = abs(bottom_row[0][1] - top_row[0][1])
    
    # Check aspect ratio (should be reasonable for a rectangle/she shape)
    if width == 0 or height == 0:
        return False, 0, 0
    
    aspect_ratio = width / height
    if aspect_ratio < 0.3 or aspect_ratio > 3.0:  # More permissive for she shape
        return False, 0, 0
    
    # Calculate rectangle area
    rectangle_area = width * height
    
    # Check for non-uniformness (allowing for perspective distortion)
    # Calculate the four sides of the rectangle
    top_side = abs(top_row[1][0] - top_row[0][0])
    bottom_side = abs(bottom_row[1][0] - bottom_row[0][0])
    left_side = abs(bottom_row[0][1] - top_row[0][1])
    right_side = abs(bottom_row[1][1] - top_row[1][1])
    
    # Check if sides are reasonably similar (allowing for perspective)
    sides = [top_side, bottom_side, left_side, right_side]
    avg_side = sum(sides) / len(sides)
    
    # Calculate non-uniformness (how much sides differ from average)
    non_uniformness = sum(abs(side - avg_side) / avg_side for side in sides) / len(sides)
    
    if non_uniformness > non_uniformness_tolerance:
        return False, 0, 0
    
    # Check if corners are roughly at right angles
    # Calculate vectors between corners
    vectors = []
    for i in range(4):
        for j in range(i+1, 4):
            dx = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            vectors.append((dx, dy))
    
    # Check for perpendicular vectors (dot product ≈ 0)
    perpendicular_count = 0
    for i, v1 in enumerate(vectors):
        for j, v2 in enumerate(vectors[i+1:], i+1):
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            magnitude1 = np.sqrt(v1[0]**2 + v1[1]**2)
            magnitude2 = np.sqrt(v2[0]**2 + v2[1]**2)
            if magnitude1 > 0 and magnitude2 > 0:
                cos_angle = dot_product / (magnitude1 * magnitude2)
                angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                if abs(angle - 90) < angle_tolerance:
                    perpendicular_count += 1
    
    # Should have at least 1 perpendicular pair (more permissive for she shape)
    is_valid_rectangle = perpendicular_count >= 1
    
    return is_valid_rectangle, rectangle_area, rectangle_area

def could_form_rectangle_part(three_circles, img=None):
    """
    Check if 3 circles could form part of a rectangle pattern.
    Ensures circles have similar characteristics (size, white-to-black ratio).
    
    Args:
        three_circles: List of 3 (x, y, r) tuples
        img: Image for property validation (optional)
    
    Returns:
        bool: True if they could form part of a rectangle
    """
    if len(three_circles) != 3:
        return False
    
    centers = [(x, y) for x, y, r in three_circles]
    
    # Check if they are roughly in a triangular pattern (3 corners of a rectangle)
    x_coords = [x for x, y in centers]
    y_coords = [y for x, y in centers]
    
    # Calculate the spread of the 3 circles
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    
    # They should have some reasonable spread (not all clustered together)
    if x_range < 50 or y_range < 50:
        return False
    
    # Check if they are roughly the same size (stricter tolerance)
    radii = [r for x, y, r in three_circles]
    avg_radius = sum(radii) / 3
    for r in radii:
        if abs(r - avg_radius) / avg_radius > 0.3:  # 30% size tolerance (stricter)
            return False
    
    # If image is provided, check that all circles have similar properties
    if img is not None:
        properties = []
        for x, y, r in three_circles:
            # Use validate_target_circle instead of the removed validate_circle_properties
            is_valid, contrast_ratio, ring_width_ratio, ring_uniformity = validate_target_circle(img, x, y, r)
            if not is_valid:
                return False
            properties.append((contrast_ratio, ring_width_ratio))
        
        # Check that all circles have similar contrast ratios
        contrast_ratios = [prop[0] for prop in properties]
        avg_contrast = sum(contrast_ratios) / len(contrast_ratios)
        for ratio in contrast_ratios:
            if abs(ratio - avg_contrast) / avg_contrast > 0.4:  # 40% contrast tolerance
                return False
        
        # Check that all circles have similar ring width ratios
        ring_ratios = [prop[1] for prop in properties]
        avg_ring_ratio = sum(ring_ratios) / len(ring_ratios)
        for ratio in ring_ratios:
            if abs(ratio - avg_ring_ratio) / avg_ring_ratio > 0.4:  # 40% ring ratio tolerance
                return False
    
    return True


def find_fourth_circle_for_combo(three_circles, all_candidates, img=None):
    """
    Find the 4th circle for a specific combination of 3 circles.
    Calculates the exact 4th corner position and searches within 3x radius area.
    
    Args:
        three_circles: List of 3 (x, y, r) tuples
        all_candidates: List of all candidate circles
        img: Image for property validation (optional)
    
    Returns:
        tuple: (x, y, r) of the 4th circle, or None if not found
    """
    if len(three_circles) != 3:
        return None
    
    # Sort the 3 circles by position
    centers = [(x, y) for x, y, r in three_circles]
    centers.sort(key=lambda p: (p[1], p[0]))  # Sort by y, then x
    
    # Calculate the bounding box of the 3 circles
    x_coords = [x for x, y in centers]
    y_coords = [y for x, y in centers]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Calculate expected positions for all 4 corners
    expected_corners = [
        (min_x, min_y),  # top-left
        (max_x, min_y),  # top-right
        (min_x, max_y),  # bottom-left
        (max_x, max_y)   # bottom-right
    ]
    
    # Find which corner is missing
    found_corners = set()
    for x, y in centers:
        # Find the closest expected corner
        closest_corner = min(expected_corners, 
                           key=lambda corner: abs(corner[0] - x) + abs(corner[1] - y))
        found_corners.add(closest_corner)
    
    # Find the missing corner
    missing_corner = None
    for corner in expected_corners:
        if corner not in found_corners:
            missing_corner = corner
            break
    
    if missing_corner is None:
        return None
    
    # Verify that adding the 4th corner would form a rectangle
    test_centers = centers + [missing_corner]
    test_circles = [(x, y, sum(r for _, _, r in three_circles) // 3) for x, y in test_centers]
    
    # Check if this would form a rectangle
    if not is_rectangle(test_circles):
        print(f"    Rectangle validation failed for missing corner at {missing_corner}")
        return None
    
    # Search for a circle near the missing corner
    search_x, search_y = missing_corner
    
    # Calculate the average radius of the 3 circles
    avg_radius = sum(r for x, y, r in three_circles) / 3
    
    # Define search area (3 times larger than the circle radius)
    search_radius = avg_radius * 3
    
    print(f"    Looking for 4th circle near position ({search_x}, {search_y})")
    print(f"    Search radius: {search_radius:.1f} pixels")
    print(f"    Average radius of 3 circles: {avg_radius:.1f}")
    
    best_circle = None
    best_score = float('inf')
    candidates_in_range = 0
    
    for x, y, r in all_candidates:
        # Skip if this circle is already in the 3 circles
        if any(abs(x-cx) < 10 and abs(y-cy) < 10 for cx, cy, cr in three_circles):
            continue
        
        # Calculate distance from expected position
        distance = np.sqrt((x - search_x)**2 + (y - search_y)**2)
        
        # Only consider circles within the search area
        if distance > search_radius:
            continue
        
        candidates_in_range += 1
        
        # Calculate size similarity score (more relaxed)
        size_diff = abs(r - avg_radius) / avg_radius
        
        # Use relaxed validation for the 4th circle
        if img is not None:
            is_valid, contrast_ratio, ring_width_ratio = validate_circle_properties_relaxed(img, x, y, r)
            if not is_valid:
                print(f"      Candidate at ({x}, {y}) failed relaxed validation")
                continue
        
        # Combined score (lower is better) - prioritize distance over size
        score = distance + size_diff * 20  # Reduced weight for size difference
        
        # More relaxed size tolerance
        if size_diff < 0.6:  # 60% size tolerance (more relaxed)
            if score < best_score:
                best_score = score
                best_circle = (x, y, r)
                print(f"      Found better candidate at ({x}, {y}) with score {score:.1f}")
        else:
            print(f"      Candidate at ({x}, {y}) failed size check (size_diff: {size_diff:.2f})")
    
    print(f"    Found {candidates_in_range} candidates in search range")
    if best_circle is None:
        print(f"    No valid 4th circle found")
    else:
        print(f"    Best 4th circle found at ({best_circle[0]}, {best_circle[1]}) with score {best_score:.1f}")
    
    return best_circle


def detect_circles(image_path, output_dir=None, debug=False):
    """
    Detect 4 black circles with white centers that form a rectangle in an image and generate YOLO format annotations.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save annotation files (if None, saves in same dir as image)
        debug (bool): Whether to show debug visualization
    
    Returns:
        bool: True if successful (found 4 circles forming rectangle), False otherwise
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False
    
    height, width = img.shape[:2]
    
    # First, find the bright rectangular target area
    target_area = find_bright_rectangular_area(img, debug)
    
    if target_area is None:
        print(f"Could not find target area in {image_path}")
        # if debug:
        #     # Show the image and its grayscale version for debugging
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     
        #     # Resize images to fit screen (max 1200x800)
        #     height, width = img.shape[:2]
        #     max_width, max_height = 1200, 800
        #     
        #     if width > max_width or height > max_height:
        #         scale = min(max_width / width, max_height / height)
        #         new_width = int(width * scale)
        #         new_height = int(height * scale)
        #         img_resized = cv2.resize(img, (new_width, new_height))
        #         gray_resized = cv2.resize(gray, (new_width, new_height))
        #     else:
        #         img_resized = img
        #         gray_resized = gray
        #     
        #     cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
        #     cv2.namedWindow('Grayscale', cv2.WINDOW_NORMAL)
        #     cv2.imshow('Original Image', img_resized)
        #     cv2.imshow('Grayscale', gray_resized)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        return False
    
    x, y, w, h = target_area
    print(f"Found target area at ({x}, {y}) with size ({w}, {h})")
    
    # if debug:
    #     print(f"Target area covers {w*h} pixels out of {img.shape[0]*img.shape[1]} total pixels")
    #     print(f"Target area is {w*h/(img.shape[0]*img.shape[1])*100:.1f}% of the image")
    
    # Crop the image to the target area
    target_img = img[y:y+h, x:x+w]
    target_height, target_width = target_img.shape[:2]
    
    # if debug:
    #     # Show the target area
    #     debug_img = img.copy()
    #     cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    #     
    #     # Resize image to fit screen (max 1200x800)
    #     height, width = debug_img.shape[:2]
    #     max_width, max_height = 1200, 800
    #     
    #     if width > max_width or height > max_height:
    #         scale = min(max_width / width, max_height / height)
    #         new_width = int(width * scale)
    #         new_height = int(height * scale)
    #         debug_img = cv2.resize(debug_img, (new_width, new_height))
    #     
    #     cv2.namedWindow('Target Area', cv2.WINDOW_NORMAL)
    #     cv2.imshow('Target Area', debug_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    # Convert target area to grayscale
    gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    # if debug:
    #     # Show the grayscale image used for circle detection
    #     height, width = gray.shape[:2]
    #     max_width, max_height = 1200, 800
    #     if width > max_width or height > max_height:
    #         scale = min(max_width / width, max_height / height)
    #         new_width = int(width * scale)
    #         new_height = int(height * scale)
    #         gray_resized = cv2.resize(gray, (new_width, new_height))
    #     else:
    #         gray_resized = gray
        
    #     cv2.namedWindow('Grayscale Target Area', cv2.WINDOW_NORMAL)
    #     cv2.imshow('Grayscale Target Area', gray_resized)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    # if debug:
    #     print(f"Target area size: {w}x{h}")
    #     print(f"Circle detection parameters:")
    #     print(f"  minDist: {min(w, h) // 6}")
    #     print(f"  minRadius: {min(w, h) // 12}")
    #     print(f"  maxRadius: {min(w, h) // 3}")
    
    # NEW CIRCLE DETECTION APPROACH
    # Use multiple detection strategies with different parameters
    
    detected_circles = []
    
    # Strategy 1: Detect large circles with conservative parameters
    circles1 = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(w, h) // 4,  # Balanced minimum distance
        param1=40,   # Balanced edge threshold
        param2=25,   # Balanced accumulator threshold
        minRadius=min(w, h) // 30,  # Much smaller minimum radius
        maxRadius=min(w, h) // 8    # Much smaller maximum radius
    )
    
    if circles1 is not None:
        circles1 = np.round(circles1[0, :]).astype("int")
        detected_circles.extend(circles1)
    
    # Strategy 2: Detect medium circles with balanced parameters
    circles2 = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(w, h) // 6,  # Smaller distance for better detection
        param1=35,   # More sensitive edge detection
        param2=20,   # Less strict accumulator
        minRadius=min(w, h) // 40,  # Much smaller radius range
        maxRadius=min(w, h) // 12   # Much smaller maximum
    )
    
    if circles2 is not None:
        circles2 = np.round(circles2[0, :]).astype("int")
        detected_circles.extend(circles2)
    
    # Strategy 3: Detect corner circles specifically (most important)
    # circles3 = cv2.HoughCircles(
    #     blurred,
    #     cv2.HOUGH_GRADIENT,
    #     dp=1,
    #     minDist=min(w, h) // 8,  # Small distance for corners
    #     param1=30,   # Sensitive edge detection
    #     param2=18,   # Permissive accumulator
    #     minRadius=min(w, h) // 15,  # Small radius for corner circles
    #     maxRadius=min(w, h) // 5    # Not too large
    # )
    
    # if circles3 is not None:
    #     circles3 = np.round(circles3[0, :]).astype("int")
    #     detected_circles.extend(circles3)
    
    if not detected_circles:
        print(f"No circles detected in target area of {image_path}")
        return False
    
    # Remove duplicates (circles that are too close to each other)
    unique_circles = []
    for circle in detected_circles:
        cx, cy, cr = circle
        is_duplicate = False
        for existing in unique_circles:
            ex, ey, er = existing
            distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)
            if distance < min(cr, er) * 0.5:  # If centers are too close
                is_duplicate = True
                break
        if not is_duplicate:
            unique_circles.append(circle)
    
    if debug:
        print(f"Found {len(unique_circles)} unique circles")
    
    # Filter circles based on target-specific criteria
    candidate_circles = []
    
    if debug:
        # Create debug image to show all detected circles
        debug_img = img.copy()
    
    for (cx, cy, r) in unique_circles:
        # Convert coordinates back to original image space
        orig_x = x + cx
        orig_y = y + cy
        
        # Ensure circle is within target area bounds (more permissive for corner circles)
        margin = 5  # Reduced margin to allow corner circles
        if cx - r < margin or cy - r < margin or cx + r >= target_width - margin or cy + r >= target_height - margin:
            continue
        
        # Skip circles that are too close to the center of the image
        # center_x = target_width // 2
        # center_y = target_height // 2
        # distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        # 
        # # Skip if circle is within 20% of image dimensions from center (more permissive)
        # center_threshold = min(target_width, target_height) * 0.2
        # if distance_from_center < center_threshold:
        #     continue
        
        # NEW VALIDATION: Check if this looks like a target circle
        is_valid, contrast_ratio, ring_width_ratio, ring_uniformity = validate_target_circle(img, orig_x, orig_y, r)
        
        # Debug output
        if debug:
            print(f"Circle at ({orig_x},{orig_y}) r={r}: contrast_ratio={contrast_ratio:.2f}, ring_width_ratio={ring_width_ratio:.2f}, uniformity={ring_uniformity:.1f}, valid={is_valid}")
            # Draw red circle around all detected circles
            cv2.circle(debug_img, (orig_x, orig_y), r, (0, 0, 255), 2)
            # Add text with validation info
            text = f"R:{r} C:{contrast_ratio:.2f} RW:{ring_width_ratio:.1f} U:{ring_uniformity:.0f} V:{is_valid}"
            cv2.putText(debug_img, text, (orig_x-r, orig_y-r-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # TEMPORARILY: Accept all circles for testing
        candidate_circles.append((orig_x, orig_y, r))
    
    if debug and len(unique_circles) > 0:
        # Show debug image with all detected circles
        height, width = debug_img.shape[:2]
        max_width, max_height = 1200, 800
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            debug_img = cv2.resize(debug_img, (new_width, new_height))
        
        cv2.namedWindow('All Detected Circles', cv2.WINDOW_NORMAL)
        cv2.imshow('All Detected Circles', debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

    
    print(f"Found {len(candidate_circles)} candidate circles in target area of {image_path}")
    
    # Find 4 circles that form a rectangle
    valid_circles = find_rectangle_circles(candidate_circles, tolerance=0.3, img_shape=img.shape, 
                                         target_area_size=(w, h), img=img)
    
    # If we don't find 4 circles, try a more targeted approach
    if len(valid_circles) < 4 and len(candidate_circles) >= 4:
        print(f"Trying targeted approach for 4-circle detection...")
        
        # Look for circles that are likely to be the target circles
        # Target circles should be roughly in a rectangle pattern around the center
        # Note: find_target_circles function removed as it was not effectively used
        
        print(f"Targeted approach not implemented - continuing with combination search")
    
    # If we still don't have 4 circles, try different combinations of 3 circles
    if len(valid_circles) < 4 and len(candidate_circles) >= 4:
        print(f"Trying different combinations of 3 circles to find the 4th...")
        print(f"Total candidate circles: {len(candidate_circles)}")
        print(f"First 10 candidate circles: {[(x,y,r) for x,y,r in candidate_circles[:10]]}")
        
        # Try all combinations of 3 circles from candidates
        for i, three_circles_combo in enumerate(combinations(candidate_circles, 3)):
            # Check if these 3 circles could form part of a rectangle
            if could_form_rectangle_part(three_circles_combo, img):
                print(f"  Testing combination {i+1}: 3 circles at positions {[(x,y) for x,y,r in three_circles_combo]}")
                
                # Try to find the 4th circle for this combination
                fourth_circle = find_fourth_circle_for_combo(three_circles_combo, candidate_circles, img)
                
                if fourth_circle is not None:
                    valid_circles = list(three_circles_combo) + [fourth_circle]
                    print(f"  Found 4th circle at ({fourth_circle[0]}, {fourth_circle[1]})")
                    print(f"Found 4 circles using combination approach")
                    break
                else:
                    print(f"  No 4th circle found for this combination")
    
    # Debug visualization - show when debug is enabled and fewer than 10 candidates
    if debug and len(candidate_circles) < 20:
        debug_img = img.copy()
        
        # Draw all candidate circles in red with thicker lines
        for (x, y, r) in candidate_circles:
            cv2.circle(debug_img, (x, y), r, (0, 0, 255), 3)  # Red for candidates, thicker line
            cv2.circle(debug_img, (x, y), 3, (255, 255, 255), -1)  # White center dot
        
        # Draw the selected circles in green (if any)
        for (x, y, r) in valid_circles:
            cv2.circle(debug_img, (x, y), r, (0, 255, 0), 4)  # Green for selected, thicker
            cv2.circle(debug_img, (x, y), 4, (0, 0, 255), -1)  # Red center dot
        
        # Draw rectangle connecting the 4 circles (if found)
        if len(valid_circles) == 4:
            centers = [(x, y) for x, y, r in valid_circles]
            # Sort by position to draw rectangle
            centers.sort(key=lambda p: (p[1], p[0]))  # Sort by y, then x
            
            # Draw rectangle lines
            for i in range(4):
                for j in range(i+1, 4):
                    cv2.line(debug_img, centers[i], centers[j], (255, 255, 0), 3)  # Yellow lines, thicker
        
        # Add text overlay with detection info
        info_text = f"Candidates: {len(candidate_circles)}, Selected: {len(valid_circles)}"
        cv2.putText(debug_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Add status text
        if len(valid_circles) == 4:
            status_text = "SUCCESS: Found 4-circle rectangle"
            color = (0, 255, 0)  # Green
        elif len(candidate_circles) >= 4:
            status_text = "FAILED: Found candidates but no rectangle pattern"
            color = (0, 165, 255)  # Orange
        elif len(candidate_circles) > 0:
            status_text = f"FAILED: Only {len(candidate_circles)} valid circles found"
            color = (0, 0, 255)  # Red
        else:
            status_text = "FAILED: No valid circles detected"
            color = (0, 0, 255)  # Red
        
        cv2.putText(debug_img, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Resize image to fit screen if it's too large
        max_height = 800
        max_width = 1200
        h, w = debug_img.shape[:2]
        
        if h > max_height or w > max_width:
            # Calculate scaling factor
            scale_h = max_height / h
            scale_w = max_width / w
            scale = min(scale_h, scale_w)
            
            new_width = int(w * scale)
            new_height = int(h * scale)
            debug_img = cv2.resize(debug_img, (new_width, new_height))
        
        window_name = f"Circle Detection - {Path(image_path).name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, debug_img.shape[1], debug_img.shape[0])
        cv2.imshow(window_name, debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif debug and len(candidate_circles) >= 10:
        print(f"Skipping debug display for {Path(image_path).name} - too many candidates ({len(candidate_circles)})")
    
    if len(valid_circles) == 0:
        print(f"No rectangle pattern found with 4 circles. Skipping {image_path}")
        return False
    
    print(f"Found {len(valid_circles)} valid circles forming rectangle in {image_path}")
    
    # Check if we have exactly 4 circles forming a rectangle
    if len(valid_circles) != 4:
        print(f"Expected 4 circles forming rectangle, found {len(valid_circles)}. Skipping {image_path}")
        return False
    
    # Limit to maximum 4 circles (already handled in sorting above)
    if len(valid_circles) > 4:
        valid_circles = valid_circles[:4]
    
    # Generate YOLO format annotations
    annotations = []
    for (x, y, r) in valid_circles:
        # Convert to YOLO format (normalized coordinates)
        # YOLO format: class_id center_x center_y width height
        center_x = x / width
        center_y = y / height
        bbox_width = (2 * r) / width
        bbox_height = (2 * r) / height
        
        # Class ID is 0 as specified
        annotation = f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}"
        annotations.append(annotation)
    
    # Save annotation file
    image_name = Path(image_path).stem
    if output_dir:
        output_path = Path(output_dir) / f"{image_name}.txt"
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_path = Path(image_path).parent / f"{image_name}.txt"
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(annotations))
    
    print(f"Saved annotations to {output_path}")
    
    return True

def process_directory(input_dir="data/raw/", output_dir=None, debug=False):
    """
    Process all JPG images in the input directory.
    
    Args:
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save annotations (if None, saves alongside images)
        debug (bool): Whether to show debug visualizations
    
    Returns:
        tuple: (successful_count, total_count)
    """
    # Ensure input directory exists
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return 0, 0
    
    # Find all JPG files
    jpg_patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    image_files = []
    for pattern in jpg_patterns:
        image_files.extend(input_path.glob(pattern))
    
    if not image_files:
        print(f"No JPG files found in '{input_dir}'")
        return 0, 0
    
    print(f"Found {len(image_files)} image files to process")
    
    successful = 0
    total = len(image_files)
    
    for image_file in image_files:
        print(f"\nProcessing: {image_file}")
        if detect_circles(str(image_file), output_dir, debug):
            successful += 1
        
        # If debug mode, wait for user input between images
        if debug and len(image_files) > 1:
            input("Press Enter to continue to next image...")
    
    return successful, total

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Detect circles in images and generate YOLO annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python circle_detector.py
    python circle_detector.py --input_dir images/ --output_dir labels/
    python circle_detector.py --debug
        """
    )
    
    parser.add_argument(
        "--input_dir", 
        default="data/raw/",
        help="Directory containing input JPG images (default: data/raw/)"
    )
    
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save annotation files (default: same as input images)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug visualizations"
    )
    
    args = parser.parse_args()
    
    print("Circle Detection and YOLO Annotation Generator")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir or 'Same as input'}")
    print(f"Debug mode: {args.debug}")
    print()
    
    # Check if virtual environment is active
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Virtual environment detected")
    else:
        print("⚠ Warning: No virtual environment detected. Consider using 'python -m venv venv' and 'source venv/bin/activate'")
    
    try:
        successful, total = process_directory(args.input_dir, args.output_dir, args.debug)
        
        print("\n" + "=" * 50)
        print("Processing Complete!")
        print(f"Successfully processed: {successful}/{total} images")
        
        if successful < total:
            print(f"Skipped: {total - successful} images (no 4-circle rectangle pattern found)")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()