# Image Stitching & Panorama Construction

## Overview

This project implements a robust image stitching pipeline to create seamless panoramas from multiple overlapping images. It utilizes advanced computer vision techniques to accurately detect features, align images, and blend them into a single cohesive wide-angle view, handling exposure differences and seamless transitions.

## Features

- **Feature Detection**: Utilizes **SIFT (Scale-Invariant Feature Transform)** to identify robust keypoints that are invariant to scale and rotation.
- **Feature Matching**: Implements **FLANN-based matching** with **Lowe's Ratio Test** to ensure high-quality feature correspondences.
- **Robust Alignment**: Computes **Homography** matrices using **RANSAC** (Random Sample Consensus) to accurately map images while ignoring outliers.
- **Minimal Distortion**: Uses a **Centered Anchoring** strategy (aligning Left and Right images to the Center) to reduce perspective distortion.
- **Advanced Blending**:
  - **Exposure Compensation**: Automatically adjusts gain to match brightness levels across overlapping regions.
  - **Seamless Stitching**: Application of **Distance Transform (Feathering)** to weight pixels based on distance from edges, eliminating visible seams.
- **Auto-Cropping**: Automatically detects and crops the black borders from the final stitched panorama.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`, `opencv-contrib-python`)
- NumPy

## Setup & Usage

1.  **Install Dependencies**:

    ```bash
    pip install opencv-python opencv-contrib-python numpy
    ```

2.  **Prepare Images**:
    Place your source images in the `code/` directory. The script expects the following filenames (supports `.jpg`, `.jpeg`):
    - `left.jpg`
    - `center.jpg`
    - `right.jpg`

3.  **Run the Script**:
    Navigate to the project root and run:
    ```bash
    python code/code.py
    ```

## Output

- **final_panorama.jpg**: The fully stitched and blended panorama (saved in `code/`).
- **matches_LC.jpg**: A visualization of the feature matches between Left and Center images.
- **Console Stats**: The script outputs detailed metrics including keypoint counts, match quality, and RANSAC inlier percentages for evaluation.

## Project Structure

- `code/`: Source code (`code.py`, `naive_code.py`) and input/output images.
- `report/`: Project report and documentation.
- `figures/`: Supplementary figures for the report.
