import cv2
import numpy as np
import os
import sys


# Helper function to load images with different extensions
def load_image_robust(directory, filename_no_ext):
    for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        full_path = os.path.join(directory, filename_no_ext + ext)
        if os.path.exists(full_path):
            return cv2.imread(full_path)
    return None


script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the three input images
img_L = load_image_robust(script_dir, "left")
img_C = load_image_robust(script_dir, "center")
img_R = load_image_robust(script_dir, "right")

if img_L is None or img_C is None or img_R is None:
    print(
        "Error: Cannot find images. Make sure they are named 'left', 'center', 'right' and are in the same folder as this script."
    )
    sys.exit(1)

# Resize images to the same height for concatenation
h_min = min(img_L.shape[0], img_C.shape[0], img_R.shape[0])


def resize_to_height(img, height):
    ratio = height / img.shape[0]
    width = int(img.shape[1] * ratio)
    return cv2.resize(img, (width, height))


img_L_res = resize_to_height(img_L, h_min)
img_C_res = resize_to_height(img_C, h_min)
img_R_res = resize_to_height(img_R, h_min)

print(f"Image dimensions after resize:")
print(f"  Left:   {img_L_res.shape[1]} x {img_L_res.shape[0]}")
print(f"  Center: {img_C_res.shape[1]} x {img_C_res.shape[0]}")
print(f"  Right:  {img_R_res.shape[1]} x {img_R_res.shape[0]}")

# Naive stitching with 35% overlap
# This demonstrates why simple overlay fails compared to proper homography based stitching
overlap_percentage = 0.35

overlap_width_LC = int(img_C_res.shape[1] * overlap_percentage)
overlap_width_CR = int(img_R_res.shape[1] * overlap_percentage)

print(f"\nOverlap regions:")
print(f"  Left-Center overlap: {overlap_width_LC} pixels")
print(f"  Center-Right overlap: {overlap_width_CR} pixels")

# Create canvas to hold the stitched result
total_width = (
    img_L_res.shape[1]
    + img_C_res.shape[1]
    - overlap_width_LC
    + img_R_res.shape[1]
    - overlap_width_CR
)
canvas = np.zeros((h_min, total_width, 3), dtype=np.uint8)

print(f"\nCanvas dimensions: {total_width} x {h_min}")

x_offset = 0

# Place left image
canvas[0:h_min, x_offset : x_offset + img_L_res.shape[1]] = img_L_res
x_offset += img_L_res.shape[1]

# Overlay center image with left image
x_offset -= overlap_width_LC

left_overlap_region = img_L_res[:, -overlap_width_LC:]
center_overlap_region = img_C_res[:, :overlap_width_LC]

# Simple averaging in overlap region creates visible ghosting
overlap_blend_LC = cv2.addWeighted(
    left_overlap_region, 0.5, center_overlap_region, 0.5, 0
)

canvas[0:h_min, x_offset : x_offset + overlap_width_LC] = overlap_blend_LC

# Place remaining part of center image
x_offset += overlap_width_LC
canvas[0:h_min, x_offset : x_offset + (img_C_res.shape[1] - overlap_width_LC)] = (
    img_C_res[:, overlap_width_LC:]
)
x_offset += img_C_res.shape[1] - overlap_width_LC

# Overlay right image with center image
x_offset -= overlap_width_CR

center_overlap_region_R = img_C_res[:, -overlap_width_CR:]
right_overlap_region = img_R_res[:, :overlap_width_CR]

overlap_blend_CR = cv2.addWeighted(
    center_overlap_region_R, 0.5, right_overlap_region, 0.5, 0
)

canvas[0:h_min, x_offset : x_offset + overlap_width_CR] = overlap_blend_CR

# Place remaining part of right image
x_offset += overlap_width_CR
canvas[0:h_min, x_offset : x_offset + (img_R_res.shape[1] - overlap_width_CR)] = (
    img_R_res[:, overlap_width_CR:]
)

# Add visual markers to highlight overlap regions
overlay_marker = canvas.copy()
lc_overlap_start = img_L_res.shape[1] - overlap_width_LC
lc_overlap_end = img_L_res.shape[1]

# Red tint for left-center overlap
cv2.rectangle(
    overlay_marker,
    (lc_overlap_start, 0),
    (lc_overlap_end, h_min),
    (0, 0, 255),
    -1,
)

# Blue tint for center-right overlap
cr_overlap_start = (
    img_L_res.shape[1] + img_C_res.shape[1] - overlap_width_LC - overlap_width_CR
)
cr_overlap_end = cr_overlap_start + overlap_width_CR

cv2.rectangle(
    overlay_marker,
    (cr_overlap_start, 0),
    (cr_overlap_end, h_min),
    (255, 0, 0),
    -1,
)

# Blend markers with canvas
canvas_with_markers = cv2.addWeighted(canvas, 0.9, overlay_marker, 0.1, 0)

# Add text labels
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
font_thickness = 3
text_color = (0, 255, 255)

cv2.putText(
    canvas_with_markers,
    "L-C Overlap",
    (lc_overlap_start + 10, 50),
    font,
    font_scale,
    text_color,
    font_thickness,
    cv2.LINE_AA,
)

cv2.putText(
    canvas_with_markers,
    "C-R Overlap",
    (cr_overlap_start + 10, 50),
    font,
    font_scale,
    text_color,
    font_thickness,
    cv2.LINE_AA,
)

# Save both versions
output_path_basic = os.path.join(script_dir, "naive_stitch.jpg")
cv2.imwrite(output_path_basic, canvas)
print(f"\nSaved: {output_path_basic}")

output_path_annotated = os.path.join(script_dir, "naive_stitch_annotated.jpg")
cv2.imwrite(output_path_annotated, canvas_with_markers)
print(f"Saved: {output_path_annotated}")

print("\nNaive stitching complete.")
