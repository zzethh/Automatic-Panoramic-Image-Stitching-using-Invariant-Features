import cv2
import numpy as np
import os
import sys

# Global stats to print for the report
stats = {
    "kp_L": 0,
    "kp_C": 0,
    "kp_R": 0,
    "matches_LC": 0,
    "inliers_LC": 0,
    "matches_RC": 0,
    "inliers_RC": 0,
}


def load_images_robust():
    # I'm just checking for both jpg and jpeg extensions so the script doesn't crash if I name them differently.
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def find_file(name):
        for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
            path = os.path.join(script_dir, name + ext)
            if os.path.exists(path):
                return cv2.imread(path), path
        return None, None

    img_L, path_L = find_file("left")
    img_C, path_C = find_file("center")
    img_R, path_R = find_file("right")

    if img_L is None:
        print(f"ERROR: Could not find 'left.jpg' or 'left.jpeg' in {script_dir}")
    if img_C is None:
        print(f"ERROR: Could not find 'center.jpg' or 'center.jpeg' in {script_dir}")
    if img_R is None:
        print(f"ERROR: Could not find 'right.jpg' or 'right.jpeg' in {script_dir}")

    if img_L is None or img_C is None or img_R is None:
        print("Stopping execution. Please check your file names.")
        sys.exit(1)

    print(
        f"Loaded: {os.path.basename(path_L)}, {os.path.basename(path_C)}, {os.path.basename(path_R)}"
    )
    return img_L, img_C, img_R


def detect_and_match(img1, img2, tag=""):
    # I'm using SIFT (Scale-Invariant Feature Transform) here.
    # It's good because it finds keypoints that don't change even if we zoom in or rotate the camera.
    sift = cv2.SIFT_create()

    # detectAndCompute finds the keypoints and describing vectors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Store keypoint counts
    global stats
    if tag == "LC":
        stats["kp_L"] = len(kp1)
        stats["kp_C"] = len(kp2)  # C is img2
    elif tag == "RC":
        stats["kp_R"] = len(kp1)  # R is img1
        # C is already counted

    # Using FLANN matcher because it's faster than BFMatcher (Brute Force) for large datasets.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Getting the top 2 matches for each point so I can run Lowe's Ratio Test
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's Ratio Test:
    # If the first match is noticeably better (distance < 0.7) than the second match, I keep it.
    # This filters out ambiguous matches (like repeated patterns in windows/bricks).
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if tag == "LC":
        stats["matches_LC"] = len(good)
    if tag == "RC":
        stats["matches_RC"] = len(good)

    return kp1, kp2, good


def compute_homography(img_src, img_dst, tag=""):
    # This function figures out the perspective transform (Homography)
    # that maps the source image onto the destination image plane.
    kp_src, kp_dst, matches = detect_and_match(img_src, img_dst, tag)

    # Need at least 4 points to solve the homography matrix equations
    if len(matches) < 4:
        print("Not enough matches found!")
        return None, None, None, None

    # Extract the (x, y) coordinates of the matching points
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Using RANSAC to ignore outliers.
    # It randomly tries subsets of points to find the best fit model.
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Count inliers
    inliers = np.sum(mask)
    if tag == "LC":
        stats["inliers_LC"] = int(inliers)
    if tag == "RC":
        stats["inliers_RC"] = int(inliers)

    return H, kp_src, kp_dst, matches


def stitch_images_centered(img_L, img_C, img_R):
    # Instead of stitching L->C and then (L+C)->R, I'm anchoring everything to the Center image.
    # This prevents the Left image from getting warped twice and looking super distorted.
    print("Computing Alignment: Left -> Center...")
    H_L_to_C, kpL, kpC, matchLC = compute_homography(img_L, img_C, tag="LC")

    # Save a visualization of matches for the report
    if kpL is not None:
        # Draw top 50 matches
        vis_match = cv2.drawMatches(
            img_L,
            kpL,
            img_C,
            kpC,
            matchLC[:50],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "matches_LC.jpg"),
            vis_match,
        )

    print("Computing Alignment: Right -> Center...")
    H_R_to_C, _, _, _ = compute_homography(img_R, img_C, tag="RC")

    if H_L_to_C is None or H_R_to_C is None:
        print("Could not compute homographies. Aborting.")
        return img_C

    # I need to figure out how big the new canvas should be.
    # I'll do this by transforming the corners of L and R into C's coordinate system.
    h_c, w_c = img_C.shape[:2]
    h_l, w_l = img_L.shape[:2]
    h_r, w_r = img_R.shape[:2]

    corners_C = np.float32([[0, 0], [0, h_c], [w_c, h_c], [w_c, 0]]).reshape(-1, 1, 2)
    corners_L = np.float32([[0, 0], [0, h_l], [w_l, h_l], [w_l, 0]]).reshape(-1, 1, 2)
    corners_R = np.float32([[0, 0], [0, h_r], [w_r, h_r], [w_r, 0]]).reshape(-1, 1, 2)

    corners_L_trans = cv2.perspectiveTransform(corners_L, H_L_to_C)
    corners_R_trans = cv2.perspectiveTransform(corners_R, H_R_to_C)

    # Using the min/max x and y from all transformed corners to get the bounding box
    all_points = np.concatenate((corners_C, corners_L_trans, corners_R_trans), axis=0)

    [xmin, ymin] = np.int32(all_points.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_points.max(axis=0).ravel() + 0.5)

    # Since xmin/ymin might be negative (to the left/top of Center image),
    # I need a translation matrix to shift everything into positive coordinates.
    translation_dist = [-xmin, -ymin]
    H_translation = np.array(
        [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]],
        dtype=np.float32,
    )

    # Combine the homographies with the translation
    final_H_L = H_translation.dot(H_L_to_C)
    final_H_R = H_translation.dot(H_R_to_C)

    output_shape = (xmax - xmin, ymax - ymin)

    print("Warping Left image...")
    warped_L = cv2.warpPerspective(img_L, final_H_L, output_shape)

    print("Warping Right image...")
    warped_R = cv2.warpPerspective(img_R, final_H_R, output_shape)

    print("Placing Center image...")
    # Center image just needs the translation shift, no perspective warp
    warped_C = cv2.warpPerspective(img_C, H_translation, output_shape)

    # I'm checking for seamless blending now.
    # ---------------------------------------------------------
    # IMPROVEMENT: Seam Removal & Exposure Compensation
    # ---------------------------------------------------------

    # 1. Create masks for where we actually have image data (pixels > 0)
    #    I'm making sure to check all channels so I don't miss dark pixels that are valid.
    mask_L = (warped_L.sum(axis=2) > 0).astype(np.float32)
    mask_C = (warped_C.sum(axis=2) > 0).astype(np.float32)
    mask_R = (warped_R.sum(axis=2) > 0).astype(np.float32)

    # 2. Exposure Compensation (Simple Gain Adjustment)
    #    Sometimes one image is brighter than the other. I'll match L and R to C's brightness
    #    by looking at the overlapping regions.

    print("Applying Exposure Compensation...")
    # Overlap between Left and Center
    overlap_LC = (mask_L * mask_C) > 0
    if np.sum(overlap_LC) > 0:
        # Calculate mean brightness in the overlap
        mean_L = np.mean(warped_L[overlap_LC])
        mean_C_sub = np.mean(warped_C[overlap_LC])
        gain_L = mean_C_sub / (mean_L + 1e-5)  # Avoid divide by zero
        print(f"  -> Compensating L->C: Gain = {gain_L:.2f}")

        # Apply gain but clamp to 255
        warped_L = np.clip(warped_L * gain_L, 0, 255).astype(np.uint8)

    # Overlap between Right and Center
    overlap_RC = (mask_R * mask_C) > 0
    if np.sum(overlap_RC) > 0:
        mean_R = np.mean(warped_R[overlap_RC])
        mean_C_sub = np.mean(warped_C[overlap_RC])
        gain_R = mean_C_sub / (mean_R + 1e-5)
        print(f"  -> Compensating R->C: Gain = {gain_R:.2f}")

        # Apply gain but clamp to 255
        warped_R = np.clip(warped_R * gain_R, 0, 255).astype(np.uint8)

    # 3. Linear Blending using Distance Transform (Feathering)
    #    Instead of just taking the max pixel (which leaves seams), I'll weight pixels
    #    based on how far they are from the edge. Center pixels get more weight.
    print("Blending images using Distance Transform...")

    # Compute Distance Transform (this gives the distance to the nearest zero pixel)
    dist_L = cv2.distanceTransform(mask_L.astype(np.uint8), cv2.DIST_L2, 5)
    dist_C = cv2.distanceTransform(mask_C.astype(np.uint8), cv2.DIST_L2, 5)
    dist_R = cv2.distanceTransform(mask_R.astype(np.uint8), cv2.DIST_L2, 5)

    # We need to broadcast weights to 3 channels to multiply with RGB images
    w_L = dist_L[..., np.newaxis]
    w_C = dist_C[..., np.newaxis]
    w_R = dist_R[..., np.newaxis]

    # Weighted Sum: (Image * Weight) / (Sum of Weights)
    numerator = (
        warped_L.astype(np.float32) * w_L
        + warped_C.astype(np.float32) * w_C
        + warped_R.astype(np.float32) * w_R
    )

    denominator = w_L + w_C + w_R

    # Compute final result
    # Where denominator is 0 (no image), the result doesn't matter.
    # I'll add epsilon to the denominator to avoid div/0 errors.
    final_pano = numerator / (denominator + 1e-5)

    # Convert back to uint8
    final_pano = np.clip(final_pano, 0, 255).astype(np.uint8)

    return final_pano


def crop_black_borders(img):
    # This just gets rid of the extra black space around the panorama.
    # It finds the largest object (the pano) and crops to its bounding box.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find constraint bounding box
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return img[y : y + h, x : x + w]
    return img


# Main execution block
if __name__ == "__main__":
    print("Starting Panorama Stitching...")
    img_L, img_C, img_R = load_images_robust()

    print("Stitching images using Center anchor...")
    final_pano = stitch_images_centered(img_L, img_C, img_R)

    print("Cropping borders...")
    final_pano_cropped = crop_black_borders(final_pano)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "final_panorama.jpg"
    )
    cv2.imwrite(output_path, final_pano_cropped)
    print(f"Success! Saved to {output_path}")

    # Print Report Stats
    print("\n" + "=" * 50)
    print("STITCHING COMPLETE - Experimental Results Summary")
    print("=" * 50)
    print(f"\nFeature Detection:")
    print(f"  • Left image:   {stats['kp_L']:,} keypoints")
    print(f"  • Center image: {stats['kp_C']:,} keypoints")
    print(f"  • Right image:  {stats['kp_R']:,} keypoints")
    print(f"\nFeature Matching & RANSAC:")
    print(
        f"  • Left → Center:  {stats['matches_LC']:,} matches → {stats['inliers_LC']:,} inliers ({stats['inliers_LC'] / stats['matches_LC'] * 100:.1f}%)"
    )
    print(
        f"  • Right → Center: {stats['matches_RC']:,} matches → {stats['inliers_RC']:,} inliers ({stats['inliers_RC'] / stats['matches_RC'] * 100:.1f}%)"
    )
    print("\n" + "=" * 50 + "\n")
