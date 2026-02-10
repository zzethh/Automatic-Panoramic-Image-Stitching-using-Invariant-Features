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
    # Check for both .jpg and .jpeg extensions to handle file naming variations.
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
    # Initialize SIFT (Scale-Invariant Feature Transform).
    # SIFT provides keypoints invariant to scale and rotation, ensuring robust matching.
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Store keypoint counts
    global stats
    if tag == "LC":
        stats["kp_L"] = len(kp1)
        stats["kp_C"] = len(kp2)
    elif tag == "RC":
        stats["kp_R"] = len(kp1)
        # Center keypoints were already counted in the LC step.

    # Use FLANN matcher for efficiency on larger datasets compared to BFMatcher.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Retrieve top 2 matches for each point to apply Lowe's Ratio Test.
    if des1 is None or des2 is None:
        print("Warning: No descriptors found.")
        return kp1, kp2, []

    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's Ratio Test:
    # Retain matches where the first match is significantly better (distance < 0.7) than the second.
    # This filters out ambiguous matches, such as repetitive patterns.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if tag == "LC":
        stats["matches_LC"] = len(good)
    if tag == "RC":
        stats["matches_RC"] = len(good)

    return kp1, kp2, good


def compute_homography_dlt(src_pts, dst_pts):
    # Manual implementation of Homography matrix estimation.
    # Uses the Direct Linear Transform (DLT) method to solve for H.
    num_points = src_pts.shape[0]
    if num_points < 4:
        return None

    A = []
    for i in range(num_points):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]

        # Each point correspondence contributes two rows to matrix A for the system Ah = 0.
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    A = np.array(A)

    # Solve Ah = 0 using Singular Value Decomposition (SVD).
    # The solution h corresponds to the last column of V, associated with the smallest singular value.
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # Normalize H such that the last element is 1, following standard convention.
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]

    return H


def ransac_homography(src_pts, dst_pts, thresh=5.0, max_iters=2000):
    # Manual implementation of RANSAC (Random Sample Consensus).
    # Robustly estimates the best Homography matrix in the presence of outliers.
    num_points = src_pts.shape[0]
    if num_points < 4:
        return None, None

    best_H = None
    best_inliers_count = 0
    best_mask = np.zeros((num_points, 1), dtype=np.uint8)

    # Convert points to homogeneous coordinates (x, y, 1).
    src_pts_h = np.hstack((src_pts, np.ones((num_points, 1))))

    for _ in range(max_iters):
        # Randomly select 4 points to estimate a model
        idx = np.random.choice(num_points, 4, replace=False)
        src_sample = src_pts[idx]
        dst_sample = dst_pts[idx]

        # Compute the homography for these 4 points
        H_sample = compute_homography_dlt(src_sample, dst_sample)
        if H_sample is None:
            continue

        # Project all points using this estimated H
        dst_proj = (H_sample @ src_pts_h.T).T

        # Normalize by dividing by the last component w (perspective division)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = dst_proj[:, 2:3]
            w[np.abs(w) < 1e-10] = 1e-10  # Avoiding division by zero
            dst_proj_norm = dst_proj[:, :2] / w

        # Calculate the distance between projected points and actual destination points
        distances = np.linalg.norm(dst_pts - dst_proj_norm, axis=1)

        # Count inliers (points that fit the model well)
        inliers_idx = distances < thresh
        num_inliers = np.sum(inliers_idx)

        # Update best model if current model has more inliers
        if num_inliers > best_inliers_count:
            best_inliers_count = num_inliers
            best_H = H_sample
            best_mask = inliers_idx.astype(np.uint8).reshape(-1, 1)

    # Recompute H using all inliers for improved accuracy.
    if best_H is not None and best_inliers_count >= 4:
        inlier_src = src_pts[best_mask.flatten() == 1]
        inlier_dst = dst_pts[best_mask.flatten() == 1]
        refined_H = compute_homography_dlt(inlier_src, inlier_dst)
        if refined_H is not None:
            best_H = refined_H

    return best_H, best_mask


def compute_homography(img_src, img_dst, tag=""):
    # Wrapper function to align the source image to the destination image.
    kp_src, kp_dst, matches = detect_and_match(img_src, img_dst, tag)

    # At least 4 matches are required to solve for the 8 degrees of freedom in Homography.
    if len(matches) < 4:
        print("Not enough matches found!")
        return None, None, None, None

    # Extracting the (x, y) coordinates
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches])

    # Use manual RANSAC implementation to robustly find H.
    H, mask = ransac_homography(src_pts, dst_pts, thresh=5.0)

    # Count inliers for reporting.
    inliers = np.sum(mask)
    if tag == "LC":
        stats["inliers_LC"] = int(inliers)
    if tag == "RC":
        stats["inliers_RC"] = int(inliers)

    return H, kp_src, kp_dst, matches


def stitch_images_centered(img_L, img_C, img_R):
    # Anchor all images to the Center frame instead of sequential stitching.
    # This minimizes distortion by preventing double warping of the Left image.
    print("Computing Alignment: Left -> Center...")
    H_L_to_C, kpL, kpC, matchLC = compute_homography(img_L, img_C, tag="LC")

    # Save visualization of matches for the report.
    if kpL is not None:
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

    # Determine the size of the new canvas.
    # Transform the corners of L and R into C's coordinate system.
    h_c, w_c = img_C.shape[:2]
    h_l, w_l = img_L.shape[:2]
    h_r, w_r = img_R.shape[:2]

    corners_C = np.float32([[0, 0], [0, h_c], [w_c, h_c], [w_c, 0]]).reshape(-1, 1, 2)
    corners_L = np.float32([[0, 0], [0, h_l], [w_l, h_l], [w_l, 0]]).reshape(-1, 1, 2)
    corners_R = np.float32([[0, 0], [0, h_r], [w_r, h_r], [w_r, 0]]).reshape(-1, 1, 2)

    corners_L_trans = cv2.perspectiveTransform(corners_L, H_L_to_C)
    corners_R_trans = cv2.perspectiveTransform(corners_R, H_R_to_C)

    # Calculate min/max coordinates to determine the final panorama size.
    all_points = np.concatenate((corners_C, corners_L_trans, corners_R_trans), axis=0)

    [xmin, ymin] = np.int32(all_points.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_points.max(axis=0).ravel() + 0.5)

    # Create a translation matrix to shift everything into positive coordinates.
    translation_dist = [-xmin, -ymin]
    H_translation = np.array(
        [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]],
        dtype=np.float32,
    )

    # Combining the homographies with the translation
    final_H_L = H_translation.dot(H_L_to_C)
    final_H_R = H_translation.dot(H_R_to_C)

    output_shape = (xmax - xmin, ymax - ymin)

    print("Warping Left image...")
    warped_L = cv2.warpPerspective(img_L, final_H_L, output_shape)

    print("Warping Right image...")
    warped_R = cv2.warpPerspective(img_R, final_H_R, output_shape)

    print("Placing Center image...")
    warped_C = cv2.warpPerspective(img_C, H_translation, output_shape)

    # Create masks for regions with image data (pixels > 0), checking all channels.
    mask_L = (warped_L.sum(axis=2) > 0).astype(np.float32)
    mask_C = (warped_C.sum(axis=2) > 0).astype(np.float32)
    mask_R = (warped_R.sum(axis=2) > 0).astype(np.float32)

    # Exposure Compensation (Gain Adjustment).
    # Maintain brightness consistency by matching L and R to C.
    print("Applying Exposure Compensation...")

    # Overlap between Left and Center
    overlap_LC = (mask_L * mask_C) > 0
    if np.sum(overlap_LC) > 0:
        mean_L = np.mean(warped_L[overlap_LC])
        mean_C_sub = np.mean(warped_C[overlap_LC])
        gain_L = mean_C_sub / (mean_L + 1e-5)
        print(f"  -> Compensating L->C: Gain = {gain_L:.2f}")
        warped_L = np.clip(warped_L * gain_L, 0, 255).astype(np.uint8)

    # Overlap between Right and Center
    overlap_RC = (mask_R * mask_C) > 0
    if np.sum(overlap_RC) > 0:
        mean_R = np.mean(warped_R[overlap_RC])
        mean_C_sub = np.mean(warped_C[overlap_RC])
        gain_R = mean_C_sub / (mean_R + 1e-5)
        print(f"  -> Compensating R->C: Gain = {gain_R:.2f}")
        warped_R = np.clip(warped_R * gain_R, 0, 255).astype(np.uint8)

    # Linear Blending using Distance Transform (Feathering).
    # Pixels are weighted based on their distance from the edge.
    print("Blending images using Distance Transform...")
    dist_L = cv2.distanceTransform(mask_L.astype(np.uint8), cv2.DIST_L2, 5)
    dist_C = cv2.distanceTransform(mask_C.astype(np.uint8), cv2.DIST_L2, 5)
    dist_R = cv2.distanceTransform(mask_R.astype(np.uint8), cv2.DIST_L2, 5)

    w_L = dist_L[..., np.newaxis]
    w_C = dist_C[..., np.newaxis]
    w_R = dist_R[..., np.newaxis]

    numerator = (
        warped_L.astype(np.float32) * w_L
        + warped_C.astype(np.float32) * w_C
        + warped_R.astype(np.float32) * w_R
    )

    denominator = w_L + w_C + w_R
    final_pano = numerator / (denominator + 1e-5)

    return np.clip(final_pano, 0, 255).astype(np.uint8)


def crop_black_borders(img):
    # Remove extra black space around the panorama.
    # Find the largest object (the panorama) and crop to its bounding box.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return img[y : y + h, x : x + w]
    return img


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
    match_lc_pct = (
        (stats["inliers_LC"] / stats["matches_LC"] * 100)
        if stats["matches_LC"] > 0
        else 0
    )
    match_rc_pct = (
        (stats["inliers_RC"] / stats["matches_RC"] * 100)
        if stats["matches_RC"] > 0
        else 0
    )

    print(
        f"  • Left → Center:  {stats['matches_LC']:,} matches → {stats['inliers_LC']:,} inliers ({match_lc_pct:.1f}%)"
    )
    print(
        f"  • Right → Center: {stats['matches_RC']:,} matches → {stats['inliers_RC']:,} inliers ({match_rc_pct:.1f}%)"
    )
    print("\n" + "=" * 50 + "\n")
