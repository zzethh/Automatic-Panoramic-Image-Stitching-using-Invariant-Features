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
        stats["kp_C"] = len(kp2)
    elif tag == "RC":
        stats["kp_R"] = len(kp1)
        # Center keypoints already counted if LC ran first, but safe to overwrite or ignore

    # Using FLANN matcher because it's faster than BFMatcher (Brute Force) for large datasets.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Getting the top 2 matches for each point so I can run Lowe's Ratio Test
    if des1 is None or des2 is None:
        print("Warning: No descriptors found.")
        return kp1, kp2, []

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


def compute_homography_dlt(src_pts, dst_pts):
    # This acts as my manual implementation of finding the Homography matrix.
    # I'm using the Direct Linear Transform (DLT) method to solve for H.
    num_points = src_pts.shape[0]
    if num_points < 4:
        return None

    A = []
    for i in range(num_points):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]

        # Each point correspondence gives me two rows in matrix A for the equation Ah = 0
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    A = np.array(A)

    # I solve Ah = 0 using Singular Value Decomposition (SVD).
    # The solution h is the last column of V corresponding to the smallest singular value.
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # I normalize H so that the last element is 1, which is standard convention.
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]

    return H


def ransac_homography(src_pts, dst_pts, thresh=5.0, max_iters=2000):
    # Here is my manual implementation of RANSAC (Random Sample Consensus).
    # It helps me find the best Homography matrix even when there are outliers (bad matches).
    num_points = src_pts.shape[0]
    if num_points < 4:
        return None, None

    best_H = None
    best_inliers_count = 0
    best_mask = np.zeros((num_points, 1), dtype=np.uint8)

    # I need to add 1 to the points to make them homogeneous coordinates (x, y, 1)
    src_pts_h = np.hstack((src_pts, np.ones((num_points, 1))))

    for _ in range(max_iters):
        # 1. I randomly select 4 points to estimate a model
        idx = np.random.choice(num_points, 4, replace=False)
        src_sample = src_pts[idx]
        dst_sample = dst_pts[idx]

        # 2. I compute the homography for just these 4 points
        H_sample = compute_homography_dlt(src_sample, dst_sample)
        if H_sample is None:
            continue

        # 3. I project all points using this estimated H
        dst_proj = (H_sample @ src_pts_h.T).T

        # I normalize by dividing by the last component w (perspective division)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = dst_proj[:, 2:3]
            w[np.abs(w) < 1e-10] = 1e-10  # Avoiding division by zero
            dst_proj_norm = dst_proj[:, :2] / w

        # 4. I calculate the distance between projected points and actual destination points
        distances = np.linalg.norm(dst_pts - dst_proj_norm, axis=1)

        # 5. I count how many points fit this model well (inliers)
        inliers_idx = distances < thresh
        num_inliers = np.sum(inliers_idx)

        # If this model is better than my previous best, I keep it.
        if num_inliers > best_inliers_count:
            best_inliers_count = num_inliers
            best_H = H_sample
            best_mask = inliers_idx.astype(np.uint8).reshape(-1, 1)

    # Optional: I can recompute H using all the inliers to get a slightly more accurate result.
    if best_H is not None and best_inliers_count >= 4:
        inlier_src = src_pts[best_mask.flatten() == 1]
        inlier_dst = dst_pts[best_mask.flatten() == 1]
        refined_H = compute_homography_dlt(inlier_src, inlier_dst)
        if refined_H is not None:
            best_H = refined_H

    return best_H, best_mask


def compute_homography(img_src, img_dst, tag=""):
    # This wrapper aligns the source image to the destination image.
    kp_src, kp_dst, matches = detect_and_match(img_src, img_dst, tag)

    # I need at least 4 matches to solving for the 8 degrees of freedom in Homography.
    if len(matches) < 4:
        print("Not enough matches found!")
        return None, None, None, None

    # Extracting the (x, y) coordinates
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches])

    # Using my manual RANSAC implementation to robustly find H
    H, mask = ransac_homography(src_pts, dst_pts, thresh=5.0)

    # Counting inliers for my report
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

    # I'll save a visualization of the matches for the report.
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

    # I'll find the min/max coordinates to know the size of the final panorama
    all_points = np.concatenate((corners_C, corners_L_trans, corners_R_trans), axis=0)

    [xmin, ymin] = np.int32(all_points.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_points.max(axis=0).ravel() + 0.5)

    # I need a translation matrix to shift everything into positive coordinates if xmin/ymin < 0
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

    # 1. I create masks for where we actually have image data (pixels > 0)
    #    I checking all channels to avoid missing dark pixels.
    mask_L = (warped_L.sum(axis=2) > 0).astype(np.float32)
    mask_C = (warped_C.sum(axis=2) > 0).astype(np.float32)
    mask_R = (warped_R.sum(axis=2) > 0).astype(np.float32)

    # 2. Exposure Compensation (Simple Gain Adjustment)
    #    Sometimes one image is brighter than the other. I'll match L and R to C's brightness.
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

    # 3. Linear Blending using Distance Transform (Feathering)
    #    I'm weighting pixels based on how far they are from the edge.
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
    # This just gets rid of the extra black space around the panorama.
    # It finds the largest object (the pano) and crops to its bounding box.
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
