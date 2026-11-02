
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================

MIN_MATCH_COUNT = 10
TEST_SIZE = 0.20
RANDOM_SEED = 42
RANSAC_THRESHOLD = 3.0
TEST_THRESHOLD = 3.0

ORB_FEATURES = 20000
LOWE_RATIO = 0.75

# Update filenames here if yours are different.
IMAGE_PAIRS = [
    ("Moderate Viewpoint", "test1.png", "test2.png"),
]


def calculate_error(H, src, dst):
    projected = cv.perspectiveTransform(
        src.reshape(-1, 1, 2), H
    ).reshape(-1, 2)

    return np.linalg.norm(projected - dst, axis=1)


def run_orb_experiment(pair_name, path1, path2):

    print("\n" + "=" * 55)
    print(pair_name)
    print("=" * 55)

    img0 = cv.imread(path1, cv.IMREAD_GRAYSCALE)
    img1 = cv.imread(path2, cv.IMREAD_GRAYSCALE)

    if img0 is None or img1 is None:
        print("ERROR: Could not load images.")
        print("Check filenames and image paths.")
        return

    # ==========================================
    # 1. ORB FEATURE DETECTION
    # ==========================================

    orb = cv.ORB_create(nfeatures=ORB_FEATURES)

    kp1, des1 = orb.detectAndCompute(img0, None)
    kp2, des2 = orb.detectAndCompute(img1, None)

    print(f"Keypoints in image 1: {len(kp1)}")
    print(f"Keypoints in image 2: {len(kp2)}")

    if des1 is None or des2 is None:
        print("ERROR: Descriptors not found.")
        return

    # ==========================================
    # 2. BRUTE FORCE MATCHING (HAMMING)
    # ==========================================

    matcher = cv.BFMatcher(cv.NORM_HAMMING)

    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []

    for match in knn_matches:
        if len(match) == 2:
            m, n = match

            if m.distance < LOWE_RATIO * n.distance:
                good_matches.append(m)

    print(f"\nGood matches: {len(good_matches)}")

    if len(good_matches) < MIN_MATCH_COUNT:
        print("Not enough matches for homography.")
        return

    # ==========================================
    # 3. EXTRACT CORRESPONDENCES
    # ==========================================

    src_pts = np.float32([
        kp1[m.queryIdx].pt for m in good_matches
    ])

    dst_pts = np.float32([
        kp2[m.trainIdx].pt for m in good_matches
    ])

    # ==========================================
    # 4. TRAIN / TEST SPLIT
    # ==========================================

    rng = np.random.default_rng(RANDOM_SEED)

    indices = rng.permutation(len(good_matches))

    split_index = int(len(indices) * (1 - TEST_SIZE))

    train_idx = indices[:split_index]
    test_idx = indices[split_index:]

    train_src = src_pts[train_idx]
    train_dst = dst_pts[train_idx]

    test_src = src_pts[test_idx]
    test_dst = dst_pts[test_idx]

    print("\n--- Dataset Split ---")
    print(f"Training matches: {len(train_idx)}")
    print(f"Testing matches: {len(test_idx)}")

    if len(train_idx) < 4 or len(test_idx) == 0:
        print("Not enough training/testing points.")
        return

    # ==========================================
    # 5. TRAINING HOMOGRAPHY
    # ==========================================

    H_train, train_mask = cv.findHomography(
        train_src,
        train_dst,
        cv.RANSAC,
        RANSAC_THRESHOLD
    )

    if H_train is None or train_mask is None:
        print("Training homography failed.")
        return

    train_inliers = train_mask.ravel().astype(bool)

    train_count = np.count_nonzero(train_inliers)
    train_ratio = train_count / len(train_idx)

    train_errors = calculate_error(
        H_train,
        train_src[train_inliers],
        train_dst[train_inliers]
    )

    print("\n--- Training Results ---")
    print(f"RANSAC inliers: {train_count}")
    print(f"Inlier ratio: {train_ratio:.3f}")

    print(
        f"Mean reprojection error: "
        f"{np.mean(train_errors):.3f} pixels"
    )

    print(
        f"Median reprojection error: "
        f"{np.median(train_errors):.3f} pixels"
    )

    # ==========================================
    # 6. INDEPENDENT TESTING
    # ==========================================

    test_errors = calculate_error(
        H_train,
        test_src,
        test_dst
    )

    test_inliers = test_errors <= TEST_THRESHOLD

    test_inlier_count = np.count_nonzero(test_inliers)
    test_outlier_count = len(test_errors) - test_inlier_count

    test_ratio = test_inlier_count / len(test_errors)

    print("\n--- Independent Testing Results ---")
    print(f"Total testing matches: {len(test_errors)}")
    print(f"Test inliers: {test_inlier_count}")
    print(f"Test outliers: {test_outlier_count}")
    print(f"Test inlier ratio: {test_ratio:.3f}")

    print("\n--- Validated Test Reprojection Error ---")

    if test_inlier_count > 0:

        valid_errors = test_errors[test_inliers]

        print(f"Mean error: {np.mean(valid_errors):.3f} pixels")
        print(f"Median error: {np.median(valid_errors):.3f} pixels")
        print(f"Maximum error: {np.max(valid_errors):.3f} pixels")

    else:
        print("No geometrically consistent test matches found.")

    print("\n--- All Test Correspondences ---")
    print(f"Mean error: {np.mean(test_errors):.3f} pixels")
    print(f"Median error: {np.median(test_errors):.3f} pixels")
    print(f"Maximum error: {np.max(test_errors):.3f} pixels")

    # ==========================================
    # 7. FINAL HOMOGRAPHY FOR VISUALIZATION
    # ==========================================

    H_final, final_mask = cv.findHomography(
        src_pts,
        dst_pts,
        cv.RANSAC,
        RANSAC_THRESHOLD
    )

    if H_final is None or final_mask is None:
        print("Final homography failed.")
        return

    final_inliers = np.count_nonzero(final_mask)
    final_ratio = final_inliers / len(good_matches)

    print("\n--- Final Homography Results ---")
    print(f"Total good matches: {len(good_matches)}")
    print(f"Final RANSAC inliers: {final_inliers}")
    print(f"Final inlier ratio: {final_ratio:.3f}")

    # ==========================================
    # 8. FEATURE MATCH VISUALIZATION
    # ==========================================

    match_image = cv.drawMatches(
        img0,
        kp1,
        img1,
        kp2,
        good_matches,
        None,
        matchesMask=final_mask.ravel().tolist(),
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # ==========================================
    # 9. IMAGE WARPING
    # ==========================================

    height, width = img1.shape

    warped = cv.warpPerspective(
        img0,
        H_final,
        (width, height)
    )

    overlay = cv.addWeighted(
        warped, 0.5,
        img1, 0.5,
        0
    )

    # ==========================================
    # 10. DISPLAY RESULTS
    # ==========================================

    plt.figure(figsize=(16, 8))
    plt.imshow(match_image, cmap="gray")
    plt.title(f"ORB Feature Matching - {pair_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img0, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(warped, cmap="gray")
    plt.title("Warped Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay, cmap="gray")
    plt.title("Alignment Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("\nHomography calculated successfully.")


# ==========================================
# RUN ALL THREE EXPERIMENTS
# ==========================================

for name, image1, image2 in IMAGE_PAIRS:
    run_orb_experiment(name, image1, image2)
