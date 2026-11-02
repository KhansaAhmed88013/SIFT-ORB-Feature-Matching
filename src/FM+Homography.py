
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10
TEST_SIZE = 0.20
RANDOM_SEED = 42
RANSAC_THRESHOLD = 3.0
TEST_THRESHOLD = 3.0

# Load images
img0 = cv.imread("test1.png", cv.IMREAD_GRAYSCALE)
img1 = cv.imread("test2.png", cv.IMREAD_GRAYSCALE)

if img0 is None or img1 is None:
    raise FileNotFoundError("Could not load input images. Check image paths.")

# Initialize SIFT detector
sift = cv.SIFT_create()

# Detect keypoints and descriptors
kp0, dsc0 = sift.detectAndCompute(img0, None)
kp1, dsc1 = sift.detectAndCompute(img1, None)

if dsc0 is None or dsc1 is None:
    raise ValueError("No features detected in one or both images.")

print("Keypoints in image 1:", len(kp0))
print("Keypoints in image 2:", len(kp1))

# Initialize FLANN matcher
FLANN_INDEX_KDTREE = 1

index_params = dict(
    algorithm=FLANN_INDEX_KDTREE,
    trees=5
)

search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

# Find the two nearest matches
matches = flann.knnMatch(dsc0, dsc1, k=2)

# Lowe's ratio test
good = []

for pair in matches:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good.append(m)

print("\nGood matches:", len(good))

matchesMask = None
warped = None
overlay = None
img1_display = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)

# ----------------------------------
# HOMOGRAPHY AND VALIDATION
# ----------------------------------

if len(good) >= MIN_MATCH_COUNT:

    src_pts = np.float32(
        [kp0[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp1[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    # 80/20 dataset split
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(good))

    test_count = max(1, int(len(good) * TEST_SIZE))

    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    train_src = src_pts[train_idx]
    train_dst = dst_pts[train_idx]

    test_src = src_pts[test_idx]
    test_dst = dst_pts[test_idx]

    print("\n--- Dataset Split ---")
    print("Training matches:", len(train_idx))
    print("Testing matches:", len(test_idx))

    # Estimate homography using training matches
    M, train_mask = cv.findHomography(
        train_src,
        train_dst,
        cv.RANSAC,
        RANSAC_THRESHOLD
    )

    if M is not None and train_mask is not None:

        # ----------------------------------
        # TRAINING PERFORMANCE
        # ----------------------------------

        train_inlier_mask = train_mask.ravel().astype(bool)
        train_inliers = int(np.sum(train_inlier_mask))
        train_ratio = train_inliers / len(train_idx)

        train_projected = cv.perspectiveTransform(
            train_src[train_inlier_mask],
            M
        )

        train_errors = np.linalg.norm(
            train_projected.reshape(-1, 2) -
            train_dst[train_inlier_mask].reshape(-1, 2),
            axis=1
        )

        print("\n--- Training Results ---")
        print("RANSAC inliers:", train_inliers)
        print("Inlier ratio:", round(train_ratio, 3))
        print("Mean reprojection error:",
              round(np.mean(train_errors), 3), "pixels")
        print("Median reprojection error:",
              round(np.median(train_errors), 3), "pixels")

        # ----------------------------------
        # INDEPENDENT TESTING
        # ----------------------------------

        test_projected = cv.perspectiveTransform(
            test_src,
            M
        )

        test_errors = np.linalg.norm(
            test_projected.reshape(-1, 2) -
            test_dst.reshape(-1, 2),
            axis=1
        )

        # Geometric validation of test matches
        test_inlier_mask = test_errors <= TEST_THRESHOLD

        test_inliers = int(np.sum(test_inlier_mask))
        test_outliers = len(test_errors) - test_inliers
        test_ratio = test_inliers / len(test_errors)

        valid_test_errors = test_errors[test_inlier_mask]

        print("\n--- Independent Testing Results ---")
        print("Total testing matches:", len(test_errors))
        print("Test inliers:", test_inliers)
        print("Test outliers:", test_outliers)
        print("Test inlier ratio:", round(test_ratio, 3))

        print("\n--- Validated Test Reprojection Error ---")

        if test_inliers > 0:
            print("Mean error:",
                  round(np.mean(valid_test_errors), 3), "pixels")
            print("Median error:",
                  round(np.median(valid_test_errors), 3), "pixels")
            print("Maximum error:",
                  round(np.max(valid_test_errors), 3), "pixels")
        else:
            print("No geometrically consistent test matches found.")

        print("\n--- All Test Correspondences ---")
        print("Mean error:", round(np.mean(test_errors), 3), "pixels")
        print("Median error:", round(np.median(test_errors), 3), "pixels")
        print("Maximum error:", round(np.max(test_errors), 3), "pixels")

        # ----------------------------------
        # FINAL HOMOGRAPHY USING ALL MATCHES
        # ----------------------------------

        M_final, final_mask = cv.findHomography(
            src_pts,
            dst_pts,
            cv.RANSAC,
            RANSAC_THRESHOLD
        )

        if M_final is not None and final_mask is not None:

            matchesMask = final_mask.ravel().tolist()

            final_inliers = int(final_mask.sum())
            final_ratio = final_inliers / len(good)

            print("\n--- Final Homography Results ---")
            print("Total good matches:", len(good))
            print("Final RANSAC inliers:", final_inliers)
            print("Final inlier ratio:", round(final_ratio, 3))

            # Get dimensions of first image
            h, w = img0.shape

            # Define image corners
            pts = np.float32([
                [0, 0],
                [0, h - 1],
                [w - 1, h - 1],
                [w - 1, 0]
            ]).reshape(-1, 1, 2)

            # Transform corners into second image coordinates
            dst = cv.perspectiveTransform(pts, M_final)

            # Draw projected boundary
            cv.polylines(
                img1_display,
                [np.int32(dst)],
                True,
                (0, 255, 0),
                3,
                cv.LINE_AA
            )

            # Warp first image into second image perspective
            warped = cv.warpPerspective(
                img0,
                M_final,
                (img1.shape[1], img1.shape[0])
            )

            # Blend warped image with target image
            overlay = cv.addWeighted(
                warped,
                0.5,
                img1,
                0.5,
                0
            )

            print("\nHomography calculated successfully.")

        else:
            print("Final homography calculation failed.")

    else:
        print("Training homography calculation failed.")

else:
    print(
        "Not enough matches: {}/{}".format(
            len(good),
            MIN_MATCH_COUNT
        )
    )

# ----------------------------------
# FEATURE MATCHING VISUALIZATION
# ----------------------------------

if matchesMask is not None:

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    img3 = cv.drawMatches(
        img0,
        kp0,
        img1_display,
        kp1,
        good,
        None,
        **draw_params
    )

    plt.figure(figsize=(16, 8))
    plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
    plt.title("SIFT Feature Matching + Homography (RANSAC)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ----------------------------------
# IMAGE ALIGNMENT VISUALIZATION
# ----------------------------------

if warped is not None and overlay is not None:

    plt.figure(figsize=(18, 6))

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
