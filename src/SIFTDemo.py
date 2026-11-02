"""Demonstrate SIFT feature detection and descriptor matching."""
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IMAGE_1 = ROOT / "data" / "test1.png"
IMAGE_2 = ROOT / "data" / "test2.png"
LOWE_RATIO = 0.75

img0 = cv.imread(str(IMAGE_1), cv.IMREAD_GRAYSCALE)
img1 = cv.imread(str(IMAGE_2), cv.IMREAD_GRAYSCALE)

if img0 is None or img1 is None:
    raise FileNotFoundError(
        f"Could not load input images: {IMAGE_1} and/or {IMAGE_2}"
    )

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img0, None)
kp2, des2 = sift.detectAndCompute(img1, None)

if des1 is None or des2 is None:
    raise RuntimeError("SIFT could not compute descriptors for both images.")

matcher = cv.BFMatcher(cv.NORM_L2)
knn_matches = matcher.knnMatch(des1, des2, k=2)
good_matches = [
    [m] for pair in knn_matches
    if len(pair) == 2
    for m, n in [pair]
    if m.distance < LOWE_RATIO * n.distance
]

match_image = cv.drawMatchesKnn(
    img0, kp1, img1, kp2, good_matches[:30], None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

print(f"Keypoints: image 1={len(kp1)}, image 2={len(kp2)}")
print(f"Good matches after Lowe's ratio test: {len(good_matches)}")

plt.figure(figsize=(14, 7))
plt.imshow(cv.cvtColor(match_image, cv.COLOR_BGR2RGB))
plt.title("SIFT: Best 30 Matches After Lowe's Ratio Test")
plt.axis("off")
plt.tight_layout()
plt.show()
