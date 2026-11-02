"""Demonstrate ORB feature detection and brute-force Hamming matching."""
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IMAGE_1 = ROOT / "data" / "test1.png"
IMAGE_2 = ROOT / "data" / "test2.png"

img0 = cv.imread(str(IMAGE_1), cv.IMREAD_GRAYSCALE)
img1 = cv.imread(str(IMAGE_2), cv.IMREAD_GRAYSCALE)

if img0 is None or img1 is None:
    raise FileNotFoundError(
        f"Could not load input images: {IMAGE_1} and/or {IMAGE_2}"
    )

orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(img0, None)
kp2, des2 = orb.detectAndCompute(img1, None)

if des1 is None or des2 is None:
    raise RuntimeError("ORB could not compute descriptors for both images.")

matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = sorted(matcher.match(des1, des2), key=lambda match: match.distance)

match_image = cv.drawMatches(
    img0, kp1, img1, kp2, matches[:30], None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(14, 7))
plt.imshow(cv.cvtColor(match_image, cv.COLOR_BGR2RGB))
plt.title("ORB: Best 30 Brute-Force Hamming Matches")
plt.axis("off")
plt.tight_layout()
plt.show()
