# SIFT vs ORB: Feature Matching and Homography Estimation

A Python/OpenCV computer-vision project that compares **SIFT (Scale-Invariant Feature Transform)** and **ORB (Oriented FAST and Rotated BRIEF)** for local feature detection, descriptor matching, and geometric verification across image viewpoints.

## Project overview

The pipeline covers:

1. Feature detection and descriptor extraction.
2. FLANN-based K-nearest-neighbour matching for SIFT.
3. Brute-force Hamming matching for ORB.
4. Lowe's ratio test to reject ambiguous matches.
5. An 80/20 split of matched correspondences into training and testing subsets.
6. RANSAC homography estimation using training correspondences.
7. Independent test-set geometric validation using a **3-pixel reprojection-error threshold**.
8. Perspective warping and alignment visualisation.

> **Important scope note:** The supplied archive contains one input image pair (`test1.png` and `test2.png`) and one pre-existing result image. The three SIFT scenarios and the ORB feature-count configurations described in the experiment report are recorded below, but the other original image pairs and all generated plots were not present in the supplied archive. Add those source images/results if you want every reported experiment to be directly reproducible from this repository.

## Repository structure

```text
SIFT-ORB-Feature-Matching/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── test1.png
│   └── test2.png
├── results/
│   └── result.jpg
└── src/
    ├── BFMDemo.py
    ├── FM+Homography.py
    ├── ORB+Homography.py
    ├── SIFTDemo.py
    └── Stitcher.py
```

All scripts resolve image paths relative to the repository root, so they can be launched from the root directory or another working directory.

## Experimental results

The following values are from the experiment report supplied with this project.

### SIFT results

| Metric | Original viewpoint | Extreme viewpoint | Moderate viewpoint |
|---|---:|---:|---:|
| Good matches | 7,676 | 14 | 573 |
| Training inliers | 2,945 | 4 | 232 |
| Training inlier ratio | 48.0% | 33.3% | 50.5% |
| Testing matches | 1,535 | 2 | 114 |
| Testing inliers | 705 | 0 | 46 |
| Testing inlier ratio | 45.9% | 0.0% | 40.4% |
| Validated test mean error | 1.098 px | N/A | 1.262 px |

**Observation:** SIFT performed well on the original and moderate viewpoint pairs. Its performance deteriorated substantially under the extreme viewpoint change.

### ORB results

ORB was evaluated on the moderate-viewpoint pair with two feature limits.

| Metric | ORB (8,000 features) | ORB (20,000 features) |
|---|---:|---:|
| Keypoints per image (configured limit) | 8,000 | 20,000 |
| Good matches | 80 | 177 |
| Training inliers | 20 | 44 |
| Training inlier ratio | 31.2% | 31.2% |
| Testing matches | 16 | 36 |
| Testing inliers | 4 | 10 |
| Testing inlier ratio | 25.0% | 27.8% |
| Validated test mean error | 1.738 px | 1.269 px |
| Validated test median error | 1.563 px | 1.074 px |

**Observation:** Increasing ORB's feature limit increased the number of matches and validated test inliers. In the reported moderate-viewpoint experiment, ORB remained below SIFT's matching performance.

### Overall comparison

| Criterion | SIFT | ORB |
|---|---|---|
| Descriptor | Floating-point | Binary |
| Matcher used | FLANN | Brute-force Hamming |
| Correspondences in reported tests | Higher | Lower |
| Matching/geometric results | Better in tested pairs | Lower in tested pair |
| Typical computational profile | Generally slower | Generally faster |

**Timing note:** Runtime was not benchmarked. The speed comparison is a general algorithmic characteristic, not a measured result from this project.

## Requirements

- Python 3.9 or newer (a recent Python version is recommended).
- OpenCV with SIFT support (`opencv-contrib-python`).
- NumPy.
- Matplotlib.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/SIFT-ORB-Feature-Matching.git
cd SIFT-ORB-Feature-Matching
```

Create and activate a virtual environment (recommended):

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Install only one OpenCV wheel in the environment. This project uses `opencv-contrib-python`; do not install `opencv-python` alongside it.

## Run the scripts

Run commands from the repository root.

### SIFT homography and independent test validation

```bash
python "src/FM+Homography.py"
```

### ORB homography and independent test validation

```bash
python "src/ORB+Homography.py"
```

### SIFT matching demonstration

```bash
python src/SIFTDemo.py
```

### ORB brute-force matching demonstration

```bash
python src/BFMDemo.py
```

### OpenCV image stitching sample

Pass two or more image paths:

```bash
python src/Stitcher.py data/test1.png data/test2.png
```

The stitching script writes its output to `results/stitched_result.jpg` by default. Image stitching is a separate OpenCV sample and is not the same as the SIFT/ORB homography evaluation.

## Methodology notes

- The 80/20 split is performed over the accepted feature correspondences, not over separate source images.
- RANSAC estimates the training homography with a 3-pixel threshold.
- Test correspondences are evaluated independently against that training homography. A test correspondence is counted as geometrically consistent when its reprojection error is at most 3 pixels.
- The scripts also estimate a homography from all accepted matches for final visualisation. This all-match model is for visualisation and should not be confused with the independent test evaluation.
- The report's SIFT and ORB tables are retained as reported experimental results; the included scripts may need their configuration adjusted to reproduce every historical run exactly.

## Limitations and future work

- Include all original image pairs to reproduce the three viewpoint scenarios.
- Save and add generated match, inlier, warped-image, and overlay visualisations.
- Benchmark execution time and memory use under the same hardware and settings.
- Evaluate additional scale, rotation, illumination, and viewpoint changes.
- Test on aerial/drone imagery and more challenging scenes.
- Compare additional detectors, descriptors, and matching strategies.

## Data and image attribution

The two input images are included from the project archive supplied for repository preparation. Before publishing, verify that you have permission to redistribute both images. Add the original source URL, creator, and applicable licence for each image here:

- `data/test1.png` — source/creator/licence: **please complete**
- `data/test2.png` — source/creator/licence: **please complete**

## License

No open-source licence has been selected for this repository. Unless you add a licence file, others should not assume they have permission to reuse, modify, or redistribute the project.
