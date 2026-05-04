# Homography-Based 3D Object Boundary Detection

A computer vision pipeline that detects and localizes electrical sockets (power, VGA, ethernet) in 3D space using homography, SIFT feature matching, triangulation, and Oriented Bounding Box (OBB) estimation.

---

## Overview

This project takes a sequence of frames from a camera and:

1. **Annotates** socket boundaries manually on reference frames
2. **Tracks** those boundaries across frames using RootSIFT feature matching and homography
3. **Triangulates** the 2D projections into 3D point clouds using known camera intrinsics and poses
4. **Fits a plane** to the 3D points via RANSAC and computes an OBB (Oriented Bounding Box)
5. **Evaluates** the predicted OBB against ground truth using 2D IoU on projected faces

---

## Project Structure

```
project/
│
├── homography.ipynb                        # Main pipeline notebook
│
├── Data/
│   ├── frame_000XXX.png                    # Input video frames (e.g., frame_000449.png)
│   ├── intrinsic.json                      # Camera intrinsic matrix & distortion coefficients
│   ├── poses.json                          # Camera poses per frame (4x4 T_cw matrices)
│   ├── bboxes_power_socket.json            # Bounding boxes for power sockets per frame
│   ├── bboxes_ethernet_socket.json         # Bounding boxes for ethernet sockets
│   ├── bboxes_vga_socket.json              # Bounding boxes for VGA sockets
│   └── sample_answers.json                 # Ground truth OBBs for evaluation
│
├── manual_power_socket_boundaries.json     # Annotated 4-corner boundaries (generated)
├── template_corners.npy                    # Saved template corners for the keyframe (generated)
└── README.md
```

---

## Requirements

### System Dependencies

- Python 3.8+
- OpenCV with contrib modules (for SIFT)

### Python Dependencies

Install all dependencies with:

```bash
pip install opencv-contrib-python numpy shapely matplotlib
```

Or using a `requirements.txt`:

```
opencv-contrib-python
numpy
shapely
matplotlib
```

> **Note:** Standard `opencv-python` does **not** include SIFT. You must install `opencv-contrib-python`.

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install opencv-contrib-python numpy shapely matplotlib

# 4. Launch the notebook
jupyter notebook homography.ipynb
```

---

## Data Format

### `intrinsic.json`
```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "image_width": 1280,
  "image_height": 720,
  "distortion_coefficients": [k1, k2, p1, p2, k3]
}
```

### `poses.json`
```json
{
  "319": [[r00, r01, r02, tx], [r10, r11, r12, ty], [r20, r21, r22, tz], [0,0,0,1]],
  "333": ...
}
```
Each entry is a 4×4 `T_cw` matrix (world → camera transform).

### `bboxes_<device_type>.json`
```json
{
  "319": [x, y, width, height],
  "333": ...
}
```

### `sample_answers.json`
```json
[
  {
    "entity": "power_socket",
    "obb": {
      "center": [x, y, z],
      "extent": [ex, ey, ez],
      "rotation": [[...], [...], [...]]
    }
  }
]
```

---

## How to Run

### Step 1 — Annotate Boundaries (One-time Setup)

Run the **annotation cell** to manually click the 4 corners of the socket on each frame.

- **`n`** — save current annotation and go to next frame
- **`r`** — reset current frame's annotation
- **`q`** — quit early

This saves `manual_power_socket_boundaries.json`.

### Step 2 — Run the Full Pipeline

Execute the notebook cells in order:

| Cell | Description |
|---|---|
| Load intrinsics | Reads `Data/intrinsic.json` → camera matrix `K` |
| TrackManager | Initializes feature track storage |
| `sift_in_bbox` | Extracts RootSIFT keypoints within the bounding box |
| `match_and_visualise` | Matches features, computes homography, projects boundaries |
| Load camera poses | Reads `Data/poses.json` → per-frame `R`, `t` |
| `triangulate_points` | Lifts 2D boundary corners to 3D using multi-view geometry |
| RANSAC plane fitting | Fits a plane to 3D points, removes outliers |
| OBB estimation | Computes center, extent, and rotation via SVD |
| `compare_obbs` | Projects predicted and GT OBBs and computes 2D IoU |

### Step 3 — View Results

Each frame opens an OpenCV window showing the projected boundary in **green**. The final evaluation window shows:
- 🟢 **Green** — Predicted OBB projection
- 🔴 **Red** — Ground truth OBB projection
- Console output: `IoU: <value>`

---

## Key Parameters to Configure

| Variable | Location | Description |
|---|---|---|
| `device_type` | Main loop cell | `"power_socket"`, `"ethernet_socket"`, or `"vga_socket"` |
| `keyframe_idx` | Main loop cell | The reference frame index (e.g., `426`) |
| `valid_image_ids` | Main loop cell | List of frame IDs to process |
| `RANSAC threshold` (homography) | `match_and_visualise` | `6.5` pixels — tune for accuracy |
| Lowe ratio test | `match_and_visualise` | `0.6` — lower = stricter matching |
| RANSAC threshold (plane) | Plane fitting cell | `0.002` meters — tune for point cloud density |

---

## Pipeline Architecture

```
Input Frames
     │
     ▼
Manual Boundary Annotation (4 corners per frame)
     │
     ▼
Keyframe Selection → RootSIFT Feature Extraction (within bbox)
     │
     ▼
Feature Matching (BFMatcher + Lowe ratio test)
     │
     ▼
Homography Estimation (RANSAC) → Project boundary to each frame
     │
     ▼
Multi-view Triangulation (using K + camera poses)
     │
     ▼
RANSAC Plane Fitting → PCA/SVD → OBB (center, extent, rotation)
     │
     ▼
2D IoU Evaluation vs. Ground Truth OBB
```

---

## Known Limitations & TODOs

- Hardcoded file paths — should be replaced with configurable parameters
- Manual boundary annotation is required for each new sequence
- Point ordering in homography is not safeguarded automatically
- SIFT matching could be improved (e.g., adding FLANN-based matching)
- Code cleanup needed: commented-out code, unused imports, docstrings

---

## License

This project is intended for research and educational use.
