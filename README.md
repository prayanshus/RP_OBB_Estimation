# PC I/O Panel OBB Estimation Pipeline

Robotics Perception course project — IISc Sem 2.

Computes world-frame **Oriented Bounding Boxes (OBBs)** for sockets on a PC tower's I/O panel (VGA/DVI, Ethernet, Power, PS/2, HDMI, USB) from 16 posed images.

---

## Repository Structure

```
RP_OBB_Estimation/
├── docs/                        # Documentation
├── src/
│   ├── data/
│   │   ├── frame_*.png              # 16 input frames (full dataset)
│   │   ├── intrinsic.json           # Camera intrinsic matrix
│   │   ├── poses.json               # Camera-to-world pose per frame
│   │   ├── panel_normal.npy         # Cached panel normal (auto-generated)
│   │   └── test/                    # Reference images for instance selection
│   ├── pipeline/
│   │   ├── auto_obb_sam2_cl_V7.py   # Main automated pipeline
│   │   ├── obb_tool_ge_V0.py        # Manual annotation fallback (Tkinter GUI)
│   │   └── yolo_detector.py         # YOLO inference wrapper
│   ├── yolo/
│   │   ├── data.yaml                # Class definitions
│   │   ├── train_yolo_V2.py         # Training script
│   │   ├── eval_model_V3.py         # Evaluation script
│   │   ├── weights/best.pt          # Trained YOLOv11n weights
│   │   ├── all_training_data/       # Full dataset (images + labels)
│   │   ├── train/                   # Training split (not tracked by git)
│   │   ├── valid/                   # Validation split (not tracked by git)
│   │   └── test/                    # Test split (not tracked by git)
│   ├── sam2/checkpoints/            # SAM2 checkpoint (not tracked by git)
│   ├── outputs/                     # Generated OBB JSONs (not tracked by git)
│   ├── requirements.txt
│   └── requirements_wsl.txt         # Full WSL environment snapshot
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r src/requirements.txt
```

### 2. Install SAM2
```bash
git clone https://github.com/facebookresearch/sam2.git src/sam2
pip install -e src/sam2/
```
Download checkpoint:
```bash
mkdir -p src/sam2/checkpoints
wget -P src/sam2/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

### 3. Place YOLO weights
Copy `best.pt` to `src/yolo/weights/best.pt`.

### 4. Place data
Copy 16 frames + `intrinsic.json` + `poses.json` + `sample_answers.json` into `src/data/`.

Place any images you want to use as reference views into `src/data/test/`. These must be frames whose poses are present in `poses.json` (i.e. named `frame_<N>.png`).

---

## Running the Pipeline

```bash
cd src/pipeline
python auto_obb_sam2_cl_V7.py
```

The interactive session proceeds as follows:

**1. Class selection** — choose a specific socket or run all classes at once:
```
==================================================
  AUTOMATED OBB EXTRACTION PIPELINE
==================================================
  1. VGA_socket
  2. ethernet_socket
  3. power_socket
  4. PS2_socket
  5. HDMI_socket
  6. USB_socket
  7. All classes

Enter the number of the component to measure:
```

**2. Reference image selection** — all images in `src/data/test/` are listed; enter the number of the frame you want to use as the reference view:
```
[System] Test images in src/data/test/:
    1.  frame_000465.png
    2.  frame_000371.png
    ...
Select reference image (number):
```

**3. Instance selection** — YOLO detects every instance of the target class in the reference frame and displays them numbered in a window. Press `1`–`9` to select the one you want to track. Press `M` to draw the box manually if none of the detections are suitable.

**4. Automatic tracking** — the selected instance is matched across all other frames using a two-phase 3D triangulation algorithm. Coverage is reported, and you are prompted if it falls below 15 %.

The pipeline then runs automatically:
- Reprojection-based box filtering
- SAM2 segmentation
- GPU voxel carving
- Panel-normal-aware OBB extraction

**Output files** are written to `src/outputs/`:

| File | Contents |
|---|---|
| `<class>_run<N>.json` | Per-class OBB (center, extent, rotation) |
| `results_<N>.json` | Consolidated JSON for all classes processed in this run |
| `<class>/run<N>/frame_*.png` | OBB wireframe projected onto every detection frame |
| `<class>/run<N>/obb_reference_frame_<N>.png` | OBB overlay on the chosen reference image |

**Force manual annotation mode:**
```bash
python auto_obb_sam2_cl_V7.py --manual_mode
```

---

## Configuration Flags

Two top-level flags at the top of `auto_obb_sam2_cl_V7.py` control pipeline behaviour without any code restructuring:

| Flag | Default | Effect |
|---|---|---|
| `DEBUG` | `False` | When `True`: uses the hardcoded GT panel normal, displays SAM2 mask windows interactively, and shows OBB projection windows during the run. Projection images are always saved to disk regardless. |
| `USE_SAMPLE_GT_NORMAL` | `False` | When `True`: forces use of the hardcoded GT panel normal even if `DEBUG` is `False`. Overrides `DEBUG` for the normal step only. |

---

## Panel Normal Caching

On the first run the panel normal is estimated from scratch via SIFT keypoint matching across all PC_Panel detections, followed by DLT triangulation and RANSAC plane fitting. This can take a while.

The result is saved automatically to `src/data/panel_normal.npy` in full float64 precision. On all subsequent runs the cached value is loaded instantly, skipping the entire estimation step.

To force re-estimation, simply delete `panel_normal.npy`.

---

## Multi-Instance Tracking

Classes with multiple physical instances (e.g. four USB ports) are handled correctly. The pipeline works in two phases:

**Phase 1 — Seed 3D estimate.** Unambiguous frames (exactly one detection) are collected together with the reference frame and triangulated via DLT to produce a consensus 3D position for the chosen instance.

**Phase 2 — Resolve ambiguous frames.** For each frame with multiple detections, the consensus 3D point is projected into that camera and the detection whose 2D centre lies closest to the projection is selected. If fewer than two unambiguous frames exist, a pairwise triangulation + self-consistency reprojection check is used as fallback.

After matching, coverage (% of frames with a matched detection) is reported. If coverage is below 15 %, the pipeline offers three options: proceed anyway, re-run detection with a lower YOLO confidence threshold, or fall back to manual annotation.

---

## Consolidated Output Format

When all classes are processed in a single run, or when explicitly requested, results are written to a single consolidated JSON whose format matches `sample_answers.json`:

```json
[
  {
    "entity": "VGA_socket",
    "obb": {
      "center": [0.270, 0.226, 0.835],
      "extent": [0.0354, 0.0118, 0.0061],
      "rotation": [
        [-0.004,  0.967, -0.254],
        [ 0.016,  0.254,  0.967],
        [ 0.9999, -0.00015, -0.016]
      ]
    }
  }
]
```

---

## Training YOLO

```bash
cd src/yolo
python train_yolo_V2.py
```

---

## Results (VGA/DVI socket)

| Metric | Value |
|---|---|
| Avg 2D Polygonal IoU | 88.5% |
| Exact 3D Volumetric IoU | 63.4% |

---

## Pipeline Architecture

```
src/data/test/  (reference images)
      │
      ├─► User selects reference image + socket instance
      │
Images + Poses
      │
      ├─► Panel Normal
      │     ├── Load from panel_normal.npy  (if cached)
      │     └── SIFT + DLT + RANSAC → save to panel_normal.npy
      │
      ├─► YOLO — all instances, all frames
      │        │
      │   Two-phase instance matching
      │   (consensus 3D triangulation → per-frame reprojection selection)
      │        │
      │   Coverage check  ──► warn / re-run / manual fallback
      │        │
      │   Reprojection box filter
      │        │
      ├─► SAM2 segmentation (per matched frame)
      │        │
      ├─► GPU Voxel Carving
      │        │
      └─► Panel-normal-aware OBB extraction (minAreaRect / PCA)
               │
      ┌────────┴──────────────────────────────┐
      │  <class>_run<N>.json                  │
      │  results_<N>.json  (consolidated)     │
      │  OBB projection images  (disk always) │
      │  OBB overlay on reference image       │
      └───────────────────────────────────────┘
```