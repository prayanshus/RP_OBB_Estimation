# PC I/O Panel OBB Estimation Pipeline

Robotics Perception course project — IISc Sem 2.

Computes world-frame **Oriented Bounding Boxes (OBBs)** for sockets on a PC tower's I/O panel (VGA/DVI, Ethernet, Power, PS/2, HDMI, USB) from 16 posed images.

---

## Repository Structure

```
RP_OBB_Estimation/
├── data/                   # 16 input frames + intrinsic.json + poses.json
├── pipeline/
│   ├── auto_obb_sam2_cl_V5.py   # Main automated pipeline
│   ├── obb_tool_ge_V0.py        # Manual annotation fallback (Tkinter GUI)
│   └── yolo_detector.py         # YOLO inference wrapper
├── yolo/
│   ├── data.yaml                # Class definitions
│   ├── train_yolo_V2.py         # Training script
│   ├── eval_model_V3.py         # Evaluation script
│   ├── weights/best.pt          # Trained YOLOv11n weights (not tracked by git)
│   ├── train/                   # Training split
│   ├── valid/                   # Validation split
│   └── test/                    # Test split
├── sam2/checkpoints/            # SAM2 checkpoint (not tracked by git)
├── outputs/                     # Generated answers JSON
├── requirements.txt
└── requirements_wsl.txt         # Full WSL environment snapshot
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install SAM2
```bash
git clone https://github.com/facebookresearch/sam2.git
pip install -e sam2/
```
Download checkpoint:
```bash
mkdir -p sam2/checkpoints
wget -P sam2/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

### 3. Place YOLO weights
Copy `best.pt` to `yolo/weights/best.pt`.

### 4. Place data
Copy 16 frames + `intrinsic.json` + `poses.json` + `sample_answers.json` into `data/`.

---

## Running the Pipeline

```bash
cd pipeline
python auto_obb_sam2_cl_V5.py
```

Select the socket to measure when prompted. The pipeline will:
1. Estimate the panel normal via SIFT + RANSAC on PC_Panel detections
2. Detect the socket with YOLO, filter outlier boxes by reprojection
3. Segment with SAM2
4. Carve a voxel grid to find surviving 3D points
5. Extract OBB using panel-normal-aware PCA / minAreaRect

If fewer than 3 clean detections are found, the manual annotation tool launches automatically.

**Force manual mode:**
```bash
python auto_obb_sam2_cl_V5.py --manual_mode
```

---

## Training YOLO

```bash
cd yolo
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
Images + Poses
     │
     ├─► YOLO detection (all 16 frames)
     │        │
     │   Box filter (reprojection consensus)
     │        │
     ├─► PC_Panel SIFT → RANSAC → Panel Normal
     │        │
     ├─► SAM2 segmentation (per detection frame)
     │        │
     ├─► GPU Voxel Voting (3D carving)
     │        │
     └─► OBB extraction (minAreaRect / PCA)
              │
           answers.json
```