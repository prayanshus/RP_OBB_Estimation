import os
import cv2
import glob
import pandas as pd
import numpy as np
from ultralytics import YOLO

def clean_filename(filename):
    """Removes the Roboflow .rf. hash and intermediate extension markers."""
    if ".rf." in filename:
        base = filename.split(".rf.")[0]
        extension = filename.split(".")[-1]
        if base.endswith("_png") or base.endswith("_jpg"):
            base = base[:-4]
        return f"{base}.{extension}"
    return filename

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) for two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def get_ground_truth(label_path, img_w, img_h):
    """Reads YOLO format labels and converts to pixel coordinates."""
    gt_data = []
    if not os.path.exists(label_path):
        return gt_data
    with open(label_path, 'r') as f:
        for line in f:
            cls, x, y, w, h = map(float, line.split())
            x1 = (x - w/2) * img_w
            y1 = (y - h/2) * img_h
            x2 = (x + w/2) * img_w
            y2 = (y + h/2) * img_h
            gt_data.append({'cls': int(cls), 'box': [x1, y1, x2, y2]})
    return gt_data

def get_latest_weights(base_dir='runs/detect/port_metrology', prefix='yolo11n_conventional'):
    """Dynamically finds the most recent training run to avoid FileNotFoundError."""
    search_pattern = os.path.join(base_dir, f"{prefix}*")
    runs = glob.glob(search_pattern)
    if not runs:
        return None
    # Sort runs by modification time to grab the absolute latest one
    latest_run = max(runs, key=os.path.getmtime)
    return os.path.join(latest_run, 'weights', 'best.pt')

# --- Configuration ---
# Pointing to the root directory containing ALL images
EVAL_DIR = 'all_training_data/images'

# Mapped exactly to the provided data.yaml file
TARGETS = {
    0: "HDMI_socket",
    1: "PC_Panel",
    2: "PS2_socket",
    3: "USB_socket",
    4: "VGA_socket",
    5: "ethernet_socket",
    6: "power_socket"
}

# --- Initialization ---
MODEL_PATH = get_latest_weights()

if MODEL_PATH is None or not os.path.exists(MODEL_PATH):
    print("\n❌ Error: Could not find model weights. Make sure you have run the training script successfully.")
    exit()

print(f"\n✅ Automatically loading weights from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Fetch all image paths directly from the comprehensive directory
image_paths = [os.path.join(EVAL_DIR, f) for f in os.listdir(EVAL_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

all_results = []

for img_path in image_paths:
    raw_name = os.path.basename(img_path)
    clean_name = clean_filename(raw_name)
    
    img = cv2.imread(img_path)
    if img is None: continue
    h, w, _ = img.shape
    
    # Locate label file by replacing \images\ with \labels\ 
    label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
    gt_boxes = get_ground_truth(label_path, w, h)
    
    # Run Inference
    preds = model(img_path, verbose=False)[0]
    
    img_entry = {"Image Name": clean_name}
    
    for cls_idx, cls_label in TARGETS.items():
        # Get ground truth and predictions for this specific class
        class_gt = [g['box'] for g in gt_boxes if g['cls'] == cls_idx]
        class_preds = [
            {'box': b.xyxy[0].tolist(), 'conf': float(b.conf)} 
            for b in preds.boxes if int(b.cls) == cls_idx
        ]

        if class_gt:
            ious = []
            confs = []
            for g_box in class_gt:
                # Find the best matching predicted box for this GT box
                if class_preds:
                    match_data = [(calculate_iou(g_box, p['box']), p['conf']) for p in class_preds]
                    best_match_iou, best_match_conf = max(match_data, key=lambda x: x[0])
                    ious.append(best_match_iou)
                    confs.append(best_match_conf)
                else:
                    ious.append(0.0)
                    confs.append(0.0)
            
            img_entry[f"{cls_label}_IOU"] = round(np.mean(ious), 3)
            img_entry[f"{cls_label}_Conf"] = round(np.mean(confs), 3)
            img_entry[f"{cls_label}_Detected"] = 1 if any(i > 0.0 for i in ious) else 0
        else:
            # If the class is not in the ground truth for this image
            img_entry[f"{cls_label}_IOU"] = "N/A"
            img_entry[f"{cls_label}_Conf"] = "N/A"
            img_entry[f"{cls_label}_Detected"] = 0

    all_results.append(img_entry)

# --- Output Generation ---
df = pd.DataFrame(all_results)
output_filename = "eval_report_V3.txt"

with open(output_filename, "w") as f:
    f.write("### 1. Detection Summary (All Images)\n")
    f.write("---\n")
    for cls_label in TARGETS.values():
        count = df[df[f"{cls_label}_Detected"] == 1].shape[0]
        f.write(f"* **{cls_label}**: Detected in {count} images\n")

    f.write("\n### 2. Detailed Performance Table\n")
    # Selecting specific columns for the final display
    cols_to_show = ["Image Name"]
    for name in TARGETS.values():
        cols_to_show.extend([f"{name}_IOU", f"{name}_Conf"])

    f.write(df[cols_to_show].to_string(index=False) + "\n")

print(f"\n✅ Evaluation complete. Results written to {output_filename}")