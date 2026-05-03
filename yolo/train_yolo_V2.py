import os
import shutil
import random
import csv
from pathlib import Path
import torch
import gc
from ultralytics import YOLO

def distribute_data(base_path, source_name, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Distributes raw data into train/valid/test folders for conventional training."""
    print(f"[System] Distributing data into train/valid/test folders...")
    
    src_images = Path(base_path) / source_name / "images"
    src_labels = Path(base_path) / source_name / "labels"
    
    # Define target directories based on data.yaml requirements
    dirs = {
        "train": Path(base_path) / "train",
        "valid": Path(base_path) / "valid",
        "test": Path(base_path) / "test"
    }

    # Create directories
    for d in dirs.values():
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)

    # Get all image files and shuffle
    images = [f for f in os.listdir(src_images) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    # Calculate split indices
    train_end = int(len(images) * train_ratio)
    val_end = train_end + int(len(images) * val_ratio)

    splits = {
        "train": images[:train_end],
        "valid": images[train_end:val_end],
        "test": images[val_end:]
    }

    # Copy files to their respective folders
    for split_name, file_list in splits.items():
        for fname in file_list:
            # Copy Image
            shutil.copy(src_images / fname, dirs[split_name] / "images" / fname)
            # Copy Label (assuming same name with .txt)
            label_name = fname.rsplit('.', 1)[0] + ".txt"
            if (src_labels / label_name).exists():
                shutil.copy(src_labels / label_name, dirs[split_name] / "labels" / label_name)

    print(f"✅ Distribution complete: {len(splits['train'])} train, {len(splits['valid'])} valid, {len(splits['test'])} test.")

def create_kfold_splits(base_path, source_name, k=5):
    """Creates K-Fold split files (.txt) referencing the raw data directory."""
    print(f"[System] Preparing {k}-Fold splits from {source_name}...")
    src_images = Path(base_path) / source_name / "images"
    images = [str(src_images / f) for f in os.listdir(src_images) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    fold_size = len(images) // k
    folds = [images[i * fold_size:(i + 1) * fold_size if i < k - 1 else len(images)] for i in range(k)]
    
    fold_yamls = []
    for i in range(k):
        val_images = folds[i]
        train_images = [img for j, fold in enumerate(folds) if i != j for img in fold]

        train_txt, val_txt = Path(base_path) / f"fold_{i+1}_train.txt", Path(base_path) / f"fold_{i+1}_val.txt"
        with open(train_txt, 'w') as f: f.write('\n'.join(train_images))
        with open(val_txt, 'w') as f: f.write('\n'.join(val_images))

        yaml_path = Path(base_path) / f"data_fold_{i+1}.yaml"
        # Mapping for HDMI, PS2, USB, VGA, ethernet, and power sockets
        yaml_content = f"path: {base_path}\ntrain: {train_txt.name}\nval: {val_txt.name}\nnc: 7\nnames: ['HDMI_socket', 'PC_Panel', 'PS2_socket', 'USB_socket', 'VGA_socket', 'ethernet_socket', 'power_socket']"
        with open(yaml_path, 'w') as f: f.write(yaml_content)
        fold_yamls.append(yaml_path)
    return fold_yamls, images

def copy_best_weights(run_name_prefix, project_dir, base_dir):
    """
    Finds the most recently modified run folder whose name starts with
    run_name_prefix inside project_dir, then copies its best.pt to
    base_dir/weights/best.pt.
    Handles YOLO's auto-incrementing run names (e.g. yolo11n_conventional2).
    """
    project_path = Path(project_dir)
    candidates = sorted(
        [d for d in project_path.iterdir()
         if d.is_dir() and d.name.startswith(run_name_prefix)],
        key=lambda d: d.stat().st_mtime,
        reverse=True
    )
    if not candidates:
        print(f"[Weights] WARNING: no run folder starting with '{run_name_prefix}' found in {project_path}")
        return
    best_pt = candidates[0] / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[Weights] WARNING: best.pt not found in {candidates[0]}")
        return
    dst_dir = Path(base_dir) / "weights"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "best.pt"
    shutil.copy(best_pt, dst)
    print(f"[Weights] Copied {best_pt} → {dst}")


if __name__ == '__main__':
    BASE_DIR = os.getcwd() 
    
    mode = input("🚀 Select Mode: [c] Conventional Train/Val/Test or [k] K-Fold Cross-Val? ").lower().strip()
    device = 0 if torch.cuda.is_available() else 'cpu'

    if mode == 'c':
        # 1. Distribute data into folders required by data.yaml
        distribute_data(BASE_DIR, "all_training_data")
        
        # 2. Start Conventional Training
        print("\n[System] Starting Conventional Training with Cosine LR Scheduler...")
        model = YOLO('yolo11n.pt')
        model.train(
            data='data.yaml', 
            epochs=300,
            patience=0,
            imgsz=640,
            batch=8,
            device=device,
            project='port_metrology',
            name='yolo11n_conventional',
            cos_lr=True
        )
        copy_best_weights('yolo11n_conventional', Path(BASE_DIR) / 'runs' / 'detect' / 'port_metrology', BASE_DIR)

    elif mode == 'k':
        # 1. Prepare K-Fold YAMLs and text lists
        fold_yamls, all_images = create_kfold_splits(BASE_DIR, "all_training_data")
        
        # 2. Run K-Fold Loop
        for i, yaml_path in enumerate(fold_yamls):
            print(f"\n🚀 STARTING FOLD {i+1}/{len(fold_yamls)}")
            YOLO('yolo11n.pt').train(data=str(yaml_path), epochs=100, imgsz=640, batch=8, device=device, project='port_metrology', name=f'fold_{i+1}', exist_ok=True)
            torch.cuda.empty_cache()
            gc.collect()

        # 3. Final training on 100% of data
        final_txt = Path(BASE_DIR) / "final_train.txt"
        with open(final_txt, 'w') as f: f.write('\n'.join(all_images))
        
        final_yaml = Path(BASE_DIR) / "data_final.yaml"
        with open(final_yaml, 'w') as f:
            f.write(f"path: {BASE_DIR}\ntrain: {final_txt.name}\nval: {final_txt.name}\nnc: 7\nnames: ['HDMI_socket', 'PC_Panel', 'PS2_socket', 'USB_socket', 'VGA_socket', 'ethernet_socket', 'power_socket']")

        print("\n[System] Commencing Final Training on 100% data...")
        YOLO('yolo11n.pt').train(data=str(final_yaml), epochs=300, 
                         val=True,   # enables best.pt saving
                         device=device, project='port_metrology', 
                         name='yolo11n_ports_FINAL')
        copy_best_weights('yolo11n_ports_FINAL', Path(BASE_DIR) / 'runs' / 'detect' / 'port_metrology', BASE_DIR)

    print("\n🌟 METROLOGY PIPELINE TRAINING COMPLETE 🌟")