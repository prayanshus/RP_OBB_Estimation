import numpy as np
import cv2
from ultralytics import YOLO

def get_yolo_bounding_boxes(image_paths, target_class_name, model_path="runs/detect/train/weights/best.pt", conf_thresh=0.5):
    """
    Runs YOLO inference on all images and returns bounding boxes for the target class.
    
    Args:
        image_paths (dict): {fid: Path} dictionary.
        target_class_name (str): The string name of the class to find (e.g., "vga_socket").
        model_path (str): Path to your trained YOLOv10/v11 weights.
        conf_thresh (float): Minimum confidence threshold.
        
    Returns:
        dict: {fid: [x_min, y_min, x_max, y_max]} for the target object.
    """
    print(f"\n[YOLO] Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[YOLO ERROR] Could not load model: {e}")
        print("Please ensure you have trained the model and the path is correct.")
        return {}

    # Check if target class exists in the model's training data
    class_names = model.names
    name_to_id = {v: k for k, v in class_names.items()}
    
    if target_class_name not in name_to_id:
        raise ValueError(f"Class '{target_class_name}' not found in YOLO model. Available classes: {list(name_to_id.keys())}")
        
    target_class_id = name_to_id[target_class_name]
    
    print(f"[YOLO] Hunting for '{target_class_name}' in {len(image_paths)} images...")
    
    user_boxes = {}
    
    for fid, img_path in image_paths.items():
        # Run inference
        results = model(str(img_path), verbose=False, conf=conf_thresh)
        
        # Parse results for this specific image
        boxes = results[0].boxes
        
        best_box = None
        highest_conf = -1.0
        
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if cls_id == target_class_id:
                # TODO: Multi-Instance Matching. 
                # If there are 4 USB ports, we currently just take the one with the highest confidence.
                # In the future, we will need to return ALL boxes and use Epipolar Geometry to match them.
                if conf > highest_conf:
                    highest_conf = conf
                    # YOLO returns [x_center, y_center, width, height] or [x1, y1, x2, y2]. 
                    # xyxy is exactly what SAM 2 wants.
                    best_box = box.xyxy[0].cpu().numpy().tolist() 
                    
        if best_box is not None:
            user_boxes[fid] = [int(v) for v in best_box]
            print(f"  -> Found in Frame {fid} (Conf: {highest_conf:.2f})")
        else:
            # The object is simply not in this image, or below threshold. We safely skip it.
            pass 
            
    print(f"[YOLO] Extraction complete. Found {target_class_name} in {len(user_boxes)}/{len(image_paths)} frames.")
    return user_boxes