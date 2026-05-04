import json
import gc
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from scipy.spatial import ConvexHull

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from yolo_detector import get_yolo_bounding_boxes

# ==============================================================================
# Per-socket configuration
# conf_threshold : minimum YOLO confidence to accept a detection
# depth_prior_mm : physical depth used when carving cannot resolve it
# ==============================================================================
SOCKET_CONFIG = {
    "VGA_socket":      {"depth_prior_mm": 6.1, "conf_threshold": 0.25, "percentile": 1, "use_minarearect": True},
    "ethernet_socket": {"depth_prior_mm": 1.0, "conf_threshold": 0.45, "percentile": 1, "use_minarearect": False},
    "power_socket":    {"depth_prior_mm": 2.0, "conf_threshold": 0.25, "percentile": 2.5, "use_minarearect": False},
    "PS2_socket":      {"depth_prior_mm": 1.0, "conf_threshold": 0.25, "percentile": 1, "use_minarearect": False},
    "HDMI_socket":     {"depth_prior_mm": 1.0, "conf_threshold": 0.25, "percentile": 1, "use_minarearect": False},
    "USB_socket":      {"depth_prior_mm": 1.0, "conf_threshold": 0.25, "percentile": 1, "use_minarearect": False},
}


# ==============================================================================
# 0. Geometry Utilities
# ==============================================================================
def triangulate_dlt(observations, K, poses):
    """DLT triangulation from 2D observations {(frame_id, u, v)} to a 3D point."""
    A = []
    for fid, u, v in observations:
        T_w2c = np.linalg.inv(poses[fid])
        P = K @ T_w2c[:3]
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]

def filter_boxes_by_reprojection(boxes, K, poses, tol_factor=1.0):
    """
    Removes YOLO detections whose box centre is inconsistent with the consensus
    3D position triangulated from all detections.

    Steps:
      1. Triangulate a rough 3D centre from all boxes (DLT).
      2. Project that centre back into each frame.
      3. Compute distance from projected point to detected box centre,
         normalised by the box diagonal (scale-independent tolerance).
      4. Reject boxes where normalised distance > tol_factor.
      5. Re-triangulate with surviving boxes only.

    tol_factor : reject if reprojection error > tol_factor × box diagonal.
                 1.0 means the centre must be within one box-diagonal of where
                 the consensus 3D point projects. Lower = stricter.
    """
    if len(boxes) < 2:
        return boxes

    box_centers = [(fid, (b[0]+b[2])/2, (b[1]+b[3])/2) for fid, b in boxes.items()]
    center_3d   = triangulate_dlt(box_centers, K, poses)

    filtered = {}
    for fid, box in boxes.items():
        if fid not in poses:
            continue
        x1, y1, x2, y2 = box
        T_w2c   = np.linalg.inv(poses[fid])
        pt_cam  = T_w2c[:3, :3] @ center_3d + T_w2c[:3, 3]
        if pt_cam[2] <= 0:
            continue
        u_proj  = K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]
        v_proj  = K[1, 1] * pt_cam[1] / pt_cam[2] + K[1, 2]

        u_det   = (x1 + x2) / 2
        v_det   = (y1 + y2) / 2
        diag    = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        err     = np.sqrt((u_proj - u_det)**2 + (v_proj - v_det)**2)
        norm_err = err / diag if diag > 0 else float("inf")

        if norm_err <= tol_factor:
            filtered[fid] = box
            print(f"  [BoxFilter] Frame {fid}: norm_err={norm_err:.2f} ✓")
        else:
            print(f"  [BoxFilter] Frame {fid}: norm_err={norm_err:.2f} — REJECTED (>{tol_factor})")

    print(f"[BoxFilter] {len(filtered)}/{len(boxes)} boxes passed.")
    return filtered



def calculate_exact_3d_iou(obb_calc, obb_gt):
    """Exact 3D IoU via convex-hull intersection of OBB vertices."""
    c_A, e_A, r_A = (np.array(obb_calc["center"]),
                     np.array(obb_calc["extent"]),
                     np.array(obb_calc["rotation"]))
    c_B, e_B, r_B = (np.array(obb_gt["obb"]["center"]),
                     np.array(obb_gt["obb"]["extent"]),
                     np.array(obb_gt["obb"]["rotation"]))

    vol_A = e_A[0] * e_A[1] * e_A[2]
    vol_B = e_B[0] * e_B[1] * e_B[2]

    def get_obb_vertices(c, e, r):
        x, y, z = e / 2
        corners = np.array([[-x,-y,-z],[x,-y,-z],[-x,y,-z],[x,y,-z],
                             [-x,-y, z],[x,-y, z],[-x,y, z],[x,y, z]])
        return (r @ corners.T).T + c

    def get_obb_edges(v):
        return [(v[i], v[j]) for i, j in
                [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]]

    def get_obb_planes(c, e, r):
        planes = []
        for i in range(3):
            normal = r[:, i]
            planes.extend([(normal,  c + normal*(e[i]/2)),
                           (-normal, c - normal*(e[i]/2))])
        return planes

    def point_in_obb(p, c, e, r):
        return np.all(np.abs(r.T @ (p - c)) <= (e / 2) + 1e-6)

    def line_plane_intersection(p1, p2, plane_n, plane_p):
        u = p2 - p1
        dot = np.dot(plane_n, u)
        if abs(dot) > 1e-6:
            fac = -np.dot(plane_n, p1 - plane_p) / dot
            if 0 <= fac <= 1:
                return p1 + u * fac
        return None

    v_A = get_obb_vertices(c_A, e_A, r_A)
    v_B = get_obb_vertices(c_B, e_B, r_B)
    pts = []
    pts.extend([v for v in v_A if point_in_obb(v, c_B, e_B, r_B)])
    pts.extend([v for v in v_B if point_in_obb(v, c_A, e_A, r_A)])
    for p1, p2 in get_obb_edges(v_A):
        for n, p in get_obb_planes(c_B, e_B, r_B):
            inter = line_plane_intersection(p1, p2, n, p)
            if inter is not None and point_in_obb(inter, c_B, e_B, r_B):
                pts.append(inter)
    for p1, p2 in get_obb_edges(v_B):
        for n, p in get_obb_planes(c_A, e_A, r_A):
            inter = line_plane_intersection(p1, p2, n, p)
            if inter is not None and point_in_obb(inter, c_A, e_A, r_A):
                pts.append(inter)

    if len(pts) < 4:
        return 0.0
    try:
        intersect_vol = ConvexHull(np.unique(pts, axis=0)).volume
        union_vol = vol_A + vol_B - intersect_vol
        return intersect_vol / union_vol if union_vol > 0 else 0.0
    except Exception as e:
        print(f"[3D IoU] ConvexHull failed ({len(pts)} pts): {e}")
        return 0.0


def calculate_2d_projection_iou(K, poses, obb_calc, obb_gt, image_paths, user_boxes):
    """2D polygonal IoU by projecting OBB corners into each detection frame."""
    def get_8_corners(obb):
        c, e, r = (np.array(obb["center"]),
                   np.array(obb["extent"]),
                   np.array(obb["rotation"]))
        corners = np.array([[x*e[0]/2, y*e[1]/2, z*e[2]/2]
                             for x in [-1,1] for y in [-1,1] for z in [-1,1]])
        return (r @ corners.T).T + c

    corners_calc = get_8_corners(obb_calc)
    corners_gt   = get_8_corners(obb_gt["obb"])

    ious = {}
    for fid, img_path in image_paths.items():
        if fid not in poses or fid not in user_boxes:
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        T_w2c = np.linalg.inv(poses[fid])
        R_cam, t_cam = T_w2c[:3, :3], T_w2c[:3, 3]

        def project_to_mask(corners):
            cam_pts = (R_cam @ corners.T).T + t_cam
            uv = (K @ cam_pts.T).T
            u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
            pts = np.vstack((u, v)).T.astype(np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [cv2.convexHull(pts)], 1)
            return mask

        mask_calc, mask_gt = project_to_mask(corners_calc), project_to_mask(corners_gt)
        intersection = (mask_calc & mask_gt).sum()
        union        = (mask_calc | mask_gt).sum()
        ious[fid]    = intersection / union if union > 0 else 0

    return ious


# ==============================================================================
# 1. SAM2 Integration & Visualization
# ==============================================================================
def generate_sam_masks(image_paths, bounding_boxes, model_cfg, checkpoint_path):
    """Run SAM2 on all frames that have a bounding box. Returns {fid: mask_tensor}."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[SAM 2] Loading model on {device}...")
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor  = SAM2ImagePredictor(sam2_model)

    masks = {}
    for fid, box in bounding_boxes.items():
        if fid not in image_paths:
            continue
        print(f"[SAM 2] Segmenting Frame {fid}...")
        img_bgr = cv2.imread(str(image_paths[fid]))
        predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                masks_np, _, _ = predictor.predict(
                    box=np.array(box)[None, :], multimask_output=False)
        else:
            masks_np, _, _ = predictor.predict(
                box=np.array(box)[None, :], multimask_output=False)
        masks[fid] = torch.tensor(masks_np, dtype=torch.float32, device=device).unsqueeze(0)

    print("[System] Flushing SAM2 from VRAM...")
    del predictor, sam2_model
    gc.collect()
    torch.cuda.empty_cache()
    return masks


def display_sam_masks(image_paths, masks, label="Socket", bboxes=None):
    """
    Overlay SAM2 masks on frames and display one at a time. Press any key to advance.
    bboxes : optional {fid: (x1,y1,x2,y2)} draws the YOLO box in cyan over the mask.
    """
    print(f"\n[Review] Displaying {label} SAM2 masks. Press any key for next frame.")
    for fid, mask_tensor in masks.items():
        if fid not in image_paths:
            continue
        img     = cv2.imread(str(image_paths[fid]))
        overlay = img.copy()
        overlay[mask_tensor.squeeze().cpu().numpy() > 0.5] = [0, 255, 0]
        blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        if bboxes is not None and fid in bboxes:
            x1, y1, x2, y2 = [int(v) for v in bboxes[fid]]
            cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(blended, "YOLO box", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        scale = min(1000 / blended.shape[1], 800 / blended.shape[0])
        disp  = cv2.resize(blended, (0, 0), fx=scale, fy=scale)
        cv2.putText(disp, f"{label} — Frame {fid} | green=SAM2  cyan=YOLO  (any key)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("SAM2 Mask Review", disp)
        cv2.waitKey(0)
    cv2.destroyAllWindows()



def erode_masks(masks, kernel_size=3):
    """
    Morphologically erodes each SAM2 mask by kernel_size pixels.
    Shrinks the mask boundary inward, reducing boundary leakage into
    voxel voting. kernel_size must be a positive odd integer.
    """
    if kernel_size < 1:
        return masks
    kernel  = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded  = {}
    device  = next(iter(masks.values())).device
    for fid, mask_tensor in masks.items():
        m_np = (mask_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        m_np = cv2.erode(m_np, kernel, iterations=1)
        eroded[fid] = torch.tensor(
            m_np[None, None].astype(np.float32), device=device)
    print(f"[MaskErode] Applied {kernel_size}×{kernel_size} erosion to "
          f"{len(eroded)} masks.")
    return eroded

# ==============================================================================
# 2. Open3D OBB Visualization
# ==============================================================================
def display_3d_obbs(calc_obb, expected_obb, point_cloud):
    print("\n[Visualization] Opening 3D OBB viewer...")
    measured_o3d = o3d.geometry.OrientedBoundingBox(
        calc_obb["center"], calc_obb["rotation"], calc_obb["extent"])
    measured_o3d.color = (1.0, 0.0, 0.0)

    gt = expected_obb["obb"]
    expected_o3d = o3d.geometry.OrientedBoundingBox(
        gt["center"], gt["rotation"], gt["extent"])
    expected_o3d.color = (0.0, 1.0, 0.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color([0.0, 0.5, 1.0])

    o3d.visualization.draw_geometries(
        [measured_o3d, expected_o3d, pcd],
        window_name="OBB Intersection (Red=Calc, Green=GT)",
        width=1280, height=720, mesh_show_wireframe=True)

def display_obb_projections(K, poses, obb_calc, obb_gt, image_paths, user_boxes,
                            save_dir=None, show=True):
    """
    Projects OBBs onto each detection frame as proper 3D wireframe boxes with
    coloured axes (W=red, H=green, D=blue). Calc=magenta, GT=yellow.
    Displays at 1.5× upscale for clarity (show=True). Saves to save_dir if provided.
    """
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    # OBB corner indices — 8 corners ordered as xyz sign combos
    # Corner layout: bit0=x, bit1=y, bit2=z  (0=minus, 1=plus)
    EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

    def get_8_corners(obb_dict):
        c = np.array(obb_dict["center"])
        e = np.array(obb_dict["extent"])
        r = np.array(obb_dict["rotation"])
        signs = np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]],
                         dtype=float)
        return (r @ (signs * e / 2).T).T + c

    def project_pt(pt_world, T_w2c):
        p = T_w2c[:3,:3] @ pt_world + T_w2c[:3,3]
        if p[2] <= 0:
            return None
        uv = K @ p
        return (int(uv[0]/uv[2]), int(uv[1]/uv[2]))

    def draw_wireframe(img, corners_world, T_w2c, color, thickness=2):
        pts = [project_pt(c, T_w2c) for c in corners_world]
        for i, j in EDGES:
            if pts[i] and pts[j]:
                cv2.line(img, pts[i], pts[j], color, thickness, cv2.LINE_AA)

    def draw_axes(img, obb_dict, T_w2c, scale=1.0):
        """Draw W(red) H(green) D(blue) axes from OBB centre."""
        c = np.array(obb_dict["center"])
        r = np.array(obb_dict["rotation"])
        e = np.array(obb_dict["extent"])
        origin = project_pt(c, T_w2c)
        if origin is None:
            return
        axis_colors = [(0,0,255),(0,255,0),(255,0,0)]   # W=red, H=green, D=blue
        axis_labels = ["W","H","D"]
        for i in range(3):
            tip = project_pt(c + r[:,i] * e[i] * 0.5 * scale, T_w2c)
            if tip:
                cv2.arrowedLine(img, origin, tip, axis_colors[i], 2,
                                cv2.LINE_AA, tipLength=0.2)
                cv2.putText(img, axis_labels[i],
                            (tip[0]+4, tip[1]-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_colors[i], 2)

    corners_calc = get_8_corners(obb_calc)
    corners_gt   = get_8_corners(obb_gt["obb"]) if obb_gt else None

    print("\n[Projection] Displaying 3D OBB projections. Press any key for next frame.")
    for fid in user_boxes:
        if fid not in image_paths or fid not in poses:
            continue
        img   = cv2.imread(str(image_paths[fid]))
        T_w2c = np.linalg.inv(poses[fid])

        # Draw GT first (underneath) then calc on top
        if corners_gt is not None:
            draw_wireframe(img, corners_gt, T_w2c, (0, 220, 255), thickness=2)  # yellow
        draw_wireframe(img, corners_calc, T_w2c, (255, 0, 200), thickness=2)    # magenta
        draw_axes(img, obb_calc, T_w2c, scale=1.5)

        # Upscale to 1.5× for clarity, cap at 1440p
        scale = min(1440 / img.shape[1], 1080 / img.shape[0])
        disp  = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        legend = "magenta=Calc  yellow=GT  |  W=red H=green D=blue" \
                 if corners_gt is not None else "magenta=Calc  |  W=red H=green D=blue"
        cv2.putText(disp, f"Frame {fid} | {legend}  (any key)",
                    (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        if save_dir is not None:
            out_path = Path(save_dir) / f"frame_{fid}.png"
            cv2.imwrite(str(out_path), disp)
        if show:
            cv2.imshow("OBB 3D Projection", disp)
            cv2.waitKey(0)
    if show:
        cv2.destroyAllWindows()



# ==============================================================================
# 3. GPU Voxel Voting (cubic grid + YOLO gate)
# ==============================================================================
def _angular_weights(poses, frame_ids):
    """
    Computes per-frame weights based on angular diversity.
    Each frame's weight = mean angular distance (radians) to all other frames.
    Frames with a more unique viewpoint get higher weight.
    Weights are normalised so they sum to len(frame_ids).
    """
    dirs = []
    for fid in frame_ids:
        col2 = np.array(poses[fid])[:3, 2]   # camera forward direction in world
        dirs.append(col2 / np.linalg.norm(col2))
    dirs = np.array(dirs)
    n    = len(dirs)
    weights = np.zeros(n)
    for i in range(n):
        cos_vals       = np.clip(dirs @ dirs[i], -1.0, 1.0)
        cos_vals[i]    = 1.0   # exclude self (angle=0)
        angles         = np.arccos(cos_vals)
        weights[i]     = angles.sum() / (n - 1) if n > 1 else 1.0
    # Normalise so weights sum to n (preserves scale of vote counts)
    weights = weights / weights.mean()
    return {fid: float(w) for fid, w in zip(frame_ids, weights)}


def voxel_voting_gpu(K, poses, masks, center_3d, user_boxes=None,
                     size_mm=50, res_mm=0.5, consensus_ratio=0.75):
    """
    Cubic voxel carving using SAM2 masks.
    user_boxes        : optional {fid: (x1,y1,x2,y2)} — YOLO gate per frame.
    size_mm           : half-extent of the cube in mm.
    res_mm            : voxel resolution in mm.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K_t    = torch.tensor(K, dtype=torch.float32, device=device)
    steps  = int(size_mm * 2 / res_mm)
    ax     = torch.linspace(-size_mm/1000, size_mm/1000, steps, device=device)
    X, Y, Z = torch.meshgrid(ax, ax, ax, indexing='ij')
    points_3d = torch.stack([
        X.flatten() + center_3d[0],
        Y.flatten() + center_3d[1],
        Z.flatten() + center_3d[2]], dim=0)
    votes = torch.zeros(points_3d.shape[1], dtype=torch.int32, device=device)
    print(f"[Voxel Voter] Grid: {steps}³ = {points_3d.shape[1]:,} voxels | "
          f"centre {center_3d.round(3)}")

    for fid, mask_tensor in masks.items():
        if fid not in poses:
            continue
        T_w2c    = torch.linalg.inv(torch.tensor(poses[fid], dtype=torch.float32, device=device))
        R, t     = T_w2c[:3, :3], T_w2c[:3, 3].unsqueeze(1)
        points_c = R @ points_3d + t
        Z_c      = points_c[2, :]
        uv_homog = K_t @ points_c
        u_px     = uv_homog[0, :] / Z_c
        v_px     = uv_homog[1, :] / Z_c
        _, _, H_img, W_img = mask_tensor.shape
        uv_grid  = torch.stack(
            ((u_px / (W_img - 1)) * 2 - 1,
             (v_px / (H_img - 1)) * 2 - 1), dim=-1).view(1, 1, -1, 2)
        sampled  = F.grid_sample(mask_tensor, uv_grid, mode='nearest',
                                 padding_mode='zeros', align_corners=True)
        sam_vote = (Z_c > 0) & (sampled.view(-1) > 0.5)

        if user_boxes is not None and fid in user_boxes:
            x1, y1, x2, y2 = user_boxes[fid]
            yolo_gate = (u_px >= x1) & (u_px <= x2) & (v_px >= y1) & (v_px <= y2)
            votes += (sam_vote & yolo_gate).int()
        else:
            votes += sam_vote.int()

    survivors = points_3d[:, votes >= int(len(masks) * consensus_ratio)].T.cpu().numpy()
    print(f"[Voxel Voter] Carving complete. {len(survivors):,} voxels survived.")
    return survivors


# ==============================================================================
# 4. Panel Normal Estimation via PC_Panel SAM2 + SVD
# ==============================================================================
def svd_plane_normal(point_cloud, reference_direction=None):
    """
    Fit a plane to point_cloud via SVD and return the unit normal.
    If reference_direction is provided, the normal is flipped to point in the
    same hemisphere (ensures it always points outward toward the cameras).
    """
    centroid = np.mean(point_cloud, axis=0)
    _, _, Vt = np.linalg.svd(point_cloud - centroid, full_matrices=False)
    normal   = Vt[-1]   # row corresponding to smallest singular value
    if reference_direction is not None and np.dot(normal, reference_direction) < 0:
        normal = -normal
    return normal / np.linalg.norm(normal)


def ransac_plane_fit(points, n_iter=500, threshold=0.005):
    """
    RANSAC plane fit on a set of 3D points.
    threshold : inlier distance in metres (default 5 mm).
    Returns (unit_normal, n_inliers) — normal is unoriented.
    """
    best_normal, best_count = None, 0
    n = len(points)
    for _ in range(n_iter):
        idx = np.random.choice(n, 3, replace=False)
        v1  = points[idx[1]] - points[idx[0]]
        v2  = points[idx[2]] - points[idx[0]]
        crs = np.cross(v1, v2)
        if np.linalg.norm(crs) < 1e-10:
            continue
        normal = crs / np.linalg.norm(crs)
        dists  = np.abs((points - points[idx[0]]) @ normal)
        count  = int(np.sum(dists < threshold))
        if count > best_count:
            best_count, best_normal = count, normal

    # Refit SVD on all inliers for a more accurate final normal
    inliers = points[np.abs((points - np.mean(points, axis=0)) @ best_normal) < threshold]
    return svd_plane_normal(inliers), best_count


def estimate_panel_normal(image_paths, poses, K, panel_boxes):
    """
    Estimates the panel outward normal via:
      1. SIFT keypoint detection within each PC_Panel YOLO bounding box.
      2. Brute-force matching across all frame pairs (Lowe ratio test).
      3. DLT triangulation of each good match → 3D point on the panel.
      4. RANSAC plane fit on all triangulated points → panel normal.

    Falls back to mean camera viewing direction if too few points are found.
    """
    print("\n[Panel Normal] Estimating via SIFT + triangulation + RANSAC...")
    sift = cv2.SIFT_create()
    bf   = cv2.BFMatcher()

    # Detect SIFT within each bounding box, shift coords to full-image frame
    kp_data = {}
    for fid, box in panel_boxes.items():
        img = cv2.imread(str(image_paths[fid]))
        x1, y1, x2, y2 = [int(v) for v in box]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2]
        kps, descs = sift.detectAndCompute(gray, None)
        if descs is None or len(kps) == 0:
            continue
        pts = np.array([kp.pt for kp in kps]) + np.array([x1, y1])
        kp_data[fid] = (pts, descs)

    print(f"[Panel Normal] SIFT found in {len(kp_data)}/{len(panel_boxes)} frames.")

    # Match every pair of frames, triangulate each good match
    points_3d = []
    frame_ids = list(kp_data.keys())
    for i in range(len(frame_ids)):
        for j in range(i + 1, len(frame_ids)):
            fid_a, fid_b = frame_ids[i], frame_ids[j]
            pts_a, descs_a = kp_data[fid_a]
            pts_b, descs_b = kp_data[fid_b]
            matches = bf.knnMatch(descs_a, descs_b, k=2)
            for m, n_match in matches:
                if m.distance > 0.7 * n_match.distance:
                    continue
                ua, va = pts_a[m.queryIdx]
                ub, vb = pts_b[m.trainIdx]
                try:
                    pt3d = triangulate_dlt(
                        [(fid_a, ua, va), (fid_b, ub, vb)], K, poses)
                    points_3d.append(pt3d)
                except Exception:
                    continue

    points_3d = np.array(points_3d)
    print(f"[Panel Normal] {len(points_3d)} 3D points triangulated from SIFT matches.")

    # Fallback if too few points
    if len(points_3d) < 10:
        print("[Panel Normal] Too few points — falling back to mean viewing direction.")
        panel_centers = [(fid, (b[0]+b[2])/2, (b[1]+b[3])/2)
                         for fid, b in panel_boxes.items()]
        panel_center = triangulate_dlt(panel_centers, K, poses)
        cam_mean     = np.mean([np.array(poses[fid])[:3, 3]
                                for fid in panel_boxes if fid in poses], axis=0)
        ref = cam_mean - panel_center
        return ref / np.linalg.norm(ref)

    normal, n_inliers = ransac_plane_fit(points_3d)

    # Orient outward: flip if pointing away from cameras
    panel_center = np.mean(points_3d, axis=0)
    cam_mean     = np.mean([np.array(poses[fid])[:3, 3]
                            for fid in panel_boxes if fid in poses], axis=0)
    if np.dot(normal, cam_mean - panel_center) < 0:
        normal = -normal

    print(f"[Panel Normal] RANSAC: {n_inliers}/{len(points_3d)} inliers. "
          f"Normal: {normal.round(4)}")
    return normal


# ==============================================================================
# 5. OBB Extraction via Panel-Normal-Aware PCA
# ==============================================================================
def extract_obb_pca(point_cloud, depth_prior_mm=6.1, panel_normal=None, percentile=1, use_minarearect=True):
    """
    Extracts an OBB from point_cloud.

    panel_normal : (3,) unit vector of the panel outward normal.
                   When provided, d_hat is set directly to panel_normal —
                   no eigenvector selection. Points are projected onto the
                   plane perpendicular to d_hat and cv2.minAreaRect finds
                   the in-plane W/H axes. This is robust for any cloud shape
                   including nearly-square sockets where PCA eigenvalues are
                   degenerate.

                   Without panel_normal, falls back to PCA cross-product fix.
    depth_prior_mm : physical depth, always used in place of any depth extent.
    """
    centroid     = np.mean(point_cloud, axis=0)
    centered_pts = point_cloud - centroid

    if panel_normal is not None:
        # Set depth axis directly — no eigenvector voting
        d_hat = np.array(panel_normal, dtype=float)
        d_hat /= np.linalg.norm(d_hat)

        # Build two orthonormal in-plane basis vectors anchored to world up.
        # Using world_up=[0,0,1] (vertical in world frame) keeps u_hat horizontal
        # and v_hat vertical on the panel face — matching the physical socket edges.
        world_up = np.array([0, 0, 1], dtype=float)
        if abs(np.dot(d_hat, world_up)) > 0.9:   # normal is nearly vertical — use world Y
            world_up = np.array([0, 1, 0], dtype=float)
        u_hat = np.cross(d_hat, world_up);  u_hat /= np.linalg.norm(u_hat)
        v_hat = np.cross(d_hat, u_hat)      # already unit length

        # Project voxels onto the in-plane basis → 2D point cloud
        pts2d = np.column_stack([centered_pts @ u_hat,
                                 centered_pts @ v_hat]).astype(np.float32)

        # Orientation: minAreaRect for rotated sockets (e.g. VGA),
        # u_hat/v_hat directly for panel-aligned sockets (e.g. power, USB).
        if use_minarearect:
            rect      = cv2.minAreaRect(pts2d)
            _, _, angle = rect
            angle_rad = np.deg2rad(angle)
            w_hat_2d  = np.array([ np.cos(angle_rad),  np.sin(angle_rad)])
            h_hat_2d  = np.array([-np.sin(angle_rad),  np.cos(angle_rad)])
        else:
            # u_hat/v_hat are already aligned to the panel horizontal/vertical
            w_hat_2d = np.array([1.0, 0.0])
            h_hat_2d = np.array([0.0, 1.0])

        # Lift 2D in-plane axes back to 3D world frame
        w_hat_3d = w_hat_2d[0] * u_hat + w_hat_2d[1] * v_hat
        h_hat_3d = h_hat_2d[0] * u_hat + h_hat_2d[1] * v_hat

        # Step 3: project all points onto the found axes and apply percentile
        # clipping to get extents. This is where boundary leakage is trimmed.
        proj_w = pts2d @ w_hat_2d
        proj_h = pts2d @ h_hat_2d
        pct    = percentile  # trim pct% each side — set via SOCKET_CONFIG
        lo_w, hi_w = np.percentile(proj_w, pct), np.percentile(proj_w, 100 - pct)
        lo_h, hi_h = np.percentile(proj_h, pct), np.percentile(proj_h, 100 - pct)
        W = float(hi_w - lo_w)
        H = float(hi_h - lo_h)

        # Convention: W >= H
        if W < H:
            W, H = H, W
            w_hat_3d, h_hat_3d = h_hat_3d, w_hat_3d
            lo_w, hi_w, lo_h, hi_h = lo_h, hi_h, lo_w, hi_w

        # Assemble right-handed rotation matrix [W, H, D]
        axes = np.column_stack([w_hat_3d, h_hat_3d, d_hat])
        if np.linalg.det(axes) < 0:
            axes[:, 1] = -axes[:, 1]

        # Centre from percentile midpoints + depth midpoint
        mid_w        = (lo_w + hi_w) / 2
        mid_h        = (lo_h + hi_h) / 2
        centre_shift = (mid_w * w_hat_2d[0] + mid_h * h_hat_2d[0]) * u_hat +                        (mid_w * w_hat_2d[1] + mid_h * h_hat_2d[1]) * v_hat
        depth_proj   = centered_pts @ d_hat
        depth_mid    = (np.percentile(depth_proj, pct) + np.percentile(depth_proj, 100 - pct)) / 2
        true_center  = centroid + centre_shift + depth_mid * d_hat
        D = depth_prior_mm / 1000.0

    else:
        # Fallback: full PCA + cross-product depth axis
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered_pts, rowvar=False))
        sort_idx = np.argsort(eigenvalues)[::-1]
        axes     = eigenvectors[:, sort_idx].copy()
        d_hat    = np.cross(axes[:, 0], axes[:, 1])
        axes[:, 2] = d_hat / np.linalg.norm(d_hat)

        projections = centered_pts @ axes
        lo = np.percentile(projections, 1, axis=0)
        hi = np.percentile(projections, 99, axis=0)
        W  = float(hi[0] - lo[0])
        H  = float(hi[1] - lo[1])
        D  = depth_prior_mm / 1000.0
        true_center = centroid + axes @ ((hi + lo) / 2.0)

    return {"center": true_center.tolist(), "extent": [W, H, D], "rotation": axes.tolist()}


# ==============================================================================
# Main Execution
# ==============================================================================

def get_next_run_number(outputs_dir, class_name):
    """
    Returns the next unused run number for a given class by scanning
    outputs/ for existing <class_name>_run<N>.json files.
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in outputs_dir.glob(f"{class_name}_run*.json")]
    if not existing:
        return 1
    nums = []
    for p in existing:
        stem = p.stem  # e.g. VGA_socket_run3
        try:
            nums.append(int(stem.split("_run")[-1]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


def run_manual_fallback(target_entity, data_dir, obb_tool_path):
    """
    Launches obb_tool_ge_V0.py as a subprocess for manual OBB annotation.
    Blocks until the tool is closed, then reads the exported JSON and returns
    the OBB dict for target_entity, or None if not found.
    """
    import subprocess
    print(f"\n[Manual Fallback] Launching annotation tool for '{target_entity}'...")
    print("[Manual Fallback] Annotate the 4 corners, click 'Set as Final', export JSON, then close the tool.")
    try:
        subprocess.run(["python", str(obb_tool_path), str(data_dir)], check=False)
    except FileNotFoundError:
        subprocess.run(["python3", str(obb_tool_path), str(data_dir)], check=False)

    json_path = input("\n[Manual Fallback] Enter path to exported JSON: ").strip().strip('"')
    if not json_path or not Path(json_path).exists():
        print("[Manual Fallback] File not found. Skipping.")
        return None

    with open(json_path) as f:
        data = json.load(f)

    for entry in data:
        if entry.get("entity", "").lower() == target_entity.lower():
            print(f"[Manual Fallback] OBB found for '{target_entity}'.")
            return entry["obb"]

    print(f"[Manual Fallback] '{target_entity}' not found in exported JSON.")
    print(f"  Available: {[e.get('entity') for e in data]}")
    return None


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="Automated OBB Estimation Pipeline")
    _ap.add_argument("--manual_mode", action="store_true",
                     help="Force manual annotation for all classes regardless of detection count. "
                          "Panel normal is still estimated automatically first.")
    _args, _ = _ap.parse_known_args()
    MANUAL_MODE = _args.manual_mode

    # ── Paths — relative to repo root (RP_OBB_Estimation/) ───────────────────
    # Script lives at src/pipeline/ so parent.parent = src/
    REPO_ROOT      = Path(__file__).resolve().parent.parent
    DATA_DIR       = REPO_ROOT / "data"
    CHECKPOINT_DIR = REPO_ROOT / "sam2" / "checkpoints"
    YOLO_WEIGHTS   = str(REPO_ROOT / "yolo" / "weights" / "best.pt")
    OBB_TOOL_PATH  = REPO_ROOT / "pipeline" / "obb_tool_ge_V0.py"
    OUTPUTS_DIR    = REPO_ROOT / "outputs"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    SAM_MODELS     = {"small": {"cfg":  "configs/sam2.1/sam2.1_hiera_s.yaml",
                                "ckpt": "sam2.1_hiera_small.pt"}}
    model_cfg       = SAM_MODELS["small"]["cfg"]
    checkpoint_path = str(CHECKPOINT_DIR / SAM_MODELS["small"]["ckpt"])

    GROUND_TRUTH_DATABASE = {
        "VGA_socket": {
            "entity": "VGA_socket",
            "obb": {
                "center":   [0.2704921202927293, 0.2261220732082181, 0.8349008829378597],
                "extent":   [0.03537766175069747, 0.011822199241650923, 0.0061316691090621735],
                "rotation": [[-0.004004375172752437,  0.9672545151126772, -0.25377680739897346],
                              [ 0.01584254528462312,   0.25380835519540434, 0.9671247761234889],
                              [ 0.9998664804554559,   -0.00014774012094266402, -0.016340117333610394]]
            }
        },
        "ethernet_socket": None,
        "power_socket":    None,
        "PS2_socket":      None,
        "HDMI_socket":     None,
        "USB_socket":      None,
    }

    print("\n" + "="*50 + "\n  AUTOMATED OBB EXTRACTION PIPELINE  \n" + "="*50)
    classes = list(GROUND_TRUTH_DATABASE.keys())
    for i, cls_name in enumerate(classes):
        print(f"  {i+1}. {cls_name}")

    try:
        selection     = int(input("\nEnter the number of the component to measure: ")) - 1
        target_entity = classes[selection]
    except (ValueError, IndexError):
        print("[System] Invalid selection. Exiting.")
        exit()

    target_gt = GROUND_TRUTH_DATABASE[target_entity]
    cfg       = SOCKET_CONFIG[target_entity]
    run_num   = get_next_run_number(OUTPUTS_DIR, target_entity)
    print(f"[Output] Run #{run_num} for {target_entity}")

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print(f"\n[System] Loading data from {DATA_DIR}...")
    with open(DATA_DIR / "intrinsic.json") as f:
        K = np.array(json.load(f)["camera_matrix"])
    with open(DATA_DIR / "poses.json") as f:
        poses = {str(k): np.array(v) for k, v in json.load(f).items()}
    image_paths = {
        str(int(p.stem.replace("frame_", ""))): p
        for p in DATA_DIR.glob("frame_*.png")
        if str(int(p.stem.replace("frame_", ""))) in poses
    }

    # ── 2. Panel normal (once, shared by all sockets) ─────────────────────────
    use_gt_normal = input("\nDo you want to use reference GT normal vector? (y/n): ").strip().lower() == "y"

    if use_gt_normal:
        # Col 2 of the GT VGA rotation matrix — the known panel outward normal.
        panel_normal = np.array([-0.25377680739897346,
                                  0.9671247761234889,
                                 -0.016340117333610394])
        panel_normal /= np.linalg.norm(panel_normal)
        print(f"[Panel Normal] Using GT normal: {panel_normal.round(4)}")
    else:
        print("\n[YOLO] Detecting PC_Panel...")
        panel_boxes = get_yolo_bounding_boxes(
            image_paths, "PC_Panel",
            model_path=YOLO_WEIGHTS, conf_thresh=0.5)
        print(f"[YOLO] Found PC_Panel in {len(panel_boxes)}/{len(image_paths)} frames.")
        panel_normal = estimate_panel_normal(image_paths, poses, K, panel_boxes)

    # ── 3. Socket detection ────────────────────────────────────────────────────
    print(f"\n[YOLO] Detecting {target_entity} (conf ≥ {cfg['conf_threshold']})...")
    user_boxes = get_yolo_bounding_boxes(
        image_paths, target_entity,
        model_path=YOLO_WEIGHTS, conf_thresh=cfg["conf_threshold"])

    print(f"[YOLO] Found {target_entity} in {len(user_boxes)}/{len(image_paths)} frames.")

    # ── 4. Manual fallback decision (before box filter) ───────────────────────
    # --manual_mode forces manual for any class.
    # Fewer than 3 clean detections also prompts for manual.
    use_manual = MANUAL_MODE
    if not use_manual and len(user_boxes) < 3:
        print(f"\n[Warning] Only {len(user_boxes)} detection(s) — below threshold of 3.")
        use_manual = input("Launch manual annotation tool? (y/n): ").strip().lower() == "y"
    if not use_manual and len(user_boxes) < 2:
        print("[System] Too few boxes for triangulation and manual not requested. Exiting.")
        exit()

    # ── 5. Visualisation options ──────────────────────────────────────────────
    show_masks = input("\nShow SAM2 segmentation masks? (y/n): ").strip().lower() == "y"
    show_proj  = input("Show 3D OBB projection on images? (y/n): ").strip().lower() == "y"

    # ── 6. Socket pipeline ─────────────────────────────────────────────────────
    calc_obb        = None
    surviving_voxels = None

    if use_manual:
        calc_obb = run_manual_fallback(target_entity, DATA_DIR, OBB_TOOL_PATH)
        if calc_obb is None:
            print("[System] Manual annotation did not produce an OBB. Exiting.")
            exit()
    else:
        # Box filter only applies for automated path
        print(f"[BoxFilter] Filtering {target_entity} detections by reprojection...")
        user_boxes  = filter_boxes_by_reprojection(user_boxes, K, poses, tol_factor=1.0)
        box_centers = [(fid, (b[0]+b[2])/2, (b[1]+b[3])/2) for fid, b in user_boxes.items()]
        if len(user_boxes) < 2:
            print("[System] Too few boxes after filtering. Exiting.")
            exit()
        try:
            rough_center_3d = triangulate_dlt(box_centers, K, poses)
            masks = generate_sam_masks(image_paths, user_boxes, model_cfg, checkpoint_path)
            if show_masks:
                display_sam_masks(image_paths, masks, label=target_entity, bboxes=user_boxes)
            surviving_voxels = voxel_voting_gpu(
                K, poses, masks, rough_center_3d,
                user_boxes=user_boxes,
                size_mm=30, res_mm=0.5, consensus_ratio=0.875)
            calc_obb = extract_obb_pca(
                surviving_voxels,
                depth_prior_mm=cfg["depth_prior_mm"],
                panel_normal=panel_normal,
                percentile=cfg["percentile"],
                use_minarearect=cfg["use_minarearect"])
        except Exception as e:
            print(f"\n[FATAL ERROR] Pipeline failed: {e}")
            raise

    print("\n================ FINAL OBB ================")
    print(json.dumps(calc_obb, indent=2))

    # ── Save OBB JSON ─────────────────────────────────────────────────────────
    obb_json_path = OUTPUTS_DIR / f"{target_entity}_run{run_num}.json"
    with open(obb_json_path, "w") as _f:
        json.dump({"entity": target_entity, "run": run_num, "obb": calc_obb}, _f, indent=2)
    print(f"[Output] OBB saved to {obb_json_path}")

    # ── Projection image save dir ─────────────────────────────────────────────
    proj_save_dir = OUTPUTS_DIR / target_entity / f"run{run_num}" if show_proj else None

    if target_gt:
        iou_3d  = calculate_exact_3d_iou(calc_obb, target_gt)
        ious_2d = calculate_2d_projection_iou(
            K, poses, calc_obb, target_gt, image_paths, user_boxes)
        print("\n================ EVALUATION ================")
        print(f"Exact 3D Volumetric IoU : {iou_3d * 100:.2f}%")
        print(f"Avg 2D Polygonal IoU    : {np.mean(list(ious_2d.values())) * 100:.2f}%")
        print("=" * 43)
        if show_proj:
            display_obb_projections(K, poses, calc_obb, target_gt, image_paths, user_boxes,
                                    save_dir=proj_save_dir, show=True)
        if surviving_voxels is not None:
            display_3d_obbs(calc_obb, target_gt, surviving_voxels)
    else:
        if show_proj:
            display_obb_projections(K, poses, calc_obb, None, image_paths, user_boxes,
                                    save_dir=proj_save_dir, show=True)

    if proj_save_dir:
        print(f"[Output] Projection images saved to {proj_save_dir}")