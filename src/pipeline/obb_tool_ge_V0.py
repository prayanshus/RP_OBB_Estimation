#!/usr/bin/env python3
"""
OBB Annotation Tool  v3.1 — Multi-View Triangulation + LightGlue
================================================================

Annotate each corner of a socket across multiple camera views.
The tool triangulates each corner into a single 3D world point,
refines it with Huber-robust optimisation, then computes the OBB.
"""

import sys, json, math
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
from scipy.optimize import least_squares

# Try to import Deep Learning dependencies
try:
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image
    HAS_LIGHTGLUE = True
except ImportError:
    HAS_LIGHTGLUE = False

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

VERSION  = "3.1"
APP_TITLE = f"OBB Annotation Tool  v{VERSION}"
MIN_W, MIN_H = 1380, 850
LEFT_W  = 215
RIGHT_W = 350

CC = ["#FF4444", "#44DD44", "#4499FF", "#FFCC00"]
CC_DIM = ["#662222", "#226622", "#223366", "#665500"]
CC_NAMES = ["C1", "C2", "C3", "C4"]

AX_W = "#FF4444"
AX_H = "#44DD44"
AX_D = "#4499FF"

OBB_COLOR = "#FF00FF"
EPIPOLAR_COLOR = "#00FFFF"
REPROJ_COLOR_GOOD = "#00FF88"
REPROJ_COLOR_BAD  = "#FF6600"

ENTITIES = ["vga_socket", "ethernet_socket", "power_socket", "PS2_socket", "HDMI_socket", "USB_socket"]

DEFAULTS_MM = {
    "vga_socket":      6.1,
    "ethernet_socket": 15.0,
    "power_socket":    18.0,
}

# GT panel normal — col 2 of VGA ground truth rotation matrix
GT_PANEL_NORMAL = np.array([-0.25377680739897346,
                              0.9671247761234889,
                             -0.016340117333610394])

OBB_EDGES = [
    (0,1),(2,3),(4,5),(6,7),
    (0,2),(1,3),(4,6),(5,7),
    (0,4),(1,5),(2,6),(3,7),
]

HELP_TEXT = """\
MULTI-VIEW TRIANGULATION  —  v3.1
══════════════════════════════════
  For each corner of the socket face:
   1. Select corner tab  (C1 / C2 / C3 / C4)
   2. Click the corner in a reference frame
   3. If LightGlue is enabled, the tool will auto-propagate.
   4. Otherwise, switch frames and click near the cyan epipolar line.

MOUSE / KEYBOARD
────────────────
  Left-click       add observation for active corner
  Right-click      remove last observation
  Scroll           zoom in / out
  Middle-drag      pan image
  Double-click     fit to window
  Ctrl+Z           undo last observation
  Ctrl+O           open data folder
  Ctrl+E           export JSON
  1/2/3/4          switch corner tab
"""

# ══════════════════════════════════════════════════════════════════════════════
# Geometry
# ══════════════════════════════════════════════════════════════════════════════

def project_point(P_w, K, T_c2w):
    T_w2c = np.linalg.inv(T_c2w)
    pc = T_w2c[:3, :3] @ P_w + T_w2c[:3, 3]
    if pc[2] <= 1e-4: return None
    uv = K @ pc
    return (uv[0] / uv[2], uv[1] / uv[2])

def triangulate_dlt(observations, K, poses):
    A = []
    for fid, u, v in observations:
        T_w2c = np.linalg.inv(poses[fid])
        P = K @ T_w2c[:3]
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def refine_point_huber(X0, observations, K, poses, f_scale=3.0):
    def residuals(X):
        res = []
        for fid, u, v in observations:
            uv_p = project_point(X, K, poses[fid])
            if uv_p is None:
                res.extend([1e3, 1e3])
            else:
                res.extend([uv_p[0] - u, uv_p[1] - v])
        return np.array(res)
    result = least_squares(residuals, X0, loss='huber', f_scale=f_scale)
    return result.x

def compute_reproj_errors(X, observations, K, poses):
    errors = {}
    for fid, u, v in observations:
        uv_p = project_point(X, K, poses[fid])
        if uv_p:
            errors[fid] = math.hypot(uv_p[0] - u, uv_p[1] - v)
        else:
            errors[fid] = float('inf')
    return errors

def compute_epipolar_line(u_src, v_src, K, T_src_c2w, T_dst_c2w):
    T_rel = np.linalg.inv(T_dst_c2w) @ T_src_c2w
    R, t = T_rel[:3, :3], T_rel[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    p = np.array([u_src, v_src, 1.0])
    line = F @ p
    n = math.hypot(line[0], line[1])
    if n > 1e-10: line /= n
    return line

def point_to_line_dist(u, v, line):
    a, b, c = line
    return abs(a*u + b*v + c) / math.hypot(a, b)

def harris_snap(gray, cx, cy, radius=12):
    h, w = gray.shape
    x0, y0 = max(0, int(cx - radius)), max(0, int(cy - radius))
    x1, y1 = min(w, int(cx + radius)), min(h, int(cy + radius))
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    if patch.size < 9: return cx, cy, False
    harris = cv2.cornerHarris(patch, blockSize=3, ksize=3, k=0.04)
    if harris.max() <= 0: return cx, cy, False
    thresh = harris.max() * 0.3
    ys, xs = np.where(harris > thresh)
    if len(xs) == 0: return cx, cy, False
    dists = np.hypot(xs - (cx - x0), ys - (cy - y0))
    best = np.argmin(dists)
    if dists[best] > radius: return cx, cy, False
    return float(xs[best] + x0), float(ys[best] + y0), True

def check_coplanarity(pts_3d):
    P = np.array(pts_3d)
    centroid = P.mean(axis=0)
    _, S, Vt = np.linalg.svd(P - centroid)
    normal = Vt[-1]
    dists = np.abs((P - centroid) @ normal)
    planarity_mm = float(dists.max() * 1000)

    edges = [P[1]-P[0], P[2]-P[1], P[3]-P[2], P[0]-P[3]]
    angles = []
    for i in range(4):
        e1, e2 = edges[i], edges[(i+1) % 4]
        cos_a = np.clip(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12), -1, 1)
        angles.append(math.degrees(math.acos(abs(cos_a))))

    d1 = np.linalg.norm(P[2] - P[0])
    d2 = np.linalg.norm(P[3] - P[1])
    diag_ratio = min(d1, d2) / (max(d1, d2) + 1e-12)

    return {"planarity_mm": planarity_mm, "angles_deg": angles, "diag_ratio": diag_ratio}

def compute_obb_from_corners(pts_3d, depth_m, panel_normal=None):
    P = np.array(pts_3d)
    w_vec = ((P[1] - P[0]) + (P[2] - P[3])) / 2
    h_vec = ((P[3] - P[0]) + (P[2] - P[1])) / 2
    W = float(np.linalg.norm(w_vec))
    H = float(np.linalg.norm(h_vec))
    w_hat = w_vec / (W + 1e-12)
    h_hat = h_vec / (H + 1e-12)

    if panel_normal is not None:
        # Use provided panel normal directly — don't derive from clicked corners
        d_hat = np.array(panel_normal, dtype=float)
        d_hat /= np.linalg.norm(d_hat)
        # Re-orthogonalise w_hat and h_hat against d_hat
        w_hat = w_hat - np.dot(w_hat, d_hat) * d_hat
        w_hat /= (np.linalg.norm(w_hat) + 1e-12)
        h_hat = np.cross(d_hat, w_hat)
        h_hat /= (np.linalg.norm(h_hat) + 1e-12)
    else:
        d_hat = np.cross(w_hat, h_hat)
        d_hat /= (np.linalg.norm(d_hat) + 1e-12)
        if np.dot(np.cross(w_hat, h_hat), d_hat) < 0:
            d_hat = -d_hat

    # Enforce W >= H convention
    if W < H:
        W, H = H, W
        w_hat, h_hat = h_hat, w_hat

    face_center = P.mean(axis=0)
    center = face_center + d_hat * (depth_m / 2)
    R = np.column_stack([w_hat, h_hat, d_hat])
    return {"center": center.tolist(), "extent": [W, H, float(depth_m)], "rotation": R.tolist()}

def project_obb(obb, K, T_c2w):
    c = np.array(obb["center"])
    e = np.array(obb["extent"])
    R = np.array(obb["rotation"])
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pw = c + R @ (np.array([sx, sy, sz]) * e / 2)
                uv = project_point(pw, K, T_c2w)
                if uv is None: return None
                pts.append(uv)
    return pts

# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _arrowhead(draw, tip, base, color, size=10):
    dx, dy = tip[0] - base[0], tip[1] - base[1]
    ln = math.hypot(dx, dy)
    if ln < 1e-3: return
    ux, uy = dx / ln, dy / ln
    px, py = -uy, ux
    p1 = (tip[0] - ux*size + px*size*0.45, tip[1] - uy*size + py*size*0.45)
    p2 = (tip[0] - ux*size - px*size*0.45, tip[1] - uy*size - py*size*0.45)
    draw.polygon([tip, p1, p2], fill=color)

def draw_arrow(draw, p0, p1, color, label=None, width=3, arrow_size=12):
    draw.line([p0, p1], fill=color, width=width)
    _arrowhead(draw, p1, p0, color, size=arrow_size)
    if label: draw.text((p1[0]+6, p1[1]-10), label, fill=color)

def draw_dot(draw, p, r, fill, outline="#fff", lw=2):
    draw.ellipse([p[0]-r, p[1]-r, p[0]+r, p[1]+r], fill=fill, outline=outline, width=lw)

def draw_cross(draw, p, r, color, width=2):
    draw.line([(p[0]-r, p[1]), (p[0]+r, p[1])], fill=color, width=width)
    draw.line([(p[0], p[1]-r), (p[0], p[1]+r)], fill=color, width=width)

# ══════════════════════════════════════════════════════════════════════════════
# Data Model
# ══════════════════════════════════════════════════════════════════════════════

class CornerState:
    def __init__(self):
        self.observations = []
        self.point_3d = None         
        self.reproj_errors = {}      

    def clear(self):
        self.observations.clear()
        self.point_3d = None
        self.reproj_errors.clear()

    @property
    def n_obs(self): return len(self.observations)

    @property
    def is_triangulated(self): return self.point_3d is not None

    def triangulate(self, K, poses):
        if self.n_obs < 2: return
        X = triangulate_dlt(self.observations, K, poses)
        X = refine_point_huber(X, self.observations, K, poses)
        self.point_3d = X
        self.reproj_errors = compute_reproj_errors(X, self.observations, K, poses)

    def mean_error(self):
        if not self.reproj_errors: return float('inf')
        return float(np.mean(list(self.reproj_errors.values())))

class EntityState:
    def __init__(self):
        self.corners = [CornerState() for _ in range(4)]
        self.obb = None
        self.coplanarity = None

    def clear(self):
        for c in self.corners: c.clear()
        self.obb = None
        self.coplanarity = None

    @property
    def all_triangulated(self):
        return all(c.is_triangulated for c in self.corners)

# ══════════════════════════════════════════════════════════════════════════════
# Application
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self, data_dir=None):
        super().__init__()
        self.title(APP_TITLE)
        self.minsize(MIN_W, MIN_H)

        self.K = None
        self.poses = {}
        self.imgmap = {}
        self._gray_cache = {}

        self.frame_id = None
        self.orig_img = None
        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_ref = None

        self.entities = {e: EntityState() for e in ENTITIES}
        self.finals = {}

        self._active_entity = ENTITIES[0]
        self._active_corner = 0

        self.panel_normal = None   # loaded from panel_normal.json if present
        self._use_panel_normal = tk.BooleanVar(value=False)
        self._use_gt_normal    = tk.BooleanVar(value=False)
        self._snap_enabled = tk.BooleanVar(value=True)
        self._snap_r_var = tk.StringVar(value="12")
        self._lg_enabled = tk.BooleanVar(value=HAS_LIGHTGLUE)
        self._tol_var = tk.StringVar(value="5")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if HAS_LIGHTGLUE else None
        self.lg_extractor = None
        self.lg_matcher = None

        self._build_ui()
        self._style()

        if data_dir:
            self.after(200, lambda: self._load(str(data_dir)))

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        BG, FG, SEL, EBG = "#2b2b2b", "#e0e0e0", "#4a90d9", "#3c3c3c"
        s.configure(".", background=BG, foreground=FG, fieldbackground=EBG,
                    troughcolor=EBG, selectbackground=SEL, selectforeground="#fff", insertcolor=FG)
        s.configure("TLabelframe", background=BG, bordercolor="#555")
        s.configure("TLabelframe.Label", background=BG, foreground="#aaa")
        s.configure("TButton", background="#404040", padding=4)
        s.map("TButton", background=[("active", "#555")])
        s.configure("TCombobox", fieldbackground=EBG, selectbackground=EBG)
        s.configure("TEntry", fieldbackground=EBG)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("TFrame", background=BG)
        s.configure("TScrollbar", background="#404040", troughcolor=EBG)
        s.configure("TCheckbutton", background=BG, foreground=FG)
        s.map("TCheckbutton", background=[("active", "#333")])
        for i, col in enumerate(CC):
            s.configure(f"C{i}.TButton", background="#404040", foreground=col, padding=5)
            s.map(f"C{i}.TButton", background=[("active", "#555")])
            s.configure(f"C{i}A.TButton", background=col, foreground="#000", padding=5, font=("Helvetica", 10, "bold"))
            s.map(f"C{i}A.TButton", background=[("active", col)])
        s.configure("Solve.TButton", background="#1a5c2e", foreground="#aaffaa", padding=6)
        s.map("Solve.TButton", background=[("active", "#256b3a"), ("disabled", "#333")])
        self.configure(bg=BG)

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_menu()
        self._build_left()
        self._build_center()
        self._build_right()

    def _build_menu(self):
        M = lambda parent=None, **kw: tk.Menu(parent, tearoff=False, bg="#333", fg="#ddd", activebackground="#4a90d9", activeforeground="white", **kw)
        mb = M(self); self.configure(menu=mb)
        fm = M(mb); mb.add_cascade(label="File", menu=fm)
        fm.add_command(label="Open Data Folder…  Ctrl+O", command=self._on_open)
        fm.add_command(label="Export answers.json  Ctrl+E", command=self._on_export)
        fm.add_separator()
        fm.add_command(label="Quit", command=self.quit)
        hm = M(mb); mb.add_cascade(label="Help", menu=hm)
        hm.add_command(label="Instructions & Hotkeys", command=self._show_help)
        self.bind_all("<Control-o>", lambda _: self._on_open())
        self.bind_all("<Control-e>", lambda _: self._on_export())
        self.bind_all("<Control-z>", lambda _: self._undo())
        for i in range(4): self.bind_all(str(i+1), lambda _, idx=i: self._select_corner(idx))

    def _build_left(self):
        f = ttk.LabelFrame(self, text=" Frames ", padding=4)
        f.grid(row=0, column=0, sticky="nsew", padx=(6, 2), pady=6)
        f.rowconfigure(0, weight=1); f.columnconfigure(0, weight=1)
        f.configure(width=LEFT_W); f.grid_propagate(False)
        self.lbox = tk.Listbox(f, width=20, bg="#1e1e1e", fg="#ccc", selectbackground="#4a90d9", selectforeground="#fff", activestyle="none", exportselection=False, font=("Courier", 10), relief="flat", bd=0)
        sb = ttk.Scrollbar(f, orient="vertical", command=self.lbox.yview)
        self.lbox.config(yscrollcommand=sb.set)
        self.lbox.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.lbox.bind("<<ListboxSelect>>", self._on_frame_select)

    def _build_center(self):
        f = ttk.Frame(self)
        f.grid(row=0, column=1, sticky="nsew", padx=2, pady=6)
        f.rowconfigure(0, weight=1); f.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(f, bg="#111", cursor="crosshair", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.status_var = tk.StringVar(value="Open a data folder  ·  File → Open  (Ctrl+O)")
        ttk.Label(f, textvariable=self.status_var, relief="sunken", anchor="w", padding=(6, 2), font=("Courier", 9)).grid(row=1, column=0, sticky="ew")
        
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", lambda _: self._undo())
        self.canvas.bind("<Double-Button-1>", lambda _: self._fit())
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-4>", self._on_scroll)
        self.canvas.bind("<Button-5>", self._on_scroll)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<Configure>", lambda _: self._redraw())

    def _build_right(self):
        outer = ttk.Frame(self, width=RIGHT_W)
        outer.grid(row=0, column=2, sticky="nsew", padx=(2, 6), pady=6)
        outer.grid_propagate(False); outer.columnconfigure(0, weight=1)
        can = tk.Canvas(outer, bg="#2b2b2b", highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=can.yview)
        can.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        can.pack(side="left", fill="both", expand=True)
        f = ttk.Frame(can)
        fw = can.create_window((0, 0), window=f, anchor="nw")
        f.bind("<Configure>", lambda e: can.configure(scrollregion=can.bbox("all")))
        can.bind("<Configure>", lambda e: can.itemconfig(fw, width=e.width))
        f.columnconfigure(0, weight=1)
        r = 0

        # Entity
        ef = ttk.LabelFrame(f, text=" Entity ", padding=8)
        ef.grid(row=r, column=0, sticky="ew", pady=(0, 5)); r += 1
        ef.columnconfigure(1, weight=1)
        ttk.Label(ef, text="Socket:").grid(row=0, column=0, sticky="w")
        self.entity_var = tk.StringVar(value=ENTITIES[0])
        cb = ttk.Combobox(ef, textvariable=self.entity_var, values=ENTITIES, state="readonly", width=18)
        cb.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        cb.bind("<<ComboboxSelected>>", lambda _: self._on_entity_change())

        # Corner tabs
        cf = ttk.LabelFrame(f, text=" Active Corner ", padding=8)
        cf.grid(row=r, column=0, sticky="ew", pady=(0, 5)); r += 1
        cf.columnconfigure((0,1,2,3), weight=1)
        self._corner_btns = []
        for i in range(4):
            btn = ttk.Button(cf, text=CC_NAMES[i], command=lambda idx=i: self._select_corner(idx))
            btn.grid(row=0, column=i, sticky="ew", padx=2)
            self._corner_btns.append(btn)
        self.corner_info_var = tk.StringVar(value="C1  ·  0 observations")
        ttk.Label(cf, textvariable=self.corner_info_var, foreground="#aaa", font=("Helvetica", 9)).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

        # Deep Matching (LightGlue)
        lgf = ttk.LabelFrame(f, text=" Deep Matching (LightGlue) ", padding=8)
        lgf.grid(row=r, column=0, sticky="ew", pady=(0, 5)); r += 1
        lg_state = "normal" if HAS_LIGHTGLUE else "disabled"
        lg_text = "Enable Auto-Propagation" if HAS_LIGHTGLUE else "LightGlue Missing (pip install lightglue)"
        ttk.Checkbutton(lgf, text=lg_text, variable=self._lg_enabled, state=lg_state).grid(row=0, column=0, columnspan=2, sticky="w")
        tf = ttk.Frame(lgf)
        tf.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(tf, text="Epipolar Tolerance (px):").pack(side="left")
        ttk.Entry(tf, textvariable=self._tol_var, width=5).pack(side="left", padx=(4, 0))

        # Harris snap
        hf = ttk.LabelFrame(f, text=" Classical Harris Snap ", padding=8)
        hf.grid(row=r, column=0, sticky="ew", pady=(0, 5)); r += 1
        ttk.Checkbutton(hf, text="Snap manual clicks to nearest corner", variable=self._snap_enabled).grid(row=0, column=0, sticky="w")

        # Observations
        of = ttk.LabelFrame(f, text=" Observations (active corner) ", padding=8)
        of.grid(row=r, column=0, sticky="ew", pady=(0, 5)); r += 1
        of.columnconfigure(0, weight=1)
        self.obs_box = tk.Text(of, height=5, bg="#141414", fg="#ccc", font=("Courier", 8), relief="flat", state="disabled", wrap="none", bd=0)
        self.obs_box.grid(row=0, column=0, sticky="ew")
        bf = ttk.Frame(of)
        bf.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        bf.columnconfigure((0, 1), weight=1)
        ttk.Button(bf, text="↩ Undo Last", command=self._undo).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(bf, text="✕ Clear Corner", command=self._clear_corner).grid(row=0, column=1, sticky="ew", padx=(2, 0))

        # Triangulation info & Computations
        tf = ttk.LabelFrame(f, text=" Triangulation Status ", padding=8)
        tf.grid(row=r, column=0, sticky="ew", pady=(0, 5)); r += 1
        tf.columnconfigure(0, weight=1)
        self.tri_var = tk.StringVar(value="")
        ttk.Label(tf, textvariable=self.tri_var, foreground="#aaa", font=("Courier", 8), wraplength=310, justify="left").grid(row=0, column=0, sticky="w")

        obbf = ttk.LabelFrame(f, text=" OBB Computation ", padding=8)
        obbf.grid(row=r, column=0, sticky="ew", pady=(0, 5)); r += 1
        obbf.columnconfigure(1, weight=1)
        ttk.Label(obbf, text="Depth D (mm):").grid(row=0, column=0, sticky="w")
        self._depth_var = tk.StringVar(value="6.1")
        ttk.Entry(obbf, textvariable=self._depth_var, width=8).grid(row=0, column=1, sticky="w", padx=(6, 0))

        # Normal source controls
        ttk.Checkbutton(obbf, text="Use panel normal (from file)",
                        variable=self._use_panel_normal,
                        command=self._on_normal_toggle).grid(
                        row=1, column=0, columnspan=2, sticky="w", pady=(4,0))
        ttk.Checkbutton(obbf, text="Use GT normal (hardcoded)",
                        variable=self._use_gt_normal,
                        command=self._on_gt_normal_toggle).grid(
                        row=2, column=0, columnspan=2, sticky="w")
        self._normal_label_var = tk.StringVar(value="Normal: from corners")
        ttk.Label(obbf, textvariable=self._normal_label_var,
                  foreground="#aaa", font=("Courier", 8),
                  wraplength=310).grid(row=3, column=0, columnspan=2, sticky="w", pady=(2,4))

        self.obb_btn = ttk.Button(obbf, text="⚙  Compute OBB", command=self._compute_obb, style="Solve.TButton", state="disabled")
        self.obb_btn.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4,0))

        # Geometry check
        cpf = ttk.LabelFrame(f, text=" Geometry Check ", padding=8)
        cpf.grid(row=r, column=0, sticky="ew", pady=(0, 5)); r += 1
        cpf.columnconfigure(0, weight=1)
        self.coplanar_var = tk.StringVar(value="")
        ttk.Label(cpf, textvariable=self.coplanar_var, foreground="#aaa", font=("Courier", 8), wraplength=310, justify="left").grid(row=0, column=0, sticky="w")

        # OBB result
        rf = ttk.LabelFrame(f, text=" OBB Result ", padding=8)
        rf.grid(row=r, column=0, sticky="ew", pady=(0, 5)); r += 1
        rf.columnconfigure(0, weight=1)
        self.result_box = tk.Text(rf, height=10, bg="#141414", fg="#00ee77", font=("Courier", 8), relief="flat", state="disabled", wrap="none", bd=0)
        rsb = ttk.Scrollbar(rf, orient="vertical", command=self.result_box.yview)
        self.result_box.config(yscrollcommand=rsb.set)
        self.result_box.grid(row=0, column=0, sticky="ew")
        rsb.grid(row=0, column=1, sticky="ns")

        # Export
        xf = ttk.LabelFrame(f, text=" Finalise & Export ", padding=8)
        xf.grid(row=r, column=0, sticky="ew", pady=(0, 5)); r += 1
        xf.columnconfigure(0, weight=1)
        ttk.Button(xf, text="✅  Set as Final", command=self._set_final).grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(xf, text="💾  Export answers.json", command=self._on_export).grid(row=1, column=0, sticky="ew")
        self.export_var = tk.StringVar(value="Finals: none")
        ttk.Label(xf, textvariable=self.export_var, foreground="#aaa", font=("Helvetica", 9), wraplength=310).grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Button(xf, text="🗑  Clear All (this entity)", command=self._clear_entity).grid(row=3, column=0, sticky="ew", pady=(8, 0))

        self._select_corner(0)

    # ─────────────────────────────────────────────────────────────────────────
    # Interaction Logic
    # ─────────────────────────────────────────────────────────────────────────

    def _update_normal_label(self):
        if self._use_gt_normal.get():
            n = GT_PANEL_NORMAL
            self._normal_label_var.set(f"Normal: GT {n.round(3).tolist()}")
        elif self._use_panel_normal.get() and self.panel_normal is not None:
            n = self.panel_normal
            self._normal_label_var.set(f"Normal: file {n.round(3).tolist()}")
        else:
            self._normal_label_var.set("Normal: from corners (cross product)")

    def _on_normal_toggle(self):
        if self._use_panel_normal.get():
            self._use_gt_normal.set(False)
        self._update_normal_label()

    def _on_gt_normal_toggle(self):
        if self._use_gt_normal.get():
            self._use_panel_normal.set(False)
        self._update_normal_label()

    def _get_active_normal(self):
        """Returns the panel normal to use for OBB computation, or None."""
        if self._use_gt_normal.get():
            return GT_PANEL_NORMAL
        if self._use_panel_normal.get() and self.panel_normal is not None:
            return self.panel_normal
        return None

    def _select_corner(self, idx):
        self._active_corner = idx
        for i, btn in enumerate(self._corner_btns):
            btn.configure(style=f"C{i}A.TButton" if i == idx else f"C{i}.TButton")
        self._update_all_info()
        self._redraw()

    def _on_click(self, ev):
        if not self.orig_img or not self.frame_id: return
        if self.frame_id not in self.poses:
            self.status_var.set("⚠ No pose for this frame")
            return
            
        ox, oy = self._d2o(ev.x, ev.y)
        iw, ih = self.orig_img.size
        if not (0 <= ox < iw and 0 <= oy < ih): return

        snapped = False
        if self._snap_enabled.get():
            try: rad = int(self._snap_r_var.get())
            except ValueError: rad = 12
            gray = self._get_gray(self.frame_id)
            if gray is not None:
                ox, oy, snapped = harris_snap(gray, ox, oy, rad)

        corner = self.entities[self._active_entity].corners[self._active_corner]

        for o_fid, _, _ in corner.observations:
            if o_fid == self.frame_id:
                messagebox.showinfo("Already marked", f"{CC_NAMES[self._active_corner]} already observed in frame {self.frame_id}.\nUndo to replace.")
                return

        corner.observations.append((self.frame_id, float(ox), float(oy)))

        if self._lg_enabled.get() and corner.n_obs == 1:
            self.status_var.set("Running LightGlue Auto-Propagation...")
            self.config(cursor="watch")
            self.update()
            self._run_lightglue_propagation(corner, self.frame_id, float(ox), float(oy))
            self.config(cursor="")

        if corner.n_obs >= 2:
            corner.triangulate(self.K, self.poses)

        self._update_all_info()
        self._redraw()

    def _run_lightglue_propagation(self, corner, ref_fid, ref_u, ref_v):
        if not HAS_LIGHTGLUE: return

        if self.lg_extractor is None:
            self.lg_extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.lg_matcher = LightGlue(features='superpoint').eval().to(self.device)

        try:
            tol = float(self._tol_var.get())
        except ValueError:
            tol = 5.0

        img0 = load_image(self.imgmap[ref_fid]).to(self.device)
        feats0 = self.lg_extractor.extract(img0)
        
        kpts0 = feats0['keypoints'][0].cpu().numpy()
        dists = np.hypot(kpts0[:, 0] - ref_u, kpts0[:, 1] - ref_v)
        best_kp0_idx = np.argmin(dists)
        
        if dists[best_kp0_idx] > 20: 
            messagebox.showwarning("Feature Error", "No SuperPoint feature found near your click. Propagation may fail.")

        for tgt_fid in self.poses.keys():
            if tgt_fid == ref_fid: continue
            if tgt_fid not in self.imgmap: continue

            line = compute_epipolar_line(ref_u, ref_v, self.K, self.poses[ref_fid], self.poses[tgt_fid])

            img1 = load_image(self.imgmap[tgt_fid]).to(self.device)
            feats1 = self.lg_extractor.extract(img1)
            matches01 = self.lg_matcher({"image0": feats0, "image1": feats1})

            matches = matches01['matches'][0]
            kpts1 = feats1['keypoints'][0].cpu().numpy()

            for m in matches:
                idx0, idx1 = m[0].item(), m[1].item()
                if idx0 == best_kp0_idx:
                    tgt_u, tgt_v = kpts1[idx1]
                    epipolar_dist = point_to_line_dist(tgt_u, tgt_v, line)
                    if epipolar_dist <= tol:
                        corner.observations.append((tgt_fid, float(tgt_u), float(tgt_v)))
                    break 

    def _on_scroll(self, ev):
        zoom_in = (ev.num == 4) or (hasattr(ev, "delta") and ev.delta > 0)
        f = 1.12 if zoom_in else (1 / 1.12)
        self.pan_x = ev.x - (ev.x - self.pan_x) * f
        self.pan_y = ev.y - (ev.y - self.pan_y) * f
        self.scale *= f
        self._redraw()

    def _on_pan_start(self, ev): self._pan_ref = (ev.x, ev.y, self.pan_x, self.pan_y)

    def _on_pan_drag(self, ev):
        if not self._pan_ref: return
        x0, y0, px, py = self._pan_ref
        self.pan_x = px + (ev.x - x0)
        self.pan_y = py + (ev.y - y0)
        self._redraw()

    def _undo(self):
        corner = self.entities[self._active_entity].corners[self._active_corner]
        if corner.observations:
            corner.observations.pop()
            if corner.n_obs >= 2: corner.triangulate(self.K, self.poses)
            else:
                corner.point_3d = None
                corner.reproj_errors.clear()
        self._update_all_info()
        self._redraw()

    def _clear_corner(self):
        self.entities[self._active_entity].corners[self._active_corner].clear()
        self._update_all_info()
        self._redraw()

    def _clear_entity(self):
        name = self._active_entity
        if messagebox.askyesno("Clear", f"Clear all for '{name}'?"):
            self.entities[name].clear()
            self._update_all_info()
            self._redraw()

    def _on_entity_change(self):
        self._active_entity = self.entity_var.get()
        D = DEFAULTS_MM.get(self._active_entity, 10.0)
        self._depth_var.set(f"{D:.1f}")
        self._select_corner(0)

    def _update_all_info(self):
        self._update_corner_info()
        self._update_obs_box()
        self._update_tri_info()
        self._update_obb_btn()

    def _update_corner_info(self):
        ci = self._active_corner
        c = self.entities[self._active_entity].corners[ci]
        tri = "✓ triangulated" if c.is_triangulated else "not yet"
        self.corner_info_var.set(f"{CC_NAMES[ci]}  ·  {c.n_obs} obs  ·  {tri}")

    def _update_obs_box(self):
        b = self.obs_box
        b.config(state="normal"); b.delete("1.0", "end")
        c = self.entities[self._active_entity].corners[self._active_corner]
        if not c.observations: b.insert("end", "  (click on image to add)")
        else:
            for i, (fid, u, v) in enumerate(c.observations):
                err = c.reproj_errors.get(fid, None)
                es = f"  err={err:.1f}px" if err is not None else ""
                b.insert("end", f"  {i+1}. f{fid:>6s} ({u:.1f},{v:.1f}){es}\n")
        b.config(state="disabled")

    def _update_tri_info(self):
        ent = self.entities[self._active_entity]
        lines = []
        for ci in range(4):
            c = ent.corners[ci]
            if c.is_triangulated:
                p = c.point_3d
                me = c.mean_error()
                flag = "  ⚠" if me > 5 else ("  ✓" if me < 3 else "")
                lines.append(f"{CC_NAMES[ci]}: [{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}] err={me:.1f}px{flag}")
            else:
                lines.append(f"{CC_NAMES[ci]}: — ({c.n_obs} obs)")
        self.tri_var.set("\n".join(lines))

    def _update_obb_btn(self):
        ent = self.entities[self._active_entity]
        self.obb_btn.config(state="normal" if ent.all_triangulated else "disabled")

    def _compute_obb(self):
        ent = self.entities[self._active_entity]
        if not ent.all_triangulated:
            messagebox.showwarning("Incomplete", "All 4 corners must be triangulated.")
            return
        try: depth = float(self._depth_var.get()) / 1000.0
        except ValueError:
            messagebox.showerror("Bad depth", "Enter valid depth in mm.")
            return

        pts = [c.point_3d for c in ent.corners]
        cp = check_coplanarity(pts)
        ent.coplanarity = cp
        if cp:
            lines = [
                f"Planarity: {cp['planarity_mm']:.2f}mm ({'✓' if cp['planarity_mm'] < 3 else '⚠ high'})",
                f"Angles: {', '.join(f'{a:.1f}°' for a in cp['angles_deg'])}",
                f"Diag ratio: {cp['diag_ratio']:.3f} ({'✓' if cp['diag_ratio'] > 0.85 else '⚠ skewed'})",
            ]
            self.coplanar_var.set("\n".join(lines))

        obb = compute_obb_from_corners(pts, depth, panel_normal=self._get_active_normal())
        ent.obb = obb

        self.result_box.config(state="normal"); self.result_box.delete("1.0", "end")
        self.result_box.insert("end", json.dumps(obb, indent=2)); self.result_box.config(state="disabled")

        W = np.linalg.norm(np.array(pts[1]) - np.array(pts[0]))
        H = np.linalg.norm(np.array(pts[3]) - np.array(pts[0]))
        self.status_var.set(f"✓ OBB  '{self._active_entity}'  ·  W={W*1000:.1f}mm  H={H*1000:.1f}mm  D={depth*1000:.1f}mm")
        self._redraw()

    def _set_final(self):
        ent = self.entities[self._active_entity]
        if not ent.obb:
            messagebox.showwarning("No OBB", "Compute OBB first.")
            return
        self.finals[self._active_entity] = ent.obb.copy()
        self.export_var.set("Finals: " + ", ".join(self.finals.keys()))
        self.status_var.set(f"✓ Final set for '{self._active_entity}'")

    def _on_export(self):
        output = []
        missing = []
        for name in ENTITIES:
            ent = self.entities[name]
            obb = self.finals.get(name, ent.obb)
            if obb is None:
                missing.append(name)
                obb = {"center": ["X","Y","Z"], "extent": ["W","H","L"], "rotation": [["r","r","r"],["r","r","r"],["r","r","r"]]}
            output.append({"entity": name, "obb": obb})
        if missing and not messagebox.askyesno("Missing", f"Placeholders for: {', '.join(missing)}\nExport?"): return
        path = filedialog.asksaveasfilename(title="Save JSON", defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="my_answers.json")
        if not path: return
        with open(path, "w") as fh: json.dump(output, fh, indent=4)
        messagebox.showinfo("Exported", f"Saved → {path}")
        self.status_var.set(f"Exported → {Path(path).name}")

    def _show_help(self):
        win = tk.Toplevel(self)
        win.title("Instructions"); win.configure(bg="#2b2b2b"); win.resizable(False, False)
        txt = tk.Text(win, width=55, height=40, bg="#1e1e1e", fg="#e0e0e0", font=("Courier", 10), relief="flat", padx=12, pady=12)
        txt.pack(padx=12, pady=12); txt.insert("end", HELP_TEXT); txt.config(state="disabled")
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 12))

    def _on_open(self):
        d = filedialog.askdirectory(title="Select Data Folder")
        if d: self._load(d)

    def _load(self, d):
        d = Path(d)
        try:
            with open(d / "intrinsic.json") as fh: intr = json.load(fh)
            self.K = np.array(intr["camera_matrix"], dtype=np.float64)
            with open(d / "poses.json") as fh: raw = json.load(fh)
            self.poses = {k: np.array(v, dtype=np.float64) for k, v in raw.items()}
            self.imgmap = {}
            self._gray_cache.clear()
            for p in sorted(d.glob("frame_*.png")):
                fid = str(int(p.stem.replace("frame_", "")))
                self.imgmap[fid] = p
            if not self.imgmap: raise FileNotFoundError("No frame_*.png found.")
            self.lbox.delete(0, "end")
            for fid in sorted(self.imgmap, key=int):
                tag = "✓" if fid in self.poses else "✗"
                self.lbox.insert("end", f"  {tag}  {int(fid):06d}")
            self.title(f"{APP_TITLE}  —  {d.name}")
            self.status_var.set(f"✓  {len(self.imgmap)} images  ·  {len(self.poses)} poses  ·  K loaded")
            # Try to load panel normal written by auto_obb_sam2 pipeline
            pn_path = d / "panel_normal.json"
            if pn_path.exists():
                with open(pn_path) as fh:
                    pn_data = json.load(fh)
                self.panel_normal = np.array(pn_data["normal"])
                src = pn_data.get("source", "unknown")
                self._use_panel_normal.set(True)
                self.status_var.set(
                    f"✓  {len(self.imgmap)} images · {len(self.poses)} poses · "
                    f"panel normal loaded ({src})")
            else:
                self.panel_normal = None
                self._use_panel_normal.set(False)
            self._update_normal_label()
            self.lbox.selection_set(0)
            self.lbox.event_generate("<<ListboxSelect>>")
        except Exception as ex:
            messagebox.showerror("Load Error", str(ex))

    def _get_gray(self, fid):
        if fid not in self._gray_cache: self._gray_cache[fid] = cv2.imread(str(self.imgmap[fid]), cv2.IMREAD_GRAYSCALE)
        return self._gray_cache[fid]

    def _on_frame_select(self, _=None):
        sel = self.lbox.curselection()
        if not sel: return
        fid = sorted(self.imgmap, key=int)[sel[0]]
        if fid == self.frame_id: return
        self.frame_id = fid
        try: self.orig_img = Image.open(self.imgmap[fid])
        except Exception as ex: messagebox.showerror("Image Error", str(ex)); return
        self._fit()
        pose = "pose ✓" if fid in self.poses else "⚠ NO POSE"
        self.status_var.set(f"Frame {fid}  ·  {self.orig_img.width}×{self.orig_img.height}  ·  {pose}  ·  Corner: {CC_NAMES[self._active_corner]}")

    def _fit(self):
        if not self.orig_img: return
        cw = self.canvas.winfo_width() or 960; ch = self.canvas.winfo_height() or 680
        iw, ih = self.orig_img.size
        self.scale = min(cw / iw, ch / ih) * 0.97
        self.pan_x = (cw - iw * self.scale) / 2; self.pan_y = (ch - ih * self.scale) / 2
        self._redraw()

    def _o2d(self, x, y): return x * self.scale + self.pan_x, y * self.scale + self.pan_y
    def _d2o(self, x, y): return (x - self.pan_x) / self.scale, (y - self.pan_y) / self.scale

    def _redraw(self):
        if not self.orig_img: return
        cw = max(1, self.canvas.winfo_width()); ch = max(1, self.canvas.winfo_height())
        dw = max(1, int(self.orig_img.width * self.scale)); dh = max(1, int(self.orig_img.height * self.scale))
        resamp = Image.LANCZOS if dw < 3000 else Image.NEAREST
        disp = self.orig_img.resize((dw, dh), resamp)
        comp = Image.new("RGB", (cw, ch), "#111111")
        comp.paste(disp, (int(self.pan_x), int(self.pan_y)))
        draw = ImageDraw.Draw(comp)

        ent = self.entities[self._active_entity]
        fid = self.frame_id

        if fid and fid in self.poses:
            active = self._active_corner
            self._draw_epipolar(draw, ent.corners[active], fid)

            for ci in range(4):
                c = ent.corners[ci]
                is_act = (ci == active)
                col = CC[ci] if is_act else CC_DIM[ci]
                rd = 7 if is_act else 5

                for o_fid, u, v in c.observations:
                    if o_fid == fid:
                        dp = self._o2d(u, v)
                        draw_dot(draw, dp, rd, col, outline="#fff" if is_act else col)
                        if is_act: draw.text((dp[0]+rd+3, dp[1]-rd), CC_NAMES[ci], fill=col)

                if c.is_triangulated:
                    uv_r = project_point(c.point_3d, self.K, self.poses[fid])
                    if uv_r:
                        dp = self._o2d(*uv_r)
                        err = c.reproj_errors.get(fid, None)
                        rc = REPROJ_COLOR_GOOD
                        if err is not None and err > 5: rc = REPROJ_COLOR_BAD
                        elif err is None: rc = "#888"
                        draw_cross(draw, dp, 8 if is_act else 5, rc if is_act else CC_DIM[ci], width=2)
                        if is_act and err is not None: draw.text((dp[0]+10, dp[1]+2), f"{err:.1f}px", fill=rc)

            if ent.obb: self._draw_obb_overlay(draw, ent.obb, fid)

        self._tk_img = ImageTk.PhotoImage(comp)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

    def _draw_epipolar(self, draw, corner, dst_fid):
        if not corner.observations or not self.orig_img: return
        iw, ih = self.orig_img.size
        for o_fid, u, v in corner.observations:
            if o_fid == dst_fid or o_fid not in self.poses or dst_fid not in self.poses: continue
            line = compute_epipolar_line(u, v, self.K, self.poses[o_fid], self.poses[dst_fid])
            a, b, c = line
            pts = []
            if abs(b) > 1e-10:
                y0 = -(a * 0 + c) / b; yW = -(a * iw + c) / b
                if 0 <= y0 <= ih: pts.append((0.0, y0))
                if 0 <= yW <= ih: pts.append((float(iw), yW))
            if abs(a) > 1e-10:
                x0 = -(b * 0 + c) / a; xH = -(b * ih + c) / a
                if 0 <= x0 <= iw: pts.append((x0, 0.0))
                if 0 <= xH <= iw: pts.append((xH, float(ih)))
            if len(pts) >= 2:
                pts = sorted(set(pts))[:2]
                draw.line([self._o2d(*pts[0]), self._o2d(*pts[1])], fill=EPIPOLAR_COLOR, width=1)

    def _draw_obb_overlay(self, draw, obb, fid):
        pts = project_obb(obb, self.K, self.poses[fid])
        if pts:
            pd = [self._o2d(x, y) for x, y in pts]
            for i, j in OBB_EDGES: draw.line([pd[i], pd[j]], fill=OBB_COLOR, width=2)
        c = np.array(obb["center"]); R = np.array(obb["rotation"]); e = np.array(obb["extent"])
        c_2d = project_point(c, self.K, self.poses[fid])
        if not c_2d: return
        c_d = self._o2d(*c_2d)
        for col, lbl, ax in [(AX_W, "W", 0), (AX_H, "H", 1), (AX_D, "D", 2)]:
            tip = c + R[:, ax] * (e[ax] / 2) * 1.8
            tip_2d = project_point(tip, self.K, self.poses[fid])
            if tip_2d: draw_arrow(draw, c_d, self._o2d(*tip_2d), col, label=lbl, width=2, arrow_size=10)

# ══════════════════════════════════════════════════════════════════════════════
def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    App(data_dir).mainloop()

if __name__ == "__main__":
    main()