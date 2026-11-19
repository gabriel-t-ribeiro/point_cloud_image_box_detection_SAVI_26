"""Part 3 - main method: nonlinear least-squares box fitting.

Fits K boxes simultaneously in the table frame (z up) using
scipy.optimize.least_squares. Each box has 7 parameters:
    [x, y, z, yaw, length, width, height]

On every residual evaluation:
  1. Each point dynamically picks its closest box (unsigned surface distance).
  2. A pairwise 3D overlap penalty (exact 3D penetration depth via Separating
     Axis Theorem in xy + height overlap in z) prevents boxes from swallowing
     each other or interpenetrating across seams.
  3. A tiny (L + W) regularization keeps unseen horizontal walls from drifting
     into empty space without pulling the cropped z=0 bottom upward.

When segmentation merges touching boxes into fewer clusters than `num_boxes`
(Level 3), `optimize_boxes` grows the box set one box at a time by trying up to
three geometric split candidates and keeping the one with the lowest objective:
  - peel_add:   add a new box on the largest high-residual (> 1.5 cm) cluster
                (handles disjoint/stacked unexplained regions).
  - peel_carve: trim the existing box that owns the high-residual cluster along
                its separating axis and fit a new box to the cut-away points
                (handles touching boxes of different heights, widths, or yaws).
  - bisect:     split the highest-error (or largest) box in half along its
                longest horizontal axis (handles flush aligned boxes and acts
                as a clean fallback).
"""

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares

from bounding_box import Box
from segmentation import to_frame

F_SCALE = 0.01            # soft_l1 transition scale (m): errors > 1 cm are treated robustly
SIZE_REG_XY = 0.05        # gentle pull on (L + W) only; leaves z and H free for stacked boxes
OVERLAP_WEIGHT = 500.0    # weight on 3D penetration depth (m) between any two boxes
MIN_DIM = 0.02            # minimum box side length (m)
TABLE_Z_SNAP = 0.025      # boxes with z_init < 2.5 cm rest on the table (anchor z = 0)

# Iterative splitting parameters (used only when len(initial_boxes) < num_boxes)
RESIDUAL_THRESH = 0.015       # point distance (m) above which a point is considered unexplained
RESIDUAL_DBSCAN_EPS = 0.03    # DBSCAN radius (m) to group unexplained points
RESIDUAL_DBSCAN_MIN_PTS = 20  # minimum points for an unexplained cluster
MIN_CLUSTER_PTS = 30          # minimum points needed to fit a seed box
SEARCH_MAX_POINTS = 12000     # subsample size during candidate search for speed
SEARCH_MAX_NFEV = 60          # cap least_squares evaluations per candidate during search


def box_surface_distances(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Unsigned distance from N points (table frame) to 1 box [x, y, z, yaw, L, W, H]."""
    x, y, z, yaw, L, W, H = params
    center = np.array([x, y, z + H / 2.0])
    ca, sa = np.cos(yaw), np.sin(yaw)
    # R_T maps table-frame offsets into the box-aligned frame
    R_T = np.array([[ca, sa, 0.0],
                    [-sa, ca, 0.0],
                    [0.0, 0.0, 1.0]])
    q = (points - center) @ R_T.T
    d = np.abs(q) - np.array([L / 2.0, W / 2.0, H / 2.0])
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    return outside - inside


def _box_overlap_depth(b1: np.ndarray, b2: np.ndarray) -> float:
    """3D penetration depth (m) between two z-vertical boxes (0 if disjoint)."""
    x1, y1, z1, yaw1, L1, W1, H1 = b1
    x2, y2, z2, yaw2, L2, W2, H2 = b2

    dz = min(z1 + H1, z2 + H2) - max(z1, z2)
    if dz <= 0.0:
        return 0.0

    # Separating Axis Theorem across the 4 horizontal face normals
    u1 = np.array([np.cos(yaw1), np.sin(yaw1)])
    v1 = np.array([-u1[1], u1[0]])
    u2 = np.array([np.cos(yaw2), np.sin(yaw2)])
    v2 = np.array([-u2[1], u2[0]])
    delta = np.array([x2 - x1, y2 - y1])

    min_xy_overlap = np.inf
    for axis in (u1, v1, u2, v2):
        r1 = 0.5 * (L1 * abs(np.dot(u1, axis)) + W1 * abs(np.dot(v1, axis)))
        r2 = 0.5 * (L2 * abs(np.dot(u2, axis)) + W2 * abs(np.dot(v2, axis)))
        ov = r1 + r2 - abs(np.dot(delta, axis))
        if ov <= 0.0:
            return 0.0
        if ov < min_xy_overlap:
            min_xy_overlap = ov

    return min(min_xy_overlap, dz)


def _pack_boxes(boxes: list[Box]) -> np.ndarray:
    return np.array(
        [[b.x, b.y, b.z, b.yaw, b.length, b.width, b.height] for b in boxes],
        dtype=float,
    ).ravel()


def _unpack_boxes(theta: np.ndarray, num_boxes: int) -> list[Box]:
    rows = theta.reshape(num_boxes, 7)
    return [Box(*row) for row in rows]


def _fit_box_to_points(points_table: np.ndarray, force_ground: bool = False) -> Box:
    """Fast rough z-up Box from 3D points in the table frame (minAreaRect + z percentiles)."""
    xy = points_table[:, :2].astype(np.float32)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(xy)
    if w < h:
        length, width = float(h), float(w)
        yaw = np.radians(angle_deg + 90.0)
    else:
        length, width = float(w), float(h)
        yaw = np.radians(angle_deg)
    z_bot = 0.0 if force_ground else float(np.percentile(points_table[:, 2], 2))
    if z_bot < TABLE_Z_SNAP:
        z_bot = 0.0
    z_top = float(np.percentile(points_table[:, 2], 98))
    height = max(z_top - z_bot, MIN_DIM)
    return Box(float(cx), float(cy), z_bot, float(yaw), max(length, MIN_DIM), max(width, MIN_DIM), height)


def _split_box_along_axis(box: Box, axis: int = 0) -> tuple[Box, Box]:
    """Bisect a box into two touching halves along local u (axis=0, length) or v (axis=1, width)."""
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    if axis == 0:
        dx, dy = c * (box.length / 4.0), s * (box.length / 4.0)
        b1 = Box(box.x - dx, box.y - dy, box.z, box.yaw, box.length / 2.0, box.width, box.height)
        b2 = Box(box.x + dx, box.y + dy, box.z, box.yaw, box.length / 2.0, box.width, box.height)
    else:
        dx, dy = -s * (box.width / 4.0), c * (box.width / 4.0)
        b1 = Box(box.x - dx, box.y - dy, box.z, box.yaw, box.length, box.width / 2.0, box.height)
        b2 = Box(box.x + dx, box.y + dy, box.z, box.yaw, box.length, box.width / 2.0, box.height)
    return b1, b2


def _carve_box_for_cluster(
    box: Box,
    cluster_pts: np.ndarray,
    owner_pts: np.ndarray,
) -> tuple[Box, Box] | None:
    """Trim `box` away from `cluster_pts` along its local u or v axis, and fit a new box
    to the points of `owner_pts` on the cut-away side."""
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    u_vec = np.array([c, s])
    v_vec = np.array([-s, c])

    rel = cluster_pts[:, :2] - np.array([box.x, box.y])
    u, v = rel @ u_vec, rel @ v_vec
    u_lo, u_hi = np.percentile(u, 5), np.percentile(u, 95)
    v_lo, v_hi = np.percentile(v, 5), np.percentile(v, 95)
    u_med, v_med = float(np.median(u)), float(np.median(v))

    half_l, half_w = box.length / 2.0, box.width / 2.0
    u_keep = (max(-half_l, u_hi), half_l) if u_med < 0.0 else (-half_l, min(half_l, u_lo))
    v_keep = (max(-half_w, v_hi), half_w) if v_med < 0.0 else (-half_w, min(half_w, v_lo))
    rem_u = u_keep[1] - u_keep[0]
    rem_v = v_keep[1] - v_keep[0]

    rel_own = owner_pts[:, :2] - np.array([box.x, box.y])
    u_own, v_own = rel_own @ u_vec, rel_own @ v_vec
    min_keep = 2.0 * MIN_DIM

    if rem_u / box.length >= rem_v / box.width and rem_u >= min_keep:
        du = 0.5 * (u_keep[0] + u_keep[1])
        trimmed = Box(box.x + du * c, box.y + du * s, box.z, box.yaw, rem_u, box.width, box.height)
        cut_mask = (u_own < u_keep[0] - 0.01) if u_med < 0.0 else (u_own > u_keep[1] + 0.01)
    elif rem_v >= min_keep:
        dv = 0.5 * (v_keep[0] + v_keep[1])
        trimmed = Box(box.x - dv * s, box.y + dv * c, box.z, box.yaw, box.length, rem_v, box.height)
        cut_mask = (v_own < v_keep[0] - 0.01) if v_med < 0.0 else (v_own > v_keep[1] + 0.01)
    else:
        return None

    fit_pts = owner_pts[cut_mask] if cut_mask.sum() >= MIN_CLUSTER_PTS else cluster_pts
    new_box = _fit_box_to_points(fit_pts, force_ground=(box.z < TABLE_Z_SNAP))
    return trimmed, new_box


def _fit_boxes(
    points_table: np.ndarray,
    boxes: list[Box],
    f_scale: float,
    size_reg_xy: float,
    overlap_weight: float,
    max_nfev: int | None = None,
) -> tuple[list[Box], float]:
    """Run joint nonlinear least-squares on `boxes` in the table frame."""
    k = len(boxes)
    x0 = _pack_boxes(boxes)

    # Physical bounds per box: [x, y, z, yaw, L, W, H]
    # Anchor z at 0 for table-resting boxes so the 8 mm segmentation crop cannot pull z up;
    # keep z free for stacked boxes (z_init >= TABLE_Z_SNAP).
    lb_rows, ub_rows = [], []
    for b in boxes:
        z_ub = 1e-6 if b.z < TABLE_Z_SNAP else 1.5
        lb_rows.append([-2.0, -2.0, 0.0, -np.inf, MIN_DIM, MIN_DIM, MIN_DIM])
        ub_rows.append([2.0, 2.0, z_ub, np.inf, 1.5, 1.5, 1.5])
    lb = np.array(lb_rows, dtype=float).ravel()
    ub = np.array(ub_rows, dtype=float).ravel()
    x0 = np.clip(x0, lb + 1e-9, ub - 1e-9)

    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]

    def residuals(theta: np.ndarray) -> np.ndarray:
        boxes_params = theta.reshape(k, 7)
        dists = np.vstack([box_surface_distances(points_table, p) for p in boxes_params])
        point_res = np.min(dists, axis=0)
        size_res = size_reg_xy * (boxes_params[:, 4] + boxes_params[:, 5])
        if pairs and overlap_weight > 0.0:
            overlap_res = np.array(
                [overlap_weight * _box_overlap_depth(boxes_params[i], boxes_params[j]) for i, j in pairs],
                dtype=float,
            )
            return np.concatenate([point_res, size_res, overlap_res])
        return np.concatenate([point_res, size_res])

    res = least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=f_scale,
        max_nfev=max_nfev,
    )
    return _unpack_boxes(res.x, k), float(res.cost)


def _propose_split_candidates(points_table: np.ndarray, boxes: list[Box]) -> list[list[Box]]:
    """Build up to 3 candidate (m + 1)-box initializations from the current m-box fit."""
    bp = _pack_boxes(boxes).reshape(len(boxes), 7)
    dists = np.vstack([box_surface_distances(points_table, p) for p in bp])
    min_d = np.min(dists, axis=0)
    owner = np.argmin(dists, axis=0)

    candidates: list[list[Box]] = []

    # Candidates 1 & 2: residual peeling (add + carve) on the largest high-residual cluster
    high_mask = min_d > RESIDUAL_THRESH
    if high_mask.sum() >= MIN_CLUSTER_PTS:
        res_pts = points_table[high_mask]
        rc = o3d.geometry.PointCloud()
        rc.points = o3d.utility.Vector3dVector(res_pts)
        labels = np.asarray(rc.cluster_dbscan(eps=RESIDUAL_DBSCAN_EPS, min_points=RESIDUAL_DBSCAN_MIN_PTS))
        valid = labels >= 0
        if valid.any():
            vals, counts = np.unique(labels[valid], return_counts=True)
            cluster_pts = res_pts[labels == vals[np.argmax(counts)]]
            if len(cluster_pts) >= MIN_CLUSTER_PTS:
                candidates.append(list(boxes) + [_fit_box_to_points(cluster_pts)])
                c_dists = np.vstack([box_surface_distances(cluster_pts, p) for p in bp])
                c_owner = int(np.bincount(np.argmin(c_dists, axis=0), minlength=len(boxes)).argmax())
                carved = _carve_box_for_cluster(boxes[c_owner], cluster_pts, points_table[owner == c_owner])
                if carved is not None:
                    trimmed, b_new = carved
                    others = [boxes[i] for i in range(len(boxes)) if i != c_owner]
                    candidates.append(others + [trimmed, b_new])

    # Candidate 3: bisect the box with the highest residual energy (or largest volume if all fit tightly)
    err_per_box = [float(np.sum(np.minimum(min_d[owner == i], 0.05) ** 2)) for i in range(len(boxes))]
    if max(err_per_box) > 1e-3:
        target = int(np.argmax(err_per_box))
    else:
        target = int(np.argmax([b.length * b.width * b.height for b in boxes]))
    b = boxes[target]
    long_axis = 0 if b.length >= b.width else 1
    b1, b2 = _split_box_along_axis(b, long_axis)
    others = [boxes[i] for i in range(len(boxes)) if i != target]
    candidates.append(others + [b1, b2])

    return candidates


def optimize_boxes(
    points_camera0: np.ndarray,
    initial_boxes: list[Box],
    T_camera0_table: np.ndarray,
    num_boxes: int | None = None,
    f_scale: float = F_SCALE,
    size_reg_xy: float = SIZE_REG_XY,
    overlap_weight: float = OVERLAP_WEIGHT,
) -> list[Box]:
    """Refine box poses and dimensions in the table frame using least_squares.

    If `num_boxes` is greater than `len(initial_boxes)` (merged clusters of touching
    boxes), iteratively splits/peels until `num_boxes` boxes are fitted jointly.
    """
    target_boxes = len(initial_boxes) if num_boxes is None else num_boxes
    if target_boxes <= 0 or len(points_camera0) == 0:
        return []

    points_table = to_frame(points_camera0, T_camera0_table)
    boxes = list(initial_boxes)
    if len(boxes) == 0:
        boxes = [_fit_box_to_points(points_table)]

    if len(boxes) < target_boxes:
        rng = np.random.default_rng(0)
        if len(points_table) > SEARCH_MAX_POINTS:
            pts_search = points_table[rng.choice(len(points_table), SEARCH_MAX_POINTS, replace=False)]
        else:
            pts_search = points_table

        boxes, _ = _fit_boxes(pts_search, boxes, f_scale, size_reg_xy, overlap_weight, max_nfev=SEARCH_MAX_NFEV)
        while len(boxes) < target_boxes:
            candidates = _propose_split_candidates(pts_search, boxes)
            best_boxes, best_cost = None, np.inf
            for cand in candidates:
                fitted, cost = _fit_boxes(
                    pts_search, cand, f_scale, size_reg_xy, overlap_weight, max_nfev=SEARCH_MAX_NFEV
                )
                if cost < best_cost:
                    best_cost, best_boxes = cost, fitted
            boxes = best_boxes

    fitted_boxes, _ = _fit_boxes(points_table, boxes, f_scale, size_reg_xy, overlap_weight)
    return fitted_boxes
