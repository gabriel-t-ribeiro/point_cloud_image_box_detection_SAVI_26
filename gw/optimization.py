"""Part 3 - main method: nonlinear least-squares box fitting (Step 1: separated boxes).

Fits K boxes simultaneously in the table frame (z up) using
scipy.optimize.least_squares. Each box has 7 parameters:
    [x, y, z, yaw, length, width, height]

On every residual evaluation, each point dynamically chooses the closest box
(no outer assignment loop). A small size regularization term prevents unseen
faces from growing into empty space, and bounds keep boxes above the table.
"""

import numpy as np
from scipy.optimize import least_squares

from bounding_box import Box
from segmentation import to_frame

F_SCALE = 0.01           # soft_l1 transition scale (m): errors > 1 cm are treated robustly
SIZE_REG_WEIGHT = 0.0    # 0.0 avoids pulling the cropped z=0 bottom upward (keeps stacked boxes general)
MIN_DIM = 0.02           # minimum box side length (m)


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


def _pack_boxes(boxes: list[Box]) -> np.ndarray:
    return np.array(
        [[b.x, b.y, b.z, b.yaw, b.length, b.width, b.height] for b in boxes],
        dtype=float,
    ).ravel()


def _unpack_boxes(theta: np.ndarray, num_boxes: int) -> list[Box]:
    rows = theta.reshape(num_boxes, 7)
    return [Box(*row) for row in rows]


def optimize_boxes(
    points_camera0: np.ndarray,
    initial_boxes: list[Box],
    T_camera0_table: np.ndarray,
    f_scale: float = F_SCALE,
    size_reg_weight: float = SIZE_REG_WEIGHT,
) -> list[Box]:
    """Refine box poses and dimensions in the table frame using least_squares."""
    if len(initial_boxes) == 0 or len(points_camera0) == 0:
        return list(initial_boxes)

    points_table = to_frame(points_camera0, T_camera0_table)
    k = len(initial_boxes)
    x0 = _pack_boxes(initial_boxes)

    # Physical bounds per box: [x, y, z, yaw, L, W, H]
    lb_one = [-2.0, -2.0, 0.0, -np.inf, MIN_DIM, MIN_DIM, MIN_DIM]
    ub_one = [2.0, 2.0, 1.5, np.inf, 1.5, 1.5, 1.5]
    lb = np.tile(lb_one, k)
    ub = np.tile(ub_one, k)
    x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)

    def residuals(theta: np.ndarray) -> np.ndarray:
        boxes_params = theta.reshape(k, 7)
        dists = np.vstack([box_surface_distances(points_table, p) for p in boxes_params])
        point_res = np.min(dists, axis=0)
        size_res = size_reg_weight * np.sum(boxes_params[:, 4:7], axis=1)
        return np.concatenate([point_res, size_res])

    res = least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=f_scale,
    )
    return _unpack_boxes(res.x, k)
