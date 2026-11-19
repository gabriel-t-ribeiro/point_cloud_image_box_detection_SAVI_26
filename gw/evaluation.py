"""Comparison with ground_truth.json (only for measuring errors, never for estimating).

Registration error: same function as our part8_ex2 / part8_ex3.
Segmentation error: as described in Part 2 of the assignment. The ground truth does
not say which box each point belongs to, so we aggregate the clouds with the TRUE
transformations and label as "box k" the points inside box k (with a 5 mm margin).
"""

import json
import os

import numpy as np


def load_ground_truth(scene_folder):
    """Returns the ground truth dict, or None for test scenes (no ground_truth.json)."""
    path = os.path.join(scene_folder, "ground_truth.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Part 1 - registration
# ---------------------------------------------------------------------------
def compute_registration_error(T_est, T_true):
    """Translation error (mm) and rotation error (deg) of E = inv(T_true) @ T_est.

    Rotation angle from the trace: tr(R) = 1 + 2 cos(theta).
    """
    E = np.linalg.inv(T_true) @ T_est
    trans_err_mm = float(np.linalg.norm(E[:3, 3]) * 1000.0)
    cos_theta = np.clip((np.trace(E[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = float(np.degrees(np.arccos(cos_theta)))
    return trans_err_mm, rot_err_deg


def true_camera_transforms(ground_truth, names):
    by_name = {c["image"]: np.array(c["T_camera0_camera"]) for c in ground_truth["cameras"]}
    return [by_name[n] for n in names]


def print_photometry(names, s, b, ground_truth=None):
    """Estimated gain s and bias b of every image relative to image 000 (and the true ones, if known)."""
    import photometry
    true = photometry.true_relative_exposure(ground_truth, names) if ground_truth else None
    print("\n  Photometric correction (each image vs image 000: c_i = s * c_000 + b)")
    print("  image |     s |      b" + ("  | true s | true b" if true else ""))
    for k, name in enumerate(names):
        line = f"  {name:>5} | {s[k]:5.3f} | {b[k]:+6.3f}"
        if true:
            line += f"  |  {true[0][k]:5.3f} | {true[1][k]:+6.3f}"
        print(line)


def print_registration_errors(names, transforms, ground_truth):
    print("\n  Registration errors vs ground truth")
    print("  image | translation (mm) | rotation (deg)")
    errors = []
    for name, T, T_true in zip(names, transforms, true_camera_transforms(ground_truth, names)):
        t_err, r_err = compute_registration_error(T, T_true)
        errors.append((t_err, r_err))
        print(f"  {name:>5} | {t_err:16.2f} | {r_err:14.3f}")
    errors = np.array(errors)
    print(f"   mean | {errors[:, 0].mean():16.2f} | {errors[:, 1].mean():14.3f}")
    return errors


# ---------------------------------------------------------------------------
# Part 2 - segmentation
# ---------------------------------------------------------------------------
def points_inside_box(points, T_camera0_box, length, width, height, margin=0.005):
    """Boolean mask of the points (in camera0 frame) inside the box, with a margin."""
    T_box_camera0 = np.linalg.inv(T_camera0_box)
    p = points @ T_box_camera0[:3, :3].T + T_box_camera0[:3, 3]   # points in the box frame
    half = np.array([length, width, height]) / 2.0 + margin
    return np.all(np.abs(p) <= half, axis=1)


def print_segmentation_errors(points, object_indices, ground_truth, margin=0.005):
    """Number of boxes found, and precision / recall of each segmented point set.

    points:         (N, 3) aggregated cloud built with the TRUE transforms
                    (so registration errors do not count here)
    object_indices: one index array per segmented object, into `points`

    Each segmented set is matched with the true box that contains most of its points.
      precision = fraction of the set's points that are inside that box
      recall    = fraction of that box's points that ended in the set
    """
    boxes = ground_truth["boxes"]
    inside = [points_inside_box(points, np.array(b["T_camera0_box"]), b["length"], b["width"],
                                b["height"], margin) for b in boxes]

    print(f"\n  Segmentation: found {len(object_indices)} objects, ground truth has {len(boxes)} boxes")
    print("  object | points | box | precision | recall")
    for k, idx in enumerate(object_indices):
        counts = [mask[idx].sum() for mask in inside]
        best = int(np.argmax(counts))
        precision = counts[best] / len(idx)
        recall = counts[best] / max(inside[best].sum(), 1)
        print(f"  {k:6d} | {len(idx):6d} | {best:3d} | {precision:9.3f} | {recall:6.3f}")


# ---------------------------------------------------------------------------
# Part 3 - boxes
# ---------------------------------------------------------------------------
def box_in_table_frame(T_camera0_box, length, width, height, T_camera0_table):
    """(x, y, z_base, yaw, length, width, height) of a box, in our table frame."""
    T = np.linalg.inv(T_camera0_table) @ np.array(T_camera0_box)
    yaw = np.arctan2(T[1, 0], T[0, 0])
    return T[0, 3], T[1, 3], T[2, 3] - height / 2, yaw, length, width, height


def iou_3d(a, b):
    """3D IoU of two z-up boxes (x, y, z_base, yaw, length, width, height):
    area of the intersection of the two rectangles seen from above x overlap in height."""
    import cv2
    rect_a = ((a[0], a[1]), (a[4], a[5]), np.degrees(a[3]))
    rect_b = ((b[0], b[1]), (b[4], b[5]), np.degrees(b[3]))
    kind, polygon = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    area = cv2.contourArea(cv2.convexHull(polygon)) if kind != cv2.INTERSECT_NONE else 0.0
    overlap_z = max(0.0, min(a[2] + a[6], b[2] + b[6]) - max(a[2], b[2]))
    intersection = area * overlap_z
    union = a[4] * a[5] * a[6] + b[4] * b[5] * b[6] - intersection
    return intersection / union


def box_errors(est, true):
    """Errors with the box symmetries: length/width can swap (yaw + 90 deg) and yaw is
    defined modulo 180 deg. Returns the version with the smallest yaw error."""
    centre_est = np.array([est[0], est[1], est[2] + est[6] / 2])
    centre_true = np.array([true[0], true[1], true[2] + true[6] / 2])
    best = None
    for swap in (False, True):
        length, width = (true[5], true[4]) if swap else (true[4], true[5])
        yaw_true = true[3] + (np.pi / 2 if swap else 0.0)
        yaw_err = (est[3] - yaw_true + np.pi / 2) % np.pi - np.pi / 2   # modulo 180 deg
        candidate = (np.degrees(yaw_err), (est[4] - length), (est[5] - width))
        if best is None or abs(candidate[0]) < abs(best[0]):
            best = candidate
    return {"position_mm": np.linalg.norm(centre_est - centre_true) * 1000,
            "z_mm": (est[2] - true[2]) * 1000,
            "yaw_deg": best[0],
            "length_mm": best[1] * 1000, "width_mm": best[2] * 1000,
            "height_mm": (est[6] - true[6]) * 1000}


def print_box_errors(result_boxes, ground_truth, T_camera0_table, title):
    """Matches each estimated box to the true box with the highest IoU and prints the errors.
    Missing boxes count with IoU 0 (as on the presentation day)."""
    trues = [box_in_table_frame(b["T_camera0_box"], b["length"], b["width"], b["height"], T_camera0_table)
             for b in ground_truth["boxes"]]
    ests = [box_in_table_frame(b["T_camera0_box"], b["length"], b["width"], b["height"], T_camera0_table)
            for b in result_boxes]

    print(f"\n  {title}")
    print("  box | IoU   | pos (mm) | z (mm) | yaw (deg) | length (mm) | width (mm) | height (mm)")
    ious = []
    used = set()
    for k, true in enumerate(trues):
        options = [(iou_3d(e, true), j) for j, e in enumerate(ests) if j not in used]
        if not options or max(options)[0] == 0:
            print(f"  {k+1:3d} | 0.000 | not found")
            ious.append(0.0)
            continue
        iou, j = max(options)
        used.add(j)
        ious.append(iou)
        e = box_errors(ests[j], true)
        print(f"  {k+1:3d} | {iou:.3f} | {e['position_mm']:8.1f} | {e['z_mm']:+6.1f} | {e['yaw_deg']:+9.2f} | "
              f"{e['length_mm']:+11.1f} | {e['width_mm']:+10.1f} | {e['height_mm']:+11.1f}")
    if ious:
        print(f"  mean IoU: {np.mean(ious):.3f}")
    else:
        print("  no boxes in the ground truth")
    return ious
