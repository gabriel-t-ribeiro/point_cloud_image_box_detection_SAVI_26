"""Part 2 - Segmentation of the objects on top of the table, on the AGGREGATED cloud.

Based on our part7/ex6_part7.py (plane RANSAC + DBSCAN) and the professor's
Parte08/ex4/main.py (table frame + crop above the table), adapted to these scenes:

  1. Find the table plane with RANSAC (segment_plane). The scene has many planes
     (floor, walls, furniture), so we look for several and pick the table:
       - it is horizontal: its normal is close to "up". Camera 0 is roughly level
         and its y axis points down, so up is about -y of camera 0;
       - among the big horizontal planes (area > MIN_TABLE_AREA, so not a box top),
         it is the first one hit by the optical axis of camera 0 (all cameras look
         at the table, the floor is behind it). We cannot just take the biggest
         plane: the floor has about 10x more points than the table.
  2. Keep the biggest DBSCAN cluster of the plane inliers (drops other furniture at
     the same height), refit the plane, and build a table frame: origin at the
     table centre, z up, x along the table's long side (minimum-area rectangle).
  3. Keep the points above the table (z > MIN_HEIGHT) and inside its limits.
  4. Separate the objects with euclidean clustering (DBSCAN).

All indices returned are indices into the aggregated cloud, so the same points can
be used later by the box fitting (Part 3) and by the evaluation.
"""

import numpy as np
import open3d as o3d

# ---------------------------------------------------------------------------
# Parameters (meters)
# ---------------------------------------------------------------------------
PLANE_VOXEL = 0.01            # downsampling used only to find the planes (faster)
PLANE_DISTANCE = 0.01         # RANSAC inlier distance for the planes
MAX_PLANES = 8                # how many planes to try before giving up
MAX_TILT_DEG = 45             # max angle between a plane normal and "up" to call it horizontal
MIN_TABLE_AREA = 0.25         # m^2; smaller horizontal planes (e.g. box tops) are not the table

TABLE_CLUSTER_EPS = 0.03      # DBSCAN on the table inliers
MIN_HEIGHT = 0.008            # points closer than this to the table top are the table itself
MAX_HEIGHT = 0.6              # ignore anything higher above the table
EDGE_MARGIN = 0.01            # shrink the table limits a bit (table edge is noisy)

OBJECT_CLUSTER_EPS = 0.015    # DBSCAN eps for the objects (max gap inside one object)
OBJECT_MIN_POINTS = 30        # DBSCAN min_points
MIN_OBJECT_POINTS = 500       # clusters smaller than this are noise, not boxes


class Segmentation:
    """Result of segment_objects()."""

    def __init__(self, T_camera0_table, table_indices, object_indices, table_size):
        self.T_camera0_table = T_camera0_table   # 4x4, table frame: origin at the centre, z up
        self.table_indices = table_indices       # points of the table top
        self.object_indices = object_indices     # list, one index array per object (biggest first)
        self.table_size = table_size             # (length along x, width along y)


# ---------------------------------------------------------------------------
# 1. Table plane
# ---------------------------------------------------------------------------
def find_table_plane(cloud, up_guess=np.array([0.0, -1.0, 0.0]), verbose=True):
    """Returns (normal, d) of the table plane n.p + d = 0, with n pointing up."""
    remaining = cloud.voxel_down_sample(PLANE_VOXEL)
    candidates = []   # (inliers, distance along camera 0's optical axis, normal, d)

    for _ in range(MAX_PLANES):
        if len(remaining.points) < 100:
            break
        (a, b, c, d), inliers = remaining.segment_plane(PLANE_DISTANCE, 3, 1000)
        normal = np.array([a, b, c])
        if normal @ up_guess < 0:          # make the normal point up
            normal, d = -normal, -d

        tilt = np.degrees(np.arccos(np.clip(normal @ up_guess, -1, 1)))
        if tilt < MAX_TILT_DEG and abs(normal[2]) > 1e-6:
            # Optical axis of camera 0: points t * (0, 0, 1). Hits the plane at t = -d / n_z
            t = -d / normal[2]
            if t > 0:
                candidates.append((len(inliers), t, normal, d))
        remaining = remaining.select_by_index(inliers, invert=True)

    if not candidates:
        raise RuntimeError("No horizontal plane found: cannot find the table")

    # Each inlier of the 1 cm downsampled cloud covers about PLANE_VOXEL^2 of surface
    big_enough = [c for c in candidates if c[0] * PLANE_VOXEL ** 2 >= MIN_TABLE_AREA]
    if not big_enough:
        raise RuntimeError("No horizontal plane big enough to be the table")
    inliers, t, normal, d = min(big_enough, key=lambda c: c[1])   # first plane hit by the optical axis
    if verbose:
        print(f"  table plane: {len(candidates)} horizontal candidates, chosen one at {t:.2f} m "
              f"from camera 0 with {inliers} inliers")
    return normal, d


# ---------------------------------------------------------------------------
# 2. Table frame
# ---------------------------------------------------------------------------
def table_frame(points, normal, d):
    """Table points (biggest cluster on the plane) and T_camera0_table.

    1. Points close to the RANSAC plane -> biggest DBSCAN cluster = table top.
    2. Refit the plane to those points with least squares (SVD: the normal is the
       direction of least variance). RANSAC's plane is only as good as its 3 best
       points; a small tilt would leave part of the table "above the table".
    3. Seen from above, the table is a rectangle: cv2.minAreaRect gives its centre,
       size and direction. x = long side, z = normal (up), y = z cross x.
    """
    import cv2

    distance = points @ normal + d
    on_plane = np.where(np.abs(distance) < PLANE_DISTANCE)[0]

    plane_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[on_plane]))
    plane_cloud = plane_cloud.voxel_down_sample(PLANE_VOXEL / 2)
    labels = np.array(plane_cloud.cluster_dbscan(eps=TABLE_CLUSTER_EPS, min_points=10))
    biggest_label = np.bincount(labels[labels >= 0]).argmax()
    table_points = np.asarray(plane_cloud.points)[labels == biggest_label]

    # Least-squares plane: normal = last right singular vector of the centred points
    centre = table_points.mean(axis=0)
    _, _, Vt = np.linalg.svd(table_points - centre, full_matrices=False)
    z_axis = Vt[2] if Vt[2] @ normal > 0 else -Vt[2]

    # 2D coordinates in the plane (any two perpendicular in-plane directions u, v)
    u = Vt[0]
    v = np.cross(z_axis, u)
    uv = np.column_stack([(table_points - centre) @ u, (table_points - centre) @ v]).astype(np.float32)
    (cu, cv), (size_a, size_b), angle_deg = cv2.minAreaRect(uv)

    # Direction of the side of length size_a, rotated by angle_deg from u
    angle = np.radians(angle_deg)
    side_a = np.cos(angle) * u + np.sin(angle) * v
    if size_a >= size_b:
        x_axis, table_size = side_a, np.array([size_a, size_b])
    else:
        x_axis, table_size = np.cross(z_axis, side_a), np.array([size_b, size_a])
    y_axis = np.cross(z_axis, x_axis)

    T_camera0_table = np.eye(4)
    T_camera0_table[:3, 0], T_camera0_table[:3, 1], T_camera0_table[:3, 2] = x_axis, y_axis, z_axis
    T_camera0_table[:3, 3] = centre + cu * u + cv * v
    return T_camera0_table, table_size


# ---------------------------------------------------------------------------
# 3 + 4. Objects
# ---------------------------------------------------------------------------
def segment_objects(aggregated, verbose=True):
    points = np.asarray(aggregated.points)
    normal, d = find_table_plane(aggregated, verbose=verbose)
    T_camera0_table, table_size = table_frame(points, normal, d)

    # Crop: above the table and within its limits
    p = to_frame(points, T_camera0_table)
    half = table_size / 2 - EDGE_MARGIN
    above = ((p[:, 2] > MIN_HEIGHT) & (p[:, 2] < MAX_HEIGHT) &
             (np.abs(p[:, 0]) < half[0]) & (np.abs(p[:, 1]) < half[1]))
    candidate_indices = np.where(above)[0]
    table_indices = np.where((np.abs(p[:, 2]) < PLANE_DISTANCE) &
                             (np.abs(p[:, 0]) < half[0]) & (np.abs(p[:, 1]) < half[1]))[0]

    # Euclidean clustering
    objects_cloud = aggregated.select_by_index(candidate_indices)
    labels = np.array(objects_cloud.cluster_dbscan(eps=OBJECT_CLUSTER_EPS, min_points=OBJECT_MIN_POINTS))
    object_indices = []
    for label in range(labels.max() + 1):
        idx = candidate_indices[labels == label]
        if len(idx) >= MIN_OBJECT_POINTS:
            object_indices.append(idx)
    object_indices.sort(key=len, reverse=True)

    if verbose:
        print(f"  table {table_size[0]:.2f} x {table_size[1]:.2f} m, {len(candidate_indices)} points above it, "
              f"{len(object_indices)} objects: {[len(i) for i in object_indices]} points")
    return Segmentation(T_camera0_table, table_indices, object_indices, table_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_frame(points, T_camera0_frame):
    """Points in camera0 coordinates -> coordinates in the given frame: p_frame = inv(T) p_camera0."""
    T = np.linalg.inv(T_camera0_frame)
    return points @ T[:3, :3].T + T[:3, 3]
