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
  5. Colour (level 4): touching boxes of different colours form one euclidean
     cluster, so each cluster is split by colour. We use chromaticity
     (r, g) = (R, G) / (R + G + B): it removes the brightness, so the top and the
     sides of a box (lit differently) get the same value. The colours present are
     the peaks of the 2D chromaticity histogram; each point takes the colour of the
     nearest peak, or none if it is too far from every peak (colours mixed at the
     seams between boxes). A cluster with a single peak is left as it is.

All indices returned are indices into the aggregated cloud, so the same points can
be used later by the box fitting (Part 3) and by the evaluation.
"""

import numpy as np
import open3d as o3d
from scipy import ndimage

# ---------------------------------------------------------------------------
# Parameters (meters)
# ---------------------------------------------------------------------------
PLANE_VOXEL = 0.01            # downsampling used only to find the planes (faster)
PLANE_DISTANCE = 0.01         # RANSAC inlier distance for the planes
MAX_PLANES = 8                # how many planes to try before giving up
MAX_TILT_DEG = 45             # max angle between a plane normal and "up" to call it horizontal
MIN_TABLE_AREA = 0.25         # m^2; smaller horizontal planes (e.g. box tops) are not the table

TABLE_CLUSTER_EPS = 0.03      # DBSCAN on the table inliers
PLANE_REFIT_ITERS = 5         # iterative SVD refits to remove initial RANSAC plane tilt
MIN_HEIGHT = 0.008            # points closer than this to the table top are the table itself
MAX_HEIGHT = 0.6              # ignore anything higher above the table
EDGE_MARGIN = 0.01            # shrink the table limits a bit (table edge is noisy)

OBJECT_CLUSTER_EPS = 0.015    # DBSCAN eps for the objects (max gap inside one object)
OBJECT_MIN_POINTS = 30        # DBSCAN min_points
MIN_OBJECT_POINTS = 500       # clusters smaller than this are noise, not boxes

# Colour split (chromaticity units: r, g in [0, 1])
COLOUR_SPLIT = True           # False = euclidean clustering only (levels 1-3 behaviour)
COLOUR_BIN = 0.01             # histogram bin size (chromaticity noise of one box is ~0.01)
COLOUR_SMOOTH = 1.0           # Gaussian smoothing of the histogram (in bins)
COLOUR_MIN_PEAK = 0.003       # a colour peak must hold >= 0.3% of the cluster's points (real ~2-9%, seams ~0.01%)
COLOUR_MAX_DISTANCE = 0.06    # farther than this from every peak = no clear colour (seam mixing)
COLOUR_MIN_BRIGHTNESS = 0.15  # R + G + B below this: too dark, chromaticity is unreliable


class Segmentation:
    """Result of segment_objects()."""

    def __init__(self, T_camera0_table, table_indices, object_indices, table_size, no_colour_indices=None):
        self.T_camera0_table = T_camera0_table   # 4x4, table frame: origin at the centre, z up
        self.table_indices = table_indices       # points of the table top
        self.object_indices = object_indices     # list, one index array per object (biggest first)
        self.table_size = table_size             # (length along x, width along y)
        # points of colour-split clusters that matched no colour (left out of object_indices)
        self.no_colour_indices = no_colour_indices if no_colour_indices is not None else np.array([], int)
        self.colour_split = False                # True if some cluster was split by colour


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

    1. Iteratively collect points within PLANE_DISTANCE of the current plane, keep
       the biggest DBSCAN cluster (= table top), and refit the plane by least
       squares (SVD: the normal is the direction of least variance).
       Why iterate? RANSAC's 3-point plane with a ±1 cm slab can be tilted by
       1–2 deg, so a single pass clips one edge of the table and includes the
       bottom of the boxes on the other side, leaving a ~0.5–1 deg residual tilt.
       Over a 1.25 m table, even 0.8 deg raises one end by ~9 mm > MIN_HEIGHT,
       causing part of the table to be detected as a box. Repeating 3–5 times
       converges to < 0.02 deg.
    2. Seen from above, the table is a rectangle: cv2.minAreaRect gives its centre,
       size and direction. x = long side, z = normal (up), y = z cross x.
    """
    import cv2

    for _ in range(PLANE_REFIT_ITERS):
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
        new_d = -float(z_axis @ centre)

        angle_change = np.arccos(np.clip(z_axis @ normal, -1.0, 1.0))
        offset_change = abs(new_d - d)
        normal, d = z_axis, new_d
        if angle_change < 1e-4 and offset_change < 1e-4:
            break

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
    clusters = euclidean_clusters(aggregated, candidate_indices)

    # Colour: split clusters that contain more than one colour
    object_indices, no_colour, n_split = [], [], 0
    colours = np.asarray(aggregated.colors)
    for idx in clusters:
        groups, rejected = split_by_colour(colours[idx]) if COLOUR_SPLIT and aggregated.has_colors() else ([], [])
        if len(groups) <= 1:
            object_indices.append(idx)
            continue
        no_colour.append(idx[rejected])
        n_split += 1
        for group in groups:   # a colour may appear in separate pieces (red | blue | red): cluster again
            object_indices += euclidean_clusters(aggregated, idx[group])
        if verbose:
            print(f"  cluster of {len(idx)} points split by colour into {len(groups)} "
                  f"({100 * len(rejected) / len(idx):.1f}% of its points with no clear colour)")
    object_indices.sort(key=len, reverse=True)
    no_colour = np.concatenate(no_colour) if no_colour else np.array([], int)

    if verbose:
        print(f"  table {table_size[0]:.2f} x {table_size[1]:.2f} m, {len(candidate_indices)} points above it, "
              f"{len(object_indices)} objects: {[len(i) for i in object_indices]} points")
    segmentation = Segmentation(T_camera0_table, table_indices, object_indices, table_size, no_colour)
    segmentation.colour_split = n_split > 0
    return segmentation


def euclidean_clusters(cloud, indices):
    """DBSCAN on the given points of the cloud; returns the clusters as index arrays into the cloud."""
    labels = np.array(cloud.select_by_index(indices).cluster_dbscan(eps=OBJECT_CLUSTER_EPS,
                                                                    min_points=OBJECT_MIN_POINTS))
    clusters = [indices[labels == label] for label in range(labels.max() + 1)]
    return [c for c in clusters if len(c) >= MIN_OBJECT_POINTS]


# ---------------------------------------------------------------------------
# 5. Colour
# ---------------------------------------------------------------------------
def chromaticity(rgb):
    """(r, g) = (R, G) / (R + G + B): colour without brightness. Returns (N, 2) and the brightness R + G + B."""
    brightness = rgb.sum(axis=1)
    return rgb[:, :2] / np.maximum(brightness, 1e-6)[:, None], brightness


def colour_peaks(chrom):
    """Peaks of the smoothed 2D chromaticity histogram holding >= COLOUR_MIN_PEAK of the points -> (P, 2)."""
    bins = int(round(1.0 / COLOUR_BIN))
    hist, _, _ = np.histogram2d(chrom[:, 0], chrom[:, 1], bins=bins, range=[[0, 1], [0, 1]])
    hist = ndimage.gaussian_filter(hist, COLOUR_SMOOTH)
    is_peak = (hist == ndimage.maximum_filter(hist, size=5)) & (hist >= COLOUR_MIN_PEAK * len(chrom))
    return (np.argwhere(is_peak) + 0.5) * COLOUR_BIN   # bin centres


def split_by_colour(rgb):
    """Splits the points of one cluster by colour.

    Returns (groups, rejected): one index array per colour peak (indices into rgb), and the
    indices of the points with no clear colour (too dark, or farther than COLOUR_MAX_DISTANCE
    from every peak). With a single peak, groups has one element and nothing should change.
    """
    chrom, brightness = chromaticity(rgb)
    bright = brightness >= COLOUR_MIN_BRIGHTNESS
    peaks = colour_peaks(chrom[bright])
    if len(peaks) <= 1:
        return [np.arange(len(rgb))], np.array([], int)
    distances = np.linalg.norm(chrom[:, None, :] - peaks[None, :, :], axis=2)   # (N, P)
    nearest = np.argmin(distances, axis=1)
    clear = bright & (distances[np.arange(len(rgb)), nearest] < COLOUR_MAX_DISTANCE)
    groups = [np.where(clear & (nearest == p))[0] for p in range(len(peaks))]
    return groups, np.where(~clear)[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_frame(points, T_camera0_frame):
    """Points in camera0 coordinates -> coordinates in the given frame: p_frame = inv(T) p_camera0."""
    T = np.linalg.inv(T_camera0_frame)
    return points @ T[:3, :3].T + T[:3, 3]
