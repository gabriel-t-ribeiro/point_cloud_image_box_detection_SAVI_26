"""Part 3 - base method: bounding box of each segmented object (reference only).

The assignment suggests Open3D's get_oriented_bounding_box(), but its z axis does
not have to be perpendicular to the table. Our boxes always stand on the table, so
we work in the table frame (z up) and only need a 2D rectangle seen from above:

  - cv2.minAreaRect on the (x, y) of the points -> centre, length, width, yaw
  - height from the z of the points

To be fair to this method (as the assignment asks) we first remove outliers and
use percentiles instead of min / max, so a few flying pixels do not grow the box.

The main method of the assignment (fitting with scipy.optimize.least_squares) is
still to do; this gives us a first result.json and numbers to compare against.
"""

import cv2
import numpy as np
import open3d as o3d

from segmentation import to_frame

OUTLIER_NEIGHBOURS = 20      # statistical outlier removal before the box
OUTLIER_STD = 2.0
PERCENTILE = 1.0             # ignore the PERCENTILE % most extreme points on each side (height)
NORMAL_RADIUS = 0.01         # radius (m) for the normals; raise it if the normals look too noisy
MIN_WALL_POINTS = 30         # fewer points than this on a wall -> fall back to a percentile


class Box:
    """A box in the table frame: centre (x, y), base height z, yaw (rad), dimensions."""

    def __init__(self, x, y, z, yaw, length, width, height):
        self.x, self.y, self.z, self.yaw = x, y, z, yaw
        self.length, self.width, self.height = length, width, height

    def T_table_box(self):
        """Box frame: origin at the box centre, x along length, z up."""
        T = np.eye(4)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        T[:3, 3] = [self.x, self.y, self.z + self.height / 2]
        return T

    def __repr__(self):
        return (f"Box(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, yaw={np.degrees(self.yaw):.1f} deg, "
                f"{self.length:.3f} x {self.width:.3f} x {self.height:.3f})")


def point_normals(p):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(p)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=30))
    return np.asarray(cloud.normals)


def wall_yaw(normals_xy):
    """Common direction of the 4 walls: the normals are at yaw + k*90 deg, so average 4*angle."""
    theta = np.arctan2(normals_xy[:, 1], normals_xy[:, 0])
    yaw = np.angle(np.sum(np.exp(4j * theta))) / 4
    for _ in range(2):    # second pass without the normals that disagree (noise, corners)
        d = (theta - yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        keep = np.abs(d) < np.radians(10)
        yaw += np.angle(np.sum(np.exp(4j * d[keep]))) / 4
    return yaw


def wall_position(proj, facing, centre, positive_side):
    """Position of one wall along an axis: median of the points facing that way on that side."""
    side = proj > centre if positive_side else proj < centre
    chosen = facing & side
    if chosen.sum() >= MIN_WALL_POINTS:
        return np.median(proj[chosen])
    return np.percentile(proj, 99 if positive_side else 1)       # wall not seen: fall back

def top_plane(z):
    """Height of the top face: peak of the z histogram in the highest 3 cm (not an extreme value)."""
    z_high = np.percentile(z, 99.5)
    zs = z[z > z_high - 0.03]
    bins = np.arange(zs.min(), zs.max() + 0.001, 0.001)
    counts, bins = np.histogram(zs, bins=bins)
    counts = np.convolve(counts, np.ones(3) / 3, mode="same")
    return bins[np.argmax(counts)] + 0.0005


def bounding_box(object_cloud, T_camera0_table):
    """z-up box of one object (cloud in camera0 frame) from its 5 planes, in the table frame."""
    cloud, _ = object_cloud.remove_statistical_outlier(nb_neighbors=OUTLIER_NEIGHBOURS, std_ratio=OUTLIER_STD)
    p = to_frame(np.asarray(cloud.points), T_camera0_table)
    n = point_normals(p)

    # Top plane: points with a vertical normal, consensus height
    horizontal = (np.abs(n[:, 2]) > 0.95) & (p[:, 2] > 0.02)
    z0 = top_plane(p[horizontal, 2])
    top = np.median(p[horizontal & (np.abs(p[:, 2] - z0) < 0.004), 2])
    bottom = max(0.0, np.percentile(p[:, 2], PERCENTILE))
    if bottom < 0.02:
        bottom = 0.0

    # Side walls: points with a horizontal normal, away from the top and bottom edges
    wall = (np.abs(n[:, 2]) < 0.3) & (p[:, 2] > bottom + 0.01) & (p[:, 2] < top - 0.01)
    q = p[wall, :2]
    nxy = n[wall, :2] / np.linalg.norm(n[wall, :2], axis=1, keepdims=True)
    yaw = wall_yaw(nxy)

    c, s = np.cos(yaw), np.sin(yaw)
    M = np.array([[c, -s], [s, c]])               # columns: u (length axis), v (width axis)
    centre = q.mean(axis=0)
    edges = []
    for axis in (M[:, 0], M[:, 1]):
        proj = q @ axis
        facing = np.abs(nxy @ axis) > 0.8         # normal along this axis
        lo = wall_position(proj, facing, centre @ axis, positive_side=False)
        hi = wall_position(proj, facing, centre @ axis, positive_side=True)
        edges.append((lo, hi))
    (u0, u1), (v0, v1) = edges
    cx, cy = M @ [(u0 + u1) / 2, (v0 + v1) / 2]
    return Box(cx, cy, bottom, yaw, u1 - u0, v1 - v0, top - bottom)


def boxes_to_result(boxes, T_camera0_table):
    """The 'boxes' list of result.json."""
    return [{"T_camera0_box": (T_camera0_table @ box.T_table_box()).tolist(),
             "length": float(box.length), "width": float(box.width), "height": float(box.height)}
            for box in boxes]


def object_clouds(aggregated, segmentation):
    return [aggregated.select_by_index(idx) for idx in segmentation.object_indices]
