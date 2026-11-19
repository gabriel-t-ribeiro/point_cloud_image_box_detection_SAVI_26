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

from segmentation import to_frame

OUTLIER_NEIGHBOURS = 20      # statistical outlier removal before the box
OUTLIER_STD = 2.0
PERCENTILE = 1.0             # ignore the PERCENTILE % most extreme points on each side (height)
EDGE_BIN = 0.001             # histogram bin (m) to find the box edges
TOP_SLAB = 0.01              # only the top 1 cm of the object is used for length / width


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


def edge_positions(u, bin_size=EDGE_BIN):
    """Edges of a set of 1D coordinates: where the density falls to half of the plateau.

    Unlike min / max (or percentiles), a few stray points cannot move this edge, and
    for a blurred step the half-maximum is the true edge position.
    """
    bins = np.arange(u.min(), u.max() + bin_size, bin_size)
    counts, bins = np.histogram(u, bins=bins)
    counts = np.convolve(counts, np.ones(5) / 5, mode="same")        # light smoothing
    n = len(counts)
    plateau = np.median(counts[n // 4: 3 * n // 4])                  # density on the middle of the face
    above = np.where(counts >= 0.5 * plateau)[0]
    return bins[above[0]], bins[above[-1] + 1]


def bounding_box(object_cloud, T_camera0_table):
    """z-up bounding box of one object (cloud in camera0 frame), in the table frame."""
    cloud, _ = object_cloud.remove_statistical_outlier(nb_neighbors=OUTLIER_NEIGHBOURS, std_ratio=OUTLIER_STD)
    p = to_frame(np.asarray(cloud.points), T_camera0_table)

    # Calculate Yaw
    _, _, angle_deg = cv2.minAreaRect(p[:, :2].astype(np.float32))
    yaw = np.radians(angle_deg)

    # Calculate Yaw
    bottom = max(0.0, np.percentile(p[:, 2], PERCENTILE))
    top = np.percentile(p[:, 2], 100 - PERCENTILE)
    if bottom < 0.02:
        bottom = 0.0

    # Calculate length / width / centre from the top slab, in the box frame
    slab = p[p[:, 2] > top - TOP_SLAB]
    c, s = np.cos(yaw), np.sin(yaw)
    M = np.array([[c, -s], [s, c]])            # box frame -> table frame
    uv = slab[:, :2] @ M                       # table frame -> box frame (u along length, v along width)
    u0, u1 = edge_positions(uv[:, 0])
    v0, v1 = edge_positions(uv[:, 1])
    cx, cy = M @ [(u0 + u1) / 2, (v0 + v1) / 2]
    return Box(cx, cy, bottom, yaw, u1 - u0, v1 - v0, top - bottom)


def boxes_to_result(boxes, T_camera0_table):
    """The 'boxes' list of result.json."""
    return [{"T_camera0_box": (T_camera0_table @ box.T_table_box()).tolist(),
             "length": float(box.length), "width": float(box.width), "height": float(box.height)}
            for box in boxes]


def object_clouds(aggregated, segmentation):
    return [aggregated.select_by_index(idx) for idx in segmentation.object_indices]
