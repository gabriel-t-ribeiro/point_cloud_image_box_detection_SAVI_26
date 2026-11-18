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


def bounding_box(object_cloud, T_camera0_table):
    """z-up bounding box of one object (cloud in camera0 frame), in the table frame."""
    cloud, _ = object_cloud.remove_statistical_outlier(nb_neighbors=OUTLIER_NEIGHBOURS, std_ratio=OUTLIER_STD)
    p = to_frame(np.asarray(cloud.points), T_camera0_table)

    (cx, cy), (length, width), angle_deg = cv2.minAreaRect(p[:, :2].astype(np.float32))
    bottom = max(0.0, np.percentile(p[:, 2], PERCENTILE))
    top = np.percentile(p[:, 2], 100 - PERCENTILE)
    # Boxes on the table start at z = 0; points near the table are hidden or removed with it
    if bottom < 0.02:
        bottom = 0.0
    return Box(cx, cy, bottom, np.radians(angle_deg), length, width, top - bottom)


def boxes_to_result(boxes, T_camera0_table):
    """The 'boxes' list of result.json."""
    return [{"T_camera0_box": (T_camera0_table @ box.T_table_box()).tolist(),
             "length": float(box.length), "width": float(box.width), "height": float(box.height)}
            for box in boxes]


def object_clouds(aggregated, segmentation):
    return [aggregated.select_by_index(idx) for idx in segmentation.object_indices]
