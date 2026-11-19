"""Extra - photometric correction (scenes generated with --exposure_changes).

With automatic exposure every image has its own gain s and bias b, the same for R, G and B:
    colour seen in image i = s_i * true colour + b_i
so the same surface looks brighter in some images and darker in others, and the colours of
the aggregated cloud do not agree (bad for the colour segmentation of level 4).

We estimate each image relative to image 000 (the reference, as in Parte 06):
    c_i = s_i0 * c_0 + b_i0,     s_i0 = s_i / s_0,   b_i0 = b_i - s_i0 * b_0
and correct image i to the appearance of image 000:  c_corrected = (c_i - b_i0) / s_i0.

Least squares (Parte 06, ex. 2): the data are the points seen by two cameras. After the
registration, two points of different cameras closer than PAIR_DISTANCE are the same surface.
Image j's colour is mapped into image i (c_i = s_i * c + b_i and c_j = s_j * c + b_j give
c_i = s_i / s_j * (c_j - b_j) + b_i), and the residual is measured in image i's raw colours,
one per pair and per channel:
    c_i  -  [ s_i0 / s_j0 * (c_j - b_j0) + b_i0 ].
Why not compare the corrected colours, (c_i - b_i0)/s_i0 - (c_j - b_j0)/s_j0? Because dividing
by the unknown s rewards a larger s: compressing the colours of all the free images makes every
difference smaller (we measured s ~ 1.03, b ~ -0.02 on a scene with no exposure changes at all).
Image 000 stays fixed (s = 1, b = 0): if every image could change, the solution is not unique
(Parte 06, ex. 2, step 1). Colours clipped by the camera (near 0 or 1) do not follow the linear
model and are left out; soft_l1 limits the pairs on colour edges.
"""

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

VOXEL = 0.01              # downsampling of each camera cloud for the estimation (colours are averaged)
PAIR_DISTANCE = 0.005     # points of two cameras closer than this see the same surface (m)
MAX_PAIRS = 3000          # random subsample of the pairs of each camera pair
MIN_PAIRS = 100           # camera pairs with fewer pairs than this (little overlap) are not used
CLIPPED = (0.02, 0.98)    # colours outside this range may be clipped by the camera: not used
F_SCALE = 0.03            # soft_l1 scale, in colour units (colour noise is ~0.01)


def colour_pairs(clouds, transforms, rng):
    """[(i, j, colours_i (M, 3), colours_j (M, 3))] for every camera pair that overlaps."""
    down = []
    for cloud, T in zip(clouds, transforms):
        d = cloud.voxel_down_sample(VOXEL)
        d.transform(T)                                    # camera i -> camera 0 frame
        down.append((np.asarray(d.points), np.asarray(d.colors)))
    pairs = []
    for j in range(len(down)):
        tree = cKDTree(down[j][0])
        for i in range(j + 1, len(down)):
            dist, nearest = tree.query(down[i][0], distance_upper_bound=PAIR_DISTANCE)
            ok = np.isfinite(dist)
            ci, cj = down[i][1][ok], down[j][1][nearest[ok]]
            unclipped = np.all((ci > CLIPPED[0]) & (ci < CLIPPED[1]) & (cj > CLIPPED[0]) & (cj < CLIPPED[1]), axis=1)
            ci, cj = ci[unclipped], cj[unclipped]
            if len(ci) < MIN_PAIRS:
                continue
            if len(ci) > MAX_PAIRS:
                keep = rng.choice(len(ci), MAX_PAIRS, replace=False)
                ci, cj = ci[keep], cj[keep]
            pairs.append((i, j, ci, cj))
    return pairs


def estimate_exposure(clouds, transforms, seed=0):
    """Gain and bias of every image relative to image 000: arrays s (n,), b (n,), s[0] = 1, b[0] = 0."""
    n = len(clouds)
    pairs = colour_pairs(clouds, transforms, np.random.default_rng(seed))
    if n < 2 or not pairs:
        return np.ones(n), np.zeros(n)

    def unpack(params):
        s = np.concatenate([[1.0], params[0::2]])
        b = np.concatenate([[0.0], params[1::2]])
        return s, b

    def residuals(params):     # image j mapped into image i, compared in image i's raw colours
        s, b = unpack(params)
        return np.concatenate([(ci - (s[i] / s[j] * (cj - b[j]) + b[i])).ravel() for i, j, ci, cj in pairs])

    x0 = np.tile([1.0, 0.0], n - 1)
    lb = np.tile([0.2, -0.5], n - 1)
    ub = np.tile([5.0, 0.5], n - 1)
    result = least_squares(residuals, x0, bounds=(lb, ub), loss="soft_l1", f_scale=F_SCALE)
    return unpack(result.x)


def correct_colours(clouds, s, b):
    """Copies of the clouds with every image corrected to the appearance of image 000."""
    corrected = []
    for cloud, si, bi in zip(clouds, s, b):
        c = o3d.geometry.PointCloud(cloud)
        c.colors = o3d.utility.Vector3dVector(np.clip((np.asarray(cloud.colors) - bi) / si, 0.0, 1.0))
        corrected.append(c)
    return corrected


def true_relative_exposure(ground_truth, names):
    """(s_i0, b_i0) from the ground truth's photometry field, or None if the scene has none."""
    by_image = {p["image"]: (p["s"], p["b"]) for p in ground_truth.get("photometry", [])}
    if not by_image or names[0] not in by_image:
        return None
    s0, b0 = by_image[names[0]]
    s = np.array([by_image[n][0] / s0 for n in names])
    b = np.array([by_image[n][1] - (by_image[n][0] / s0) * b0 for n in names])
    return s, b
