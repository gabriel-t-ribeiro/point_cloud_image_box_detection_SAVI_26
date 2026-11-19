"""Dinis's hand-written RANSAC (from the first gw/main.py), kept for comparison
with Open3D's registration_ransac_based_on_feature_matching.

Run it with:  ./main.py --scene scenes/level_1 --num_boxes 1 --registration custom

Same idea as Open3D's version:
  1. Match FPFH features (mutual nearest neighbours).
  2. Repeat: pick 3 matches, reject bad samples (edge lengths, normals),
     estimate R, t with Kabsch (SVD), count the inliers.
  3. Keep the best T. Stop early when we are RANSAC_CONFIDENCE sure we found it.

What was changed from the first version (why it was "not fully working"):
  - The normals now have a consistent sign (oriented towards the camera, in
    registration.preprocess_for_features). Before, each normal had a random sign,
    so the normal check rejected many correct samples.
  - The distance-to-origin weights are off by default. The origin is camera 0,
    not the table, so they favoured points close to camera 0 instead of the table.
  - Feature matching uses a k-d tree over all features at once (scipy) instead
    of a Python loop per point (much faster, same result).
"""

import numpy as np
from scipy.spatial import cKDTree

from registration import RANSAC_CONFIDENCE, RANSAC_DISTANCE, RANSAC_MAX_ITERATIONS

USE_DISTANCE_WEIGHTS = False   # True = Dinis's original weighting (see above)
WEIGHT_SCALE = 1.5             # distance (m) at which a point's weight drops to 0.5
MAX_NORMAL_ANGLE = 0.52        # rad (~30 deg) between matched normals after rotation
EDGE_SIMILARITY = 0.9          # like CorrespondenceCheckerBasedOnEdgeLength(0.9)


def match_features(source_fpfh, target_fpfh):
    """Pairs (i, j) where i and j are each other's nearest neighbour in FPFH space."""
    source_features = source_fpfh.data.T   # (N, 33)
    target_features = target_fpfh.data.T   # (M, 33)
    _, source_to_target = cKDTree(target_features).query(source_features)
    _, target_to_source = cKDTree(source_features).query(target_features)
    i = np.arange(len(source_features))
    mutual = target_to_source[source_to_target] == i
    return np.column_stack([i[mutual], source_to_target[mutual]])


def estimate_rigid_transform(src, dst):
    """Kabsch: rigid transformation (R, t) that best moves the points src onto dst."""
    src_center, dst_center = src.mean(axis=0), dst.mean(axis=0)
    U, _, Vt = np.linalg.svd((src - src_center).T @ (dst - dst_center))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])  # avoids a reflection
    R = Vt.T @ D @ U.T

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = dst_center - R @ src_center
    return T


RNG = np.random.default_rng(0)  # created once: repeatable runs, but each call draws new samples


def global_registration_custom(source_down, target_down, source_fpfh, target_fpfh):
    rng = RNG

    pairs = match_features(source_fpfh, target_fpfh)
    if len(pairs) < 3:
        raise ValueError("Not enough feature matches to run RANSAC")

    src = np.asarray(source_down.points)[pairs[:, 0]]
    dst = np.asarray(target_down.points)[pairs[:, 1]]
    src_normals = np.asarray(source_down.normals)[pairs[:, 0]]
    dst_normals = np.asarray(target_down.normals)[pairs[:, 1]]

    if USE_DISTANCE_WEIGHTS:
        weights = 1.0 / (1.0 + (np.linalg.norm(dst, axis=1) / WEIGHT_SCALE) ** 2)
    else:
        weights = np.ones(len(pairs))
    probabilities = weights / weights.sum()

    best_T = np.eye(4)
    best_score = 0.0
    max_iterations = RANSAC_MAX_ITERATIONS
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # 1. Sample 3 matches
        idx = rng.choice(len(pairs), 3, replace=False, p=probabilities)
        a, b = src[idx], dst[idx]

        # 2. Edge-length check: a rigid motion keeps distances, so the triangles must match
        src_edges = np.linalg.norm(a[[0, 0, 1]] - a[[1, 2, 2]], axis=1)
        dst_edges = np.linalg.norm(b[[0, 0, 1]] - b[[1, 2, 2]], axis=1)
        if np.any(src_edges < EDGE_SIMILARITY * dst_edges) or np.any(dst_edges < EDGE_SIMILARITY * src_edges):
            continue

        # 3. Kabsch
        T = estimate_rigid_transform(a, b)

        # 4. Normal check: after rotating, matched normals must point the same way
        rotated_normals = src_normals[idx] @ T[:3, :3].T
        if np.any(np.sum(rotated_normals * dst_normals[idx], axis=1) < np.cos(MAX_NORMAL_ANGLE)):
            continue

        # 5. Score = (weighted) number of matches that land closer than RANSAC_DISTANCE
        moved = src @ T[:3, :3].T + T[:3, 3]
        inliers = np.linalg.norm(moved - dst, axis=1) < RANSAC_DISTANCE
        score = weights[inliers].sum()

        if score > best_score:
            best_score, best_T = score, T

            # 6. Iterations needed to be RANSAC_CONFIDENCE sure: log(1 - p) / log(1 - w^3)
            ratio = inliers.mean()
            needed = 0 if ratio >= 1.0 else np.log(1.0 - RANSAC_CONFIDENCE) / np.log(1.0 - ratio ** 3)
            max_iterations = min(RANSAC_MAX_ITERATIONS, needed)

    return best_T
