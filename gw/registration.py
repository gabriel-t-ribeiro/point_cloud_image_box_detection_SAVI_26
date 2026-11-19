"""Part 1 - Registration: align every camera's point cloud with camera 000.

Based on our Part 8 exercise 3 (FPFH features + RANSAC + ICP), adapted to the
scenes of the assignment.

Pipeline for each camera i (i = 1, 2, ...):
  1. Preprocess (downsample + normals + FPFH features).
  2. Initial estimate: RANSAC over FPFH matches between camera i and its neighbour
     i-1 (neighbours overlap the most), chained with the pose of i-1:
         T_camera0_camera_i = T_camera0_camera_(i-1) @ T_(i-1)_i
     The last camera is also RANSAC-matched against camera 000 (it closes the ring).
  3. Refine with point-to-plane ICP, coarse threshold first and then fine
     (two ICP passes improved our Part 8 results a lot). The ICP target is ALL the
     clouds already aligned, merged together, not only i-1. So the error of the
     chain 0 -> 1 -> 2 -> ... does not keep growing: each new cloud is corrected
     against everything we have (question in the assignment: "what happens to the
     error when we chain many images?").

Every point cloud of a scene is in the frame of the camera that acquired it, so
the camera is at the origin (0, 0, 0). We use that to orient the normals.
"""

import os

import numpy as np
import open3d as o3d

registration = o3d.pipelines.registration

# ---------------------------------------------------------------------------
# Parameters (meters). These are the ones worth experimenting with.
# ---------------------------------------------------------------------------
FEATURE_VOXEL = 0.05          # downsampling for the FPFH features and RANSAC
NORMAL_RADIUS = FEATURE_VOXEL * 2.0
FPFH_RADIUS = FEATURE_VOXEL * 5.0
RANSAC_DISTANCE = FEATURE_VOXEL * 1.5   # max distance for a RANSAC inlier
RANSAC_MAX_ITERATIONS = 100000
RANSAC_CONFIDENCE = 0.999
RANSAC_ATTEMPTS = 5           # RANSAC is random: run it a few times, ICP picks the best answer

ICP_VOXEL = 0.01              # ICP uses a lighter version of the FULL cloud (table + background)
ICP_THRESHOLDS = [0.04, 0.01]  # coarse pass, then fine pass (max correspondence distance)
ICP_MAX_ITERATIONS = 100
ICP_ROBUST_KERNEL = 0.0       # Tukey loss scale (m) in ICP; 0 = plain least squares
MAX_DEPTH = 0.0               # drop points farther than this from the camera (m); 0 = keep all
OUTLIER_NEIGHBOURS = 0        # statistical outlier removal (flying pixels); 0 = off
OUTLIER_STD = 2.0
REFINEMENT_PASSES = 3         # extra passes: each camera (000 too) re-aligned against ALL the others (0 = off)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_scene_clouds(scene_folder):
    """Returns the image names (e.g. '000') and the point clouds of a scene, in order."""
    folder = os.path.join(scene_folder, "point_clouds")
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(folder) if f.lower().endswith(".ply"))
    if not names:
        raise FileNotFoundError(f"No .ply files found in {folder}")
    clouds = [o3d.io.read_point_cloud(os.path.join(folder, name + ".ply")) for name in names]
    return names, clouds


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def estimate_normals(cloud, radius):
    """Normals with a consistent sign: all pointing towards the camera (at the origin).

    estimate_normals() alone gives each normal an arbitrary sign (n or -n).
    orient_normals_towards_camera_location() is fast and correct here because each
    cloud is still in its own camera frame. (Our Part 8 exercises used
    orient_normals_consistent_tangent_plane, which is very slow on 300k points.)
    """
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    cloud.orient_normals_towards_camera_location(np.zeros(3))


def filter_points(cloud):
    """Part 1 step 1: remove bad points before registration.

    - MAX_DEPTH: the depth noise grows with the square of the distance, so very far
      points (walls at 4-5 m) are the noisiest.
    - Statistical outlier removal: drops points whose mean distance to their
      neighbours is much larger than average (flying pixels at the object edges).
    """
    if MAX_DEPTH > 0:
        depth = np.linalg.norm(np.asarray(cloud.points), axis=1)
        cloud = cloud.select_by_index(np.where(depth < MAX_DEPTH)[0])
    if OUTLIER_NEIGHBOURS > 0:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=OUTLIER_NEIGHBOURS, std_ratio=OUTLIER_STD)
    return cloud


def preprocess_for_features(cloud):
    """Downsample + normals + FPFH (same steps as preprocess_point_cloud in part8_ex3)."""
    down = cloud.voxel_down_sample(FEATURE_VOXEL)
    estimate_normals(down, NORMAL_RADIUS)
    return down


def compute_fpfh(cloud_down):
    return registration.compute_fpfh_feature(
        cloud_down, o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_RADIUS, max_nn=100))


def preprocess_for_icp(cloud):
    down = filter_points(cloud.voxel_down_sample(ICP_VOXEL))
    estimate_normals(down, ICP_VOXEL * 4)
    return down


# ---------------------------------------------------------------------------
# Global registration (RANSAC) and refinement (ICP)
# ---------------------------------------------------------------------------
def global_registration_open3d(source_down, target_down, source_fpfh, target_fpfh):
    """RANSAC over FPFH matches, as in our part8_ex3 (execute_global_registration)."""
    result = registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=RANSAC_DISTANCE,
        estimation_method=registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  registration.CorrespondenceCheckerBasedOnDistance(RANSAC_DISTANCE)],
        criteria=registration.RANSACConvergenceCriteria(RANSAC_MAX_ITERATIONS, RANSAC_CONFIDENCE),
    )
    return result.transformation


def ransac_candidates(global_registration, source_down, target_down, source_fpfh, target_fpfh):
    """Runs RANSAC RANSAC_ATTEMPTS times and returns the DIFFERENT answers it gave.

    Why: in about 1 of 5 runs RANSAC returns a wrong T (in level_1, camera 005 rotated
    ~63 deg) with the same fitness as the right one, so the fitness cannot tell them
    apart. After ICP they are easy to tell apart (see register_scene), so we keep all
    the distinct answers and let ICP decide. Answers closer than 10 deg / 10 cm are
    considered the same (ICP will converge to the same place).
    """
    candidates = []
    for attempt in range(RANSAC_ATTEMPTS):
        T = global_registration(source_down, target_down, source_fpfh, target_fpfh)
        if not any(same_pose(T, other) for other in candidates):
            candidates.append(T)
    return candidates


def same_pose(T_a, T_b, max_angle_deg=10.0, max_distance=0.1):
    E = np.linalg.inv(T_a) @ T_b
    angle = np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))
    return angle < max_angle_deg and np.linalg.norm(E[:3, 3]) < max_distance


def refine_icp(source, target, T_initial):
    """Point-to-plane ICP, coarse to fine. Each pass starts where the previous ended."""
    if ICP_ROBUST_KERNEL > 0:
        # Robust kernel: points with a large residual get a small weight (Tukey)
        estimation = registration.TransformationEstimationPointToPlane(registration.TukeyLoss(k=ICP_ROBUST_KERNEL))
    else:
        estimation = registration.TransformationEstimationPointToPlane()
    T = T_initial
    result = None
    for threshold in ICP_THRESHOLDS:
        result = registration.registration_icp(
            source, target, threshold, T, estimation,
            registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITERATIONS),
        )
        T = result.transformation
    return T, result


# ---------------------------------------------------------------------------
# Whole scene
# ---------------------------------------------------------------------------
def register_scene(clouds, names, method="open3d", verbose=True):
    """Returns the list of T_camera0_camera (4x4), one per cloud.

    method: "open3d" -> Open3D RANSAC (default)
            "custom" -> Dinis's hand-written RANSAC (registration_custom.py), for comparison
    """
    if method == "custom":
        from registration_custom import global_registration_custom as global_registration
    else:
        global_registration = global_registration_open3d

    features_clouds = [preprocess_for_features(c) for c in clouds]
    features = [compute_fpfh(c) for c in features_clouds]
    icp_clouds = [preprocess_for_icp(c) for c in clouds]

    transforms = [np.eye(4)]  # camera 000 is the reference
    aligned_icp = [icp_clouds[0]]

    for i in range(1, len(clouds)):
        # Initial estimates: RANSAC between neighbours i -> j, chained to camera 0 (T_camera0_j @ T_j_i).
        # j = i-1 for every camera. The LAST camera is also next to camera 000 (the ring closes),
        # so it gets RANSAC candidates against 000 too: in level_3_stacked, RANSAC 007 -> 006 gave
        # only wrong answers (~1 m slides) while 007 -> 000 was right every time.
        neighbours = [i - 1]
        if i == len(clouds) - 1 and i - 1 != 0:
            neighbours.append(0)
        candidates = [(j, T_j_i) for j in neighbours
                      for T_j_i in ransac_candidates(global_registration, features_clouds[i], features_clouds[j],
                                                     features[i], features[j])]

        # Refinement: ICP against every cloud already aligned (normals are rotated by transform()).
        # If RANSAC gave more than one answer, keep the one with the best ICP fitness.
        target_icp = merge(aligned_icp)
        best = None
        for j, T_j_i in candidates:
            T, icp_result = refine_icp(icp_clouds[i], target_icp, transforms[j] @ T_j_i)
            if best is None or icp_result.fitness > best[1].fitness:
                best = (T, icp_result, j, T_j_i)
        T, icp_result, j, T_j_i = best

        if verbose:
            ransac_eval = registration.evaluate_registration(
                features_clouds[i], features_clouds[j], RANSAC_DISTANCE, T_j_i)
            print(f"  {names[i]}: RANSAC fitness {ransac_eval.fitness:.3f} ({len(candidates)} different answers, "
                  f"best from {names[j]}) | "
                  f"ICP fitness {icp_result.fitness:.3f} rmse {icp_result.inlier_rmse * 1000:.1f} mm")

        transforms.append(T)
        aligned_icp.append(transformed(icp_clouds[i], T))

    # Extra passes: now that every cloud is roughly in place, re-align each camera
    # against all the OTHER clouds (including the ones after it, e.g. 007 next to 000).
    # This closes the loop around the table and spreads out the drift of the chain.
    # Camera 000 is re-aligned too: if it stays fixed, cameras 1-7 agree with each other
    # but the whole group ends up ~0.4 deg rotated with respect to 000 (error grows with
    # the distance to camera 000, ~14 mm on the opposite side). Afterwards every pose is
    # re-expressed in the new camera-000 frame, T_i <- inv(T_0) @ T_i, so T_0 is the identity again.
    for refinement in range(REFINEMENT_PASSES):
        for i in range(len(clouds)):
            others = merge([c for k, c in enumerate(aligned_icp) if k != i])
            transforms[i], icp_result = refine_icp(icp_clouds[i], others, transforms[i])
            aligned_icp[i] = transformed(icp_clouds[i], transforms[i])
        T0_inv = np.linalg.inv(transforms[0])
        transforms = [T0_inv @ T for T in transforms]
        aligned_icp = [transformed(c, T) for c, T in zip(icp_clouds, transforms)]
        if verbose:
            print(f"  refinement pass {refinement + 1} done")

    return transforms


def aggregate(clouds, transforms):
    """Part 1 step 4: apply the transformations to the full-resolution clouds and merge them."""
    return merge([transformed(c, T) for c, T in zip(clouds, transforms)])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def transformed(cloud, T):
    """Copy of the cloud with T applied (transform() changes the cloud in place)."""
    copy = o3d.geometry.PointCloud(cloud)
    copy.transform(T)
    return copy


def merge(clouds):
    merged = o3d.geometry.PointCloud()
    for c in clouds:
        merged += c
    return merged
