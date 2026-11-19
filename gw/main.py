#!/usr/bin/env python3
"""SAVIBoxes - Trabalho 1.

Usage (from inside gw/):
    ./main.py --scene scenes/level_1 --num_boxes 1
    ./main.py --scene scenes/level_2 --num_boxes 3 --show            # open the 3D windows
    ./main.py --scene scenes/level_1 --num_boxes 1 --registration custom   # Dinis's RANSAC

Steps:
    Part 1  registration.py   align all cameras with camera 000, aggregate the clouds
    Part 2  segmentation.py   table plane, table frame, objects above the table (split by colour if needed)
    Part 3  bounding_box.py   base method: z-up bounding box per object
            optimization.py   main method: nonlinear least-squares box fitting (geometry + colour)
    evaluation.py compares with ground_truth.json when the scene has it.
Writes result.json (default: inside the scene folder).
See the README.md at the top of the repository for the full tutorial.
"""

import argparse
import json
import os
import time

import numpy as np
import open3d as o3d

import bounding_box
import evaluation
import optimization
import registration
import segmentation

AGGREGATE_VOXEL = 0.002   # the aggregated cloud has ~2.4M points; 2 mm keeps the detail and is much faster


def main():
    # description=__doc__ makes --help also show the examples and steps above
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="scene folder (point_clouds/, intrinsics.json, ...)")
    parser.add_argument("--num_boxes", type=int, required=True, help="number of boxes in the scene")
    parser.add_argument("--registration", choices=["open3d", "custom"], default="open3d",
                        help="RANSAC used for the initial estimate (custom = Dinis's version)")
    parser.add_argument("--output", default=None, help="result.json path (default: <scene>/result.json)")
    parser.add_argument("--show", action="store_true", help="show the intermediate results in 3D windows")
    args = parser.parse_args()

    o3d.utility.random.seed(0)
    ground_truth = evaluation.load_ground_truth(args.scene)
    start = time.time()

    # -------------------------------------------------------------------------
    # Part 1 - Registration
    # -------------------------------------------------------------------------
    print("Part 1 - registration")
    names, clouds = registration.load_scene_clouds(args.scene)
    transforms = registration.register_scene(clouds, names, method=args.registration)
    aggregated = registration.aggregate(clouds, transforms).voxel_down_sample(AGGREGATE_VOXEL)
    print(f"  aggregated cloud: {len(aggregated.points)} points ({time.time() - start:.0f} s)")

    if ground_truth:
        evaluation.print_registration_errors(names, transforms, ground_truth)
    if args.show:
        show_registration(clouds, transforms, names, ground_truth)

    # -------------------------------------------------------------------------
    # Part 2 - Segmentation
    # -------------------------------------------------------------------------
    print("\nPart 2 - segmentation")
    seg = segmentation.segment_objects(aggregated)
    if len(seg.object_indices) > args.num_boxes:
        print(f"  keeping the {args.num_boxes} biggest of {len(seg.object_indices)} objects")
        seg.object_indices = seg.object_indices[:args.num_boxes]
    elif len(seg.object_indices) < args.num_boxes:
        print(f"  WARNING: found {len(seg.object_indices)} objects but there are {args.num_boxes} boxes "
              f"(boxes touching each other form one cluster: level 3)")

    if ground_truth:
        # Evaluate only the segmentation: same steps on the cloud aggregated with the TRUE transforms
        true_transforms = evaluation.true_camera_transforms(ground_truth, names)
        true_aggregated = registration.aggregate(clouds, true_transforms).voxel_down_sample(AGGREGATE_VOXEL)
        true_seg = segmentation.segment_objects(true_aggregated, verbose=False)
        evaluation.print_segmentation_errors(np.asarray(true_aggregated.points), true_seg.object_indices,
                                             ground_truth)
    if args.show:
        show_segmentation(aggregated, seg)

    # -------------------------------------------------------------------------
    # Part 3 - Boxes (base method: bounding box)
    # -------------------------------------------------------------------------
    print("\nPart 3 - boxes (bounding box)")
    obj_clouds = bounding_box.object_clouds(aggregated, seg)
    boxes = [bounding_box.bounding_box(c, seg.T_camera0_table) for c in obj_clouds]
    for box in boxes:
        print(f"  {box}")
    bbox_result_boxes = bounding_box.boxes_to_result(boxes, seg.T_camera0_table)

    if ground_truth:
        evaluation.print_box_errors(bbox_result_boxes, ground_truth, seg.T_camera0_table, "Bounding Box Errors")

    # -------------------------------------------------------------------------
    # Part 3 - Boxes (main method: nonlinear least-squares optimization)
    # -------------------------------------------------------------------------
    # Main result: colour in the objective (level 4, approach 3). Each point goes to the box that best
    # explains its position AND its colour; with a single colour it is the same as the geometric fit.
    # The points with no clear colour (seams between boxes) are used too: geometry still places them.
    print("\nPart 3 - boxes (optimization, geometry + colour)")
    obj_points = np.vstack([np.asarray(c.points) for c in obj_clouds]) if obj_clouds else np.empty((0, 3))
    obj_colours = np.vstack([np.asarray(c.colors) for c in obj_clouds]) if obj_clouds else np.empty((0, 3))
    all_points = np.vstack([obj_points, np.asarray(aggregated.points)[seg.no_colour_indices]])
    all_colours = np.vstack([obj_colours, np.asarray(aggregated.colors)[seg.no_colour_indices]])
    opt_boxes = optimization.optimize_boxes(all_points, boxes, seg.T_camera0_table, num_boxes=args.num_boxes,
                                            association="colour", point_colours=all_colours)
    for box in opt_boxes:
        print(f"  {box}")
    result_boxes = bounding_box.boxes_to_result(opt_boxes, seg.T_camera0_table)

    if ground_truth:
        evaluation.print_box_errors(result_boxes, ground_truth, seg.T_camera0_table, "Optimization Errors")

    # Level 4 comparison (only when some cluster was split by colour):
    #   approach 1 = the bounding box above (one per colour group)
    #   approach 2 = separate by colour, then fit: each point stays with its colour group's box
    #   geometric  = same fit without colour (each point goes to the closest box)
    if ground_truth and seg.colour_split:
        print("\nPart 3 - level 4 comparison")
        if len(obj_clouds) == args.num_boxes:
            assignment = np.concatenate([np.full(len(c.points), j) for j, c in enumerate(obj_clouds)])
            fixed_boxes = optimization.optimize_boxes(obj_points, boxes, seg.T_camera0_table, num_boxes=args.num_boxes,
                                                      association="fixed", assignment=assignment)
            evaluation.print_box_errors(bounding_box.boxes_to_result(fixed_boxes, seg.T_camera0_table), ground_truth,
                                        seg.T_camera0_table, "Separate by colour, then fit (fixed association)")
        geo_boxes = optimization.optimize_boxes(all_points, boxes, seg.T_camera0_table, num_boxes=args.num_boxes)
        evaluation.print_box_errors(bounding_box.boxes_to_result(geo_boxes, seg.T_camera0_table), ground_truth,
                                    seg.T_camera0_table, "Geometry only (no colour in the objective)")
    if args.show:
        show_boxes(aggregated, result_boxes, ground_truth)

    # -------------------------------------------------------------------------
    # result.json
    # -------------------------------------------------------------------------
    result = {
        "cameras": [{"image": n, "T_camera0_camera": T.tolist()} for n, T in zip(names, transforms)],
        "boxes": result_boxes,
    }
    output = args.output or os.path.join(args.scene, "result.json")
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {output} ({time.time() - start:.0f} s in total)")


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
CAMERA_COLORS = [[0.9, 0.3, 0.3], [0.3, 0.8, 0.3], [0.3, 0.4, 0.9], [0.9, 0.8, 0.2],
                 [0.8, 0.3, 0.8], [0.2, 0.8, 0.8], [0.9, 0.5, 0.1], [0.5, 0.5, 0.5]]

# Initial viewpoint of every 3D window (copied with Ctrl+C in an Open3D window, camera 0 frame).
# The camera frame has y pointing down, so "up" is ~ -y; otherwise the scene appears upside down.
VIEW = {
    "front": [-0.38918421172498591, -0.27430820117149579, -0.87936946735377364],
    "lookat": [-0.21026629438816719, 0.58156416183949011, 1.0073384614691785],
    "up": [0.30026997198607369, -0.940268749125868, 0.16041391255361312],
    "zoom": 0.7,
}


def show_registration(clouds, transforms, names, ground_truth):
    """Aggregated cloud with one colour per camera; then overlaid with the true aggregation."""
    coloured = []
    for k, (cloud, T) in enumerate(zip(clouds, transforms)):
        c = registration.transformed(cloud.voxel_down_sample(0.01), T)
        c.paint_uniform_color(CAMERA_COLORS[k % len(CAMERA_COLORS)])
        coloured.append(c)
    o3d.visualization.draw_geometries(coloured, window_name="Part 1 - one colour per camera", **VIEW)

    if ground_truth:
        true_transforms = evaluation.true_camera_transforms(ground_truth, names)
        estimated = registration.aggregate([c.voxel_down_sample(0.01) for c in clouds], transforms)
        true = registration.aggregate([c.voxel_down_sample(0.01) for c in clouds], true_transforms)
        estimated.paint_uniform_color([1, 0.7, 0])
        true.paint_uniform_color([0, 0.65, 0.93])
        o3d.visualization.draw_geometries([estimated, true],
                                          window_name="Part 1 - estimated (orange) vs true (blue)", **VIEW)


def show_segmentation(aggregated, seg):
    background = aggregated.voxel_down_sample(0.01)
    background.paint_uniform_color([0.7, 0.7, 0.7])
    objects = []
    for k, idx in enumerate(seg.object_indices):
        o = aggregated.select_by_index(idx)
        o.paint_uniform_color(CAMERA_COLORS[k % len(CAMERA_COLORS)])
        objects.append(o)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    frame.transform(seg.T_camera0_table)
    o3d.visualization.draw_geometries([background, frame] + objects,
                                      window_name="Part 2 - objects on the table (table frame shown)", **VIEW)


def box_lines(T_camera0_box, length, width, height, color):
    T = np.array(T_camera0_box)
    obb = o3d.geometry.OrientedBoundingBox(T[:3, 3], T[:3, :3], [length, width, height])
    lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    lines.paint_uniform_color(color)
    return lines


def show_boxes(aggregated, result_boxes, ground_truth):
    geometries = [aggregated]
    geometries += [box_lines(b["T_camera0_box"], b["length"], b["width"], b["height"], [0, 0.8, 0])
                   for b in result_boxes]
    if ground_truth:
        geometries += [box_lines(b["T_camera0_box"], b["length"], b["width"], b["height"], [1, 0, 0])
                       for b in ground_truth["boxes"]]
    o3d.visualization.draw_geometries(geometries, window_name="Part 3 - estimated (green) vs true (red)", **VIEW)


if __name__ == "__main__":
    main()
