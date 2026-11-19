#!/usr/bin/env python3
"""SAVIBoxes - Trabalho 1.

Usage (from inside gw/):
    ./main.py --scene scenes/level_1 --num_boxes 1
    ./main.py --scene scenes/level_2 --num_boxes 3 --show            # open the 3D windows
    ./main.py --scene scenes/level_4 --num_boxes 3 --show_optimization   # only the boxes moving in the fit
    ./main.py --scene scenes/level_1 --num_boxes 1 --bbox open3d     # base method as in the assignment (Open3D OBB)
    ./main.py --scene scenes/level_1 --num_boxes 1 --registration custom   # Dinis's RANSAC

Steps:
    Part 1  registration.py   align all cameras with camera 000, aggregate the clouds
    Extra   photometry.py     gain and bias of every image vs image 000, colours corrected
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
import photometry
import registration
import segmentation

AGGREGATE_VOXEL = 0.002   # the aggregated cloud has ~2.4M points; 2 mm keeps the detail and is much faster


def main():
    # description=__doc__ makes --help also show the examples and steps above
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="scene folder (point_clouds/, intrinsics.json, ...)")
    parser.add_argument("--num_boxes", type=int, required=True, help="number of boxes in the scene")
    parser.add_argument("--no_photometric", action="store_true",
                        help="skip the photometric correction (gain and bias of each image vs image 000)")
    parser.add_argument("--registration", choices=["open3d", "custom"], default="open3d",
                        help="RANSAC used for the initial estimate (custom = Dinis's version)")
    parser.add_argument("--output", default=None, help="result.json path (default: <scene>/result.json)")
    parser.add_argument("--bbox", choices=["planes", "open3d", "open3d_minimal"], default="planes",
                        help="base method shown in Part 3: planes = ours (default); open3d / open3d_minimal = "
                             "the assignment's get_oriented_bounding_box() / get_minimal_oriented_bounding_box(). "
                             "The optimization always starts from 'planes'.")
    parser.add_argument("--show", action="store_true", help="show the intermediate results in 3D windows")
    parser.add_argument("--show_optimization", action="store_true",
                        help="only show the boxes moving during the optimization (also on with --show)")
    args = parser.parse_args()

    o3d.utility.random.seed(0)
    ground_truth = evaluation.load_ground_truth(args.scene)
    output = args.output or os.path.join(args.scene, "result.json")
    start = time.time()

    # -------------------------------------------------------------------------
    # Part 1 - Registration
    # -------------------------------------------------------------------------
    print("Part 1 - registration")
    names, clouds = registration.load_scene_clouds(args.scene)
    transforms = registration.register_scene(clouds, names, method=args.registration)

    # Extra - photometric correction: every image to the colours of image 000 (needs the poses, so
    # after the registration; before the aggregation, so Parts 2 and 3 use the corrected colours).
    # Harmless without exposure changes: it then estimates s ~ 1, b ~ 0.
    if not args.no_photometric:
        exposure_s, exposure_b = photometry.estimate_exposure(clouds, transforms)
        clouds = photometry.correct_colours(clouds, exposure_s, exposure_b)
        evaluation.print_photometry(names, exposure_s, exposure_b, ground_truth)

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
    # boxes = our plane-based box: the initial estimate of the optimization, whatever --bbox is.
    # baseline = the bounding box shown as the base method (--bbox: ours, or Open3D's as in the assignment).
    print(f"\nPart 3 - boxes (bounding box: {args.bbox})")
    obj_clouds = bounding_box.object_clouds(aggregated, seg)
    boxes = [bounding_box.bounding_box(c, seg.T_camera0_table) for c in obj_clouds]
    if args.bbox == "planes":
        baseline = boxes
    else:
        baseline = [bounding_box.open3d_bounding_box(c, seg.T_camera0_table, minimal=args.bbox == "open3d_minimal")
                    for c in obj_clouds]
    for box in baseline:
        print(f"  {box}")
    bbox_result_boxes = bounding_box.boxes_to_result(baseline, seg.T_camera0_table)

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
    viewer = None
    if (args.show or args.show_optimization) and len(all_points) > 0:
        viewer = OptimizationViewer(all_points, all_colours, seg.T_camera0_table, ground_truth,
                                    plot_path=os.path.join(os.path.dirname(os.path.abspath(output)),
                                                           "optimization_cost.png"))
    opt_boxes = optimization.optimize_boxes(all_points, boxes, seg.T_camera0_table, num_boxes=args.num_boxes,
                                            association="colour", point_colours=all_colours, on_iteration=viewer)
    if viewer:
        viewer.close()
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


class OptimizationViewer:
    """Part 3, item 6: the boxes moving during the optimization, over the point cloud.

    optimize_boxes() calls the viewer after every least_squares iteration (callback) with the
    boxes in the table frame. A non-blocking Open3D window (Visualizer + poll_events +
    update_renderer) is redrawn each time: green = current boxes, red = true boxes.
    """

    def __init__(self, points, colours, T_camera0_table, ground_truth=None, pause=0.02, plot_path=None):
        self.T_camera0_table = T_camera0_table
        self.pause = pause                     # seconds per frame, so the motion can be followed
        self.plot_path = plot_path             # where close() saves the cost plot (None = only show it)
        self.history = []                      # (frame, cost, stage) for the cost plot
        self.iterations = 0
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Part 3 - optimization (green = current, red = true)")
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.colors = o3d.utility.Vector3dVector(colours)
        # Display only: sparser, smaller points, so the box edges (lines on the surface) show through
        self.vis.add_geometry(cloud.voxel_down_sample(0.005))
        self.vis.get_render_option().point_size = 1.0
        if ground_truth:
            for b in ground_truth["boxes"]:
                self.vis.add_geometry(box_lines(b["T_camera0_box"], b["length"], b["width"], b["height"], [1, 0, 0]))
        # Same viewing direction as the other windows (VIEW), but centred on the objects: VIEW's
        # lookat is the middle of the room, and the zoom is relative to what is in this window.
        view = self.vis.get_view_control()
        view.set_front(VIEW["front"]), view.set_up(VIEW["up"])
        view.set_lookat(np.asarray(points).mean(axis=0)), view.set_zoom(0.8)
        self.lines = []                        # one LineSet per current box

    def __call__(self, boxes_params, cost, stage):
        boxes = [bounding_box.Box(*p) for p in boxes_params]
        new = [box_lines(b["T_camera0_box"], b["length"], b["width"], b["height"], [0, 0.8, 0])
               for b in bounding_box.boxes_to_result(boxes, self.T_camera0_table)]
        if len(new) != len(self.lines):        # the split search adds boxes: replace all the LineSets
            for ls in self.lines:
                self.vis.remove_geometry(ls, reset_bounding_box=False)
            self.lines = new
            for ls in self.lines:
                self.vis.add_geometry(ls, reset_bounding_box=False)
        else:                                  # same boxes: move the existing LineSets
            for ls, n in zip(self.lines, new):
                ls.points = n.points
                self.vis.update_geometry(ls)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.iterations += 1
        self.history.append((self.iterations, cost, stage.replace(" (start)", "")))
        print(f"\r  {stage:45s} cost {cost:.6f}  (frame {self.iterations})", end="", flush=True)
        time.sleep(self.pause)

    def close(self):
        """Keeps the final boxes on screen until the window is closed, then plots the cost."""
        print("\n  optimization finished: close the window to see the cost plot")
        self.vis.run()
        self.vis.destroy_window()
        self.plot_cost()

    def plot_cost(self):
        """Cost (least_squares' robust cost, 0.5 * sum rho(r^2)) against the iteration, one line per fit.

        The split-search fits use a subsample of the points (optimization.SEARCH_MAX_POINTS) and the
        final fit uses all of them, so their cost levels are not comparable: look at each line's shape.
        """
        import matplotlib.pyplot as plt

        if not self.history:
            return
        fits = []                                   # consecutive frames of the same stage = one fit
        for frame, cost, stage in self.history:
            if not fits or fits[-1][0] != stage:
                fits.append((stage, [], []))
            fits[-1][1].append(frame)
            fits[-1][2].append(cost)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        labelled = set()
        for stage, frames, costs in fits:
            final = stage.startswith("final")
            label = ("final fit (all points)" if final else
                     f"split search fits ({optimization.SEARCH_MAX_POINTS} points)")
            ax.plot(frames, costs, "-o", ms=2.5, lw=2 if final else 1,
                    color="tab:blue" if final else "0.6", label=None if label in labelled else label)
            labelled.add(label)
            ax.axvline(frames[0], color="0.85", lw=0.8, zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("iteration (all fits, in order)")
        ax.set_ylabel("cost  0.5 · Σ ρ(r²)  (soft_l1)")
        ax.set_title(f"Box fitting: cost per iteration ({len(fits)} fit{'s' if len(fits) > 1 else ''})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        if self.plot_path:
            fig.savefig(self.plot_path, dpi=150)
            print(f"  saved {self.plot_path}")
        plt.show()


if __name__ == "__main__":
    main()
