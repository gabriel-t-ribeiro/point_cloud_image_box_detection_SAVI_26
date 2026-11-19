# SAVIBoxes – Trabalho 1 (SAVI 2026/27)

Estimates the pose and the dimensions of boxes on a table from several RGB-D point clouds.
The assignment is in [`SAVI_26-27/Trabalho1/readme.md`](SAVI_26-27/Trabalho1/readme.md).

## Repository layout

```
AGENTS.md            rules for AI assistants working on this repo
SAVI_26-27/          professor's course repository (git submodule) – do not edit
Trabalho1/           our code
    main.py              entry point: ./main.py --scene <folder> --num_boxes N
    registration.py      Part 1 – align the cameras (FPFH + RANSAC + ICP)
    registration_custom.py   Dinis's hand-written RANSAC, for comparison
    segmentation.py      Part 2 – table plane, table frame, objects on the table
    bounding_box.py      Part 3 – base method: z-up bounding box per object
    evaluation.py        errors against ground_truth.json
    requirements.txt
    scenes/              scenes generated with the simulator (not committed, see step 3)
```

## Tutorial: running the code

### 1. Get the code (with the submodule)

```bash
git clone --recurse-submodules https://github.com/gabriel-t-ribeiro/point_cloud_image_box_detection_SAVI_26.git
cd point_cloud_image_box_detection_SAVI_26
```

Already cloned without `--recurse-submodules`, or after a `git pull`:

```bash
git submodule update --init
```

### 2. Install the Python packages

```bash
pip install -r Trabalho1/requirements.txt
```

(Python 3.10+; a virtual environment is recommended: `python3 -m venv .venv && source .venv/bin/activate`.)

### 3. Generate test scenes with the professor's simulator

The scenes are about 40 MB each, so they are not in git. Generate them into `Trabalho1/scenes`:

```bash
cd SAVI_26-27/Trabalho1/simulator
./savi_box_simulator.py --batch configs/level_1.json configs/level_2.json --output ../../../Trabalho1/scenes
cd ../../../Trabalho1
```

- `configs/` has one example per level (`level_1`, `level_2`, `level_3_*`, `level_4`); `--batch configs/level_*.json` makes all of them.
- `--no_noise` gives perfect sensor data (useful to check that a method is right before tuning it for noise).
- Without `--batch` the simulator opens a window where you can move boxes and cameras and **Export scene**.

Each scene folder has `point_clouds/000.ply ...`, `rgb/`, `intrinsics.json`, `config.json` and `ground_truth.json`.

### 4. Run

From inside `Trabalho1/`:

```bash
./main.py --scene scenes/level_1 --num_boxes 1
./main.py --scene scenes/level_2 --num_boxes 3 --show
```

| Option | Meaning |
|---|---|
| `--scene` | scene folder (required) |
| `--num_boxes` | number of boxes in the scene (required) |
| `--show` | open the 3D windows of each step (close each window to continue) |
| `--registration custom` | use Dinis's RANSAC instead of Open3D's (for comparison) |
| `--output` | where to write `result.json` (default: inside the scene folder) |

`./main.py --help` shows the same.

### 5. Read the output

The program prints, for each part:

- **Part 1 – registration**: RANSAC/ICP fitness per camera and, if the scene has `ground_truth.json`, the translation (mm) and rotation (deg) error of each camera.
- **Part 2 – segmentation**: the table found, the objects found, and their precision / recall (measured on the cloud aggregated with the *true* transforms, so registration errors do not count).
- **Part 3 – boxes**: each box (`x, y, z, yaw, length x width x height` in the table frame) and, with ground truth, the errors and the 3D IoU (the metric of the presentation day).

It writes `result.json` in the format of the assignment (`cameras` with `T_camera0_camera`, `boxes` with `T_camera0_box`, `length`, `width`, `height`).

A full run takes about 3 minutes per scene (most of it is registration).

### 6. Change parameters

The parameters are at the top of each module, with a comment saying what they do:

- `registration.py`: `FEATURE_VOXEL`, `ICP_VOXEL`, `ICP_THRESHOLDS` (one value per ICP pass), `RANSAC_ATTEMPTS`, `REFINEMENT_PASSES`, and the noise filters `MAX_DEPTH`, `OUTLIER_NEIGHBOURS`, `ICP_ROBUST_KERNEL` (off by default).
- `segmentation.py`: plane and clustering thresholds (`PLANE_DISTANCE`, `MIN_TABLE_AREA`, `OBJECT_CLUSTER_EPS`, ...).
- `bounding_box.py`: outlier removal and `PERCENTILE`.

## Current results (simulator, default noise)

| | Level 1 (1 box) | Level 2 (3 boxes) |
|---|---|---|
| Registration error (mean) | 5.9 mm, 0.27° | 6.6 mm, 0.37° |
| Segmentation | 1/1 found, precision 0.998, recall 0.94 | 3/3 found, precision 0.99, recall 0.92–0.96 |
| Bounding box 3D IoU | 0.903 | 0.879 |

Still to do: the main Part 3 method (fitting with `scipy.optimize.least_squares`), levels 3 and 4, and the robustness items in the GitHub issues.

## Getting the professor's updates

1. On GitHub, open our fork `gabriel-t-ribeiro/SAVI_26-27` and click **Sync fork**.
2. In this repo:

```bash
git submodule update --remote SAVI_26-27
git commit -am "Update professor's material"
git push
```

Never commit changes inside `SAVI_26-27/`; copy a file into `Trabalho1/` if you want to change it.
