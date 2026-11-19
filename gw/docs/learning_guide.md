# SAVI - Trabalho Prático 1 (SAVIBoxes): Master's Engineering & Visual Socratic Guide

This is your personal study guide for **Trabalho Prático 1 (SAVIBoxes)**. Every math block is paired with **plain-English intuition**, **concrete numbers**, **ASCII visual diagrams**, and **Socratic checkpoints** so you never have to decode dense matrix notation cold.

---

## 0. The Big Picture & Core Foundations

### 0.1 Why This Pipeline Exists
A single RGB-D camera looking at boxes on a table only sees the faces pointing toward it (**self-occlusion**). To measure every box to **millimeter accuracy**, we chain three stages:

```
[Multi-View RGB-D Clouds: 000.ply ... 007.ply]
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ Part 1: Multi-View Registration                        │
│  Filter & Downsample ──► FPFH + RANSAC (Coarse)        │
│                     ──► Full-Cloud Point-to-Plane ICP  │
│                     ──► Glue all views into camera0    │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ Part 2: Tabletop Segmentation                          │
│  RANSAC Plane Fit ──► Build flat "table" frame (z = 0) │
│  Crop Above Table ──► Group separated islands (DBSCAN) │
│                   ──► Separate by color if needed (Lv4)│
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ Part 3: Box Pose & Dimension Estimation                │
│  Baseline (3A): Bounding box / 5-plane (initial guess) │
│  Main (3B):     least_squares surface fitting where    │
│                 each point picks its closest box live  │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ Evaluation: 3D IoU + Position / Height / Yaw / L,W,H   │
└────────────────────────────────────────────────────────┘
```

---

### 0.2 Coordinate Frames: Moving Between Views Without Matrix Jargon

Imagine holding a ruler in three different rooms (`camera0`, `table`, `box`). To compare numbers, you must convert coordinates from one ruler into another. That is all a $4 \times 4$ transform matrix $\mathbf{T}_{A, B}$ does:

```
   Point in frame B: p_B          Transform T_A_B          Point in frame A: p_A
  ┌─────────────────────┐      ┌───────────────────┐      ┌─────────────────────┐
  │ "30 cm right of     │ ───► │ 1. Rotate axes    │ ───► │ "Where that spot is │
  │  camera 1's lens"   │      │ 2. Shift origin   │      │  seen from camera0" │
  └─────────────────────┘      └───────────────────┘      └─────────────────────┘
```

If you write it out as a $4 \times 4$ grid (appending a dummy `1` at the bottom so rotation and shift fit in one multiplication):

$$\begin{bmatrix} x_A \\ y_A \\ z_A \\ 1 \end{bmatrix} = \underbrace{\begin{bmatrix} \mathbf{R}_{A, B} & \mathbf{t}_{A, B} \\ 0\;\; 0\;\; 0 & 1 \end{bmatrix}}_{\mathbf{T}_{A, B}} \begin{bmatrix} x_B \\ y_B \\ z_B \\ 1 \end{bmatrix} \quad \Longleftrightarrow \quad \mathbf{p}_A = \mathbf{R}_{A, B}\,\mathbf{p}_B + \mathbf{t}_{A, B}$$

- **What are the 3 columns of $\mathbf{R}_{A,B}$?** Simply the $x$, $y$, and $z$ arrow directions of frame $B$, written as vectors in frame $A$.
- **What is $\mathbf{t}_{A,B}$?** The $(x, y, z)$ coordinates of frame $B$'s `(0, 0, 0)` origin, written in frame $A$.
- **Chaining & Reversing:**
  - Chaining: to go from `box` $\to$ `table` $\to$ `camera0`, multiply $\mathbf{T}_{\text{camera0}, \text{table}} \cdot \mathbf{T}_{\text{table}, \text{box}}$.
  - Reversing ($\mathbf{T}_{A, B}^{-1}$): undo the shift and rotate backward ($\mathbf{R}^T$ is the exact opposite rotation of $\mathbf{R}$!).

**The 4 Frames in SAVIBoxes:**
```
camera0:   x right, y down, z forward (origin at camera 000 lens)
table:     x, y lie flat on tabletop, z points UP (z = 0 is the table surface!)
box_b:     origin at box CENTER (x along length L, y along width W, z UP)
           Note: box parameter "z" is the BASE height on the table,
                 so the box CENTER height is (z + height / 2)!
```

> **Socratic Check:** Why is it so much easier to fit boxes in the `table` frame instead of `camera0`?
> *Think before reading:* In `camera0`, the table is tilted at some arbitrary 3D angle, so a box resting on it has 6 pose degrees of freedom (`x, y, z, roll, pitch, yaw`). Once you move points into the `table` frame, the table is flat at $z = 0$ and boxes cannot tilt in roll or pitch! Every box reduces to just **4 pose numbers** (`x, y, z, yaw`) + **3 sizes** (`L, W, H`).

---

### 0.3 Measuring Errors & Box Symmetries

#### 1. Camera Alignment Error
If your estimate is $\mathbf{T}_{\text{est}}$ and the truth is $\mathbf{T}_{\text{gt}}$:
- **Translation error ($e_t$):** Straight-line 3D distance between the estimated camera position and true camera position, in millimeters:
  $$e_t = \sqrt{(t_x - t_{\text{gt},x})^2 + (t_y - t_{\text{gt},y})^2 + (t_z - t_{\text{gt},z})^2} \times 1000\;\text{[mm]}$$
- **Rotation error ($e_r$):** How many degrees you would have to twist the estimated camera to match the true camera orientation. Any 3D rotation difference $\mathbf{R}_E = \mathbf{R}_{\text{est}}^T \mathbf{R}_{\text{gt}}$ equals one single twist of angle $\theta$ around some axis, and the sum of its diagonal entries ($\operatorname{tr}(\mathbf{R}_E) = r_{11}+r_{22}+r_{33}$) always equals $1 + 2\cos\theta$:
  $$\theta = \arccos\!\left(\operatorname{clip}\!\left(\frac{(r_{11} + r_{22} + r_{33}) - 1}{2},\; -1,\; 1\right)\right)$$
  *(We `clip` to `[-1, 1]` because floating-point rounding like `1.0000001` makes `arccos` crash with `NaN`.)*

#### 2. Box Symmetries & 3D IoU
Rotate a box $180^\circ$ around its vertical $z$-axis: it looks identical. Rotate it $90^\circ$ and swap `length` $\leftrightarrow$ `width`: **still the exact same occupied box!**
- **Yaw & Dimension Errors:** We test both `(no swap)` and `(swap L <-> W, add 90° to yaw)` with angle difference wrapped into $[-90^\circ, +90^\circ)$, and keep whichever has the smallest `abs(yaw_error)`.
- **3D IoU (Intersection over Union):** Measures overlap directly in 3D space (immune to `L/W` swaps!). Because boxes always stand vertically:

```
Top View (2D rotated rectangles):      Side View (height overlap):
   ┌──────────┐                             ┌──────┐ ─┐
   │  Est   ┌─┼─────┐                       │ Est  │  ├─ Δz_overlap
   │        │█│True │   Area A_∩            │ ┌────┼──┼─┐
   └────────┼─┘     │                       └─┼────┘  │ │ True
            └───────┘                         └───────┘ ┘
```
$$V_{\text{intersection}} = A_{\cap} \times \Delta z_{\text{overlap}}, \qquad \text{IoU}_{3\text{D}} = \frac{V_{\text{intersection}}}{V_{\text{est}} + V_{\text{true}} - V_{\text{intersection}}} \in [0, 1]$$

---

### 0.4 How `scipy.optimize.least_squares` Works (No Heavy Calculus Needed)

Think of `least_squares` as adjusting knobs (`params = [x, y, z, yaw, L, W, H]`) to make a list of errors (`residuals = [r_1, r_2, ..., r_N]`, one number per point) as close to zero as possible:

$$\text{Total Cost} = \frac{1}{2} \sum_{i=1}^{N} \rho\!\left(r_i(\mathbf{\theta})^2\right)$$

1. You give it a starting guess `x0` and a function `residuals(params)` that returns one error per point.
2. Behind the scenes, `least_squares` nudges each knob slightly (`x + 0.0001`, `yaw + 0.0001`, ...) to see how every point's error responds (that table of sensitivities is the **Jacobian** $\mathbf{J}$).
3. It takes a downhill step on all knobs at once and repeats until the cost stops dropping.
4. **Why `loss='soft_l1'`?** With ordinary squared error ($r^2$), one single noisy outlier point $10\text{ cm}$ away has $100\times$ the pull of a $1\text{ cm}$ point (`10² = 100` vs `1² = 1`). `soft_l1` caps outlier growth from quadratic ($r^2$) to linear ($|r|$) once $|r| > \text{f\_scale}$, so noisy points can't drag the box away from the clean faces.

---

## 1. Part 1: Multi-View Point Cloud Registration

### 1.1 Physical & Engineering Intuition
Each camera (`000`..`007`) measures points in its *own* local coordinate system.
- **Why ICP alone fails:** ICP pulls every point toward whichever point happens to be closest *right now*. If camera `001` starts $45^\circ$ away from camera `000`, nearest neighbors match the wrong wall to the wrong table leg and lock into a bad **local minimum**.
- **Two-step solution (Coarse $\to$ Fine):**
  1. **Coarse (FPFH + RANSAC):** Describes the local 3D shape around each point (corners, edges, flat planes) regardless of how the camera is rotated, then uses RANSAC to test random 3-point matches until one rigid transform aligns most of the scene.
  2. **Fine (Point-to-Plane ICP):** Starting from that rough alignment, slides surfaces cleanly along each other until millimeter alignment is reached.

### 1.2 Point-to-Point vs. Point-to-Plane ICP (Visualized)

```
Point-to-Point ICP:                     Point-to-Plane ICP:
      ● p' (moving cloud)                     ● p' (moving cloud)
     /                                        │
    /  pulls straight to nearest dot          │  pulls ONLY perpendicular to wall
   ▼   (resists sliding along flat walls!)    ▼  (lets flat surfaces slide freely!)
──●─────────────── Target Wall          ──────┴────────●────── Target Wall (normal ▲)
  q                                                    q
```

- **Point-to-Point:** Minimizes squared distance $\|( \mathbf{R}\mathbf{p}_i + \mathbf{t} ) - \mathbf{q}_i\|^2$. Because two RGB-D cameras never hit the exact same discrete dots on a flat wall, pulling dot-to-dot creates artificial "friction" along flat surfaces.
- **Point-to-Plane:** Projects the gap onto the target surface normal $\mathbf{n}_i$: $\left[ (\mathbf{R}\mathbf{p}_i + \mathbf{t} - \mathbf{q}_i) \cdot \mathbf{n}_i \right]^2$. Flat planes slide smoothly along their tangents and snap tightly along their normals.

### 1.3 Answers to the Professor's Conceptual Questions

1. **What happens to ICP if you remove the background and leave only the table and boxes? Why?**
   - **Socratic Check:** Imagine looking only at a giant flat tabletop with one small rectangular box in the middle. If you slide the second camera's cloud $5\text{ cm}$ sideways along the table, do the tabletop points still lie right on the tabletop plane?
   - **Answer:** **Yes — and that's the trap!** A single flat tabletop constrains up/down tilt and height, but leaves **sliding sideways and spinning in yaw weakly constrained**. Removing the room walls, corners, and floor strips away the orthogonal 3D surfaces needed to lock all 6 degrees of freedom, so ICP drifts or misaligns.

2. **What happens to the error when chaining many images? Does the last image align with the first?**
   - **Answer:** If you align `001 -> 000`, then `002 -> 001`, ..., `007 -> 006`, every small $0.3^\circ$ error adds up step by step (**drift**). By the time you reach `007`, the accumulated error creates a **loop-closure seam** where `007` and `000` don't match (double edges/walls). Aligning each new view against the **accumulated cloud** (all previously aligned views together, anchored to `000`) and running refinement passes suppresses that drift.

3. **What effect do registration errors have on the aggregated point cloud?**
   - **Answer:** **Ghosting / doubled walls.** A $10\text{ mm}$ camera misalignment splits one crisp box wall into two parallel layers $10\text{ mm}$ apart, inflating downstream bounding box dimensions and reducing 3D IoU.

### 1.4 API Mechanics & Minimal Isolated Snippet
Reference: [Open3D Global Registration](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html) & [ICP Registration](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html).

```python
import numpy as np
import open3d as o3d

# Isolated Demonstration: Coarse FPFH+RANSAC followed by Point-to-Plane ICP
src = o3d.geometry.PointCloud()
src.points = o3d.utility.Vector3dVector(np.random.rand(500, 3))
tgt = o3d.geometry.PointCloud(src)
tgt.translate([0.15, -0.05, 0.02])

for pcd in (src, tgt):
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    src, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
)
tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
)

coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    src, tgt, src_fpfh, tgt_fpfh, mutual_filter=True, max_correspondence_distance=0.05,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=3,
    checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
              o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.99),
)

fine = o3d.pipelines.registration.registration_icp(
    src, tgt, max_correspondence_distance=0.02, init=coarse.transformation,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
)
print("Recovered translation:", np.round(fine.transformation[:3, 3], 3))
```

---

## 2. Part 2: Tabletop Segmentation & Coordinate Framing

### 2.1 Physical & Engineering Intuition
1. **Why sequential RANSAC for the table?** A big room wall can have *more* points than the table! So we find planes repeatedly and pick the horizontal candidate (normal aligned with camera "up") at table height.
2. **Building the `table` frame:** Think of placing 3 perpendicular ruler arrows on the table:
   - Arrow $\mathbf{z}_t$: the table's upward unit normal $\mathbf{n}$.
   - Arrow $\mathbf{x}_t$: any direction lying flat on the table (made strictly perpendicular to $\mathbf{z}_t$).
   - Arrow $\mathbf{y}_t$: $\mathbf{z}_t \times \mathbf{x}_t$ (automatically perpendicular to both!).
   - Origin $\mathbf{p}_0$: center of the table surface.
   Put $[\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t, \mathbf{p}_0]$ into a $4 \times 4$ matrix $\mathbf{T}_{\text{camera0}, \text{table}}$, and its inverse moves every point into clean table coordinates where **`z = 0` is the tabletop**.
3. **Cropping & DBSCAN:** Keep points with $z_{\text{min}} < z < z_{\text{max}}$ inside the table's $(x, y)$ boundary, then run DBSCAN (connects points closer than `eps`) to isolate separate objects.

### 2.2 Evaluating Segmentation Without Registration Errors
To grade Part 2 fairly, we re-aggregate the clouds using `ground_truth` camera poses, check which points fall inside each true box (with a $5\text{ mm}$ noise margin), and measure:

$$\text{Precision} = \frac{\text{true box points in your cluster}}{\text{all points in your cluster}}, \qquad \text{Recall} = \frac{\text{true box points in your cluster}}{\text{all points that actually belong to that box}}$$

### 2.3 API Mechanics & Minimal Isolated Snippet
Reference: [Open3D Plane Segmentation](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Plane-segmentation) & [DBSCAN Clustering](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#DBSCAN-clustering).

```python
import numpy as np
import open3d as o3d

rng = np.random.default_rng(0)
table = np.column_stack([rng.uniform(-0.5, 0.5, 400), rng.uniform(-0.5, 0.5, 400), np.zeros(400)])
box1  = rng.uniform([-0.2, -0.1, 0.02], [-0.05, 0.1, 0.15], size=(120, 3))
box2  = rng.uniform([0.1, -0.1, 0.02], [0.25, 0.1, 0.12], size=(120, 3))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.vstack([table, box1, box2]))

plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=500)
objects_pcd = pcd.select_by_index(inliers, invert=True)

labels = np.array(objects_pcd.cluster_dbscan(eps=0.04, min_points=20))
print(f"Plane: {np.round(plane_model, 3)} | Found {labels.max() + 1} object clusters")
```

---

## 3. Part 3A: Base Method — Bounding Boxes

### 3.1 Why Open3D's OBB Has Larger Errors (2 Reasons)
1. **Unconstrained 3D tilt:** `get_oriented_bounding_box()` and `get_minimal_oriented_bounding_box()` are allowed to tilt in **roll and pitch**. Noise or an unseen side wall tilts the whole box slightly off the table, corrupting `z`, `height`, and footprint.
2. **Convex-hull extreme-point sensitivity (Defense Q5):**
   Look at what `minAreaRect` or `get_minimal_oriented_bounding_box()` actually uses:

```
Clean wall (thousands of points):      One flying pixel sticking 2 cm out:
   ●●●●●●●●●●●●●│                         ●●●●●●●●●●●●●│      ● (outlier!)
   ●●●●●●●●●●●●●│ ◄── tight wall          ●●●●●●●●●●●●●│──────┴──► dragged 2 cm!
   ●●●●●●●●●●●●●│                         ●●●●●●●●●●●●●│
```
   Minimum enclosing boxes depend **exclusively on the few outermost hull points** and ignore the thousands of clean interior wall points. That is why Dinis's 5-plane median wall estimator (taking the **median** of wall-facing points instead of min/max) beats convex-hull bounding boxes—and why `least_squares` goes even further by fitting all boxes simultaneously.

### 3.2 Minimal Isolated Snippet
Reference: [OpenCV `minAreaRect`](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html).

```python
import cv2
import numpy as np

pts_xy = np.random.uniform([-0.15, -0.10], [0.15, 0.10], size=(200, 2)).astype(np.float32)
theta = np.radians(25.0)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)

(cx, cy), (length, width), angle_deg = cv2.minAreaRect(pts_xy @ R.T)
print(f"Center=({cx:.3f}, {cy:.3f}), Size=({length:.3f}, {width:.3f}), Angle={angle_deg:.1f} deg")
```

---

## 4. Part 3B: Main Method — Nonlinear Optimization Box Fitting

This is the **heart of Trabalho 1**.

### 4.1 The Idea in Plain English
Instead of trusting extreme points or hard-splitting clusters before fitting (which breaks as soon as boxes touch in Level 3), we put **all $K$ boxes** into one list of parameters (`7 * K` numbers) and let **every point in the scene** choose whichever box explains it best **on every single solver evaluation**.

### 4.2 Step-by-Step Distance Math (With Concrete Numbers!)

Each box $b$ has $7$ parameters in the table frame:
$$\mathbf{\theta}_b = [x_b,\; y_b,\; z_b,\; \psi_b,\; L_b,\; W_b,\; H_b]$$

#### Step 1: Straighten the box (shift & un-rotate)
To measure how far a point $\mathbf{p}$ is from a tilted box, imagine putting the box flat at `(0, 0, 0)` by:
1. Subtracting the box's 3D center $\mathbf{c}_b = [x_b,\; y_b,\; z_b + H_b/2]$
2. Rotating backward by $-\psi_b$ around the vertical $z$-axis:
   $$\mathbf{q} = \mathbf{R}_z(\psi_b)^T (\mathbf{p} - \mathbf{c}_b)$$
   *(Think of turning a tilted sheet of paper straight so its walls line up with the $x$ and $y$ rulers.)*

#### Step 2: Measure overshoot past each wall ($\mathbf{d}$)
Say the straightened box has half-sizes **$10\text{ cm}$ in $x$**, **$5\text{ cm}$ in $y$**, and **$4\text{ cm}$ in $z$**. For a point at $(q_x, q_y, q_z)$, we fold across center using $|q|$ and subtract the half-sizes:
$$d_x = |q_x| - 10, \qquad d_y = |q_y| - 5, \qquad d_z = |q_z| - 4$$

- **Positive $d$** ($+3$): **$3\text{ cm}$ outside** that wall.
- **Zero $d$** ($0$): **right on** that wall.
- **Negative $d$** ($-2$): **$2\text{ cm}$ inside** that wall.

```
                      |  dy = +4 (4 cm outside top edge)
                      v
        ┌───────────────────────────┐  <- y = +5 (wall)
        │             ● (inside)    │  <- dy = -2 (2 cm inside from top wall)
        │                           │
────────┼───────────────────────────┼───────────
        │         center (0,0)      │     ● (outside right wall: dx = +3, dy = -3)
        └───────────────────────────┘
        ^                           ^
     x = -10                     x = +10
```

#### Step 3: Distance to the box SKIN (surface), not its solid interior
$$D(\mathbf{p}_i, \mathbf{\theta}_b) = \underbrace{\|\max(\mathbf{d}, \mathbf{0})\|_2}_{\text{Part A: active if OUTSIDE}} \;-\; \underbrace{\min(\max(d_x, d_y, d_z),\; 0)}_{\text{Part B: active if INSIDE}}$$

Only **one** part is ever nonzero at a time:
- **Outside point**, say $\mathbf{d} = (+3,\; +4,\; -1)$ (past the corner in $x$ and $y$):
  - **Part B turns off:** $\max(+3, +4, -1) = +4 \implies \min(+4, 0) = 0$.
  - **Part A (Pythagoras on positive overshoots):** $\max(\mathbf{d}, \mathbf{0}) = (3, 4, 0) \implies \sqrt{3^2 + 4^2 + 0^2} = \mathbf{5\text{ cm}}$.
- **Inside point**, say $\mathbf{d} = (-8,\; -2,\; -3)$ ($8\text{ cm}$ from $x$-wall, $2\text{ cm}$ from $y$-wall, $3\text{ cm}$ from $z$-wall):
  - **Part A turns off:** all negative $\implies \max(\mathbf{d}, \mathbf{0}) = (0, 0, 0) \implies 0$.
  - **Part B (closest wall):** $\max(-8, -2, -3) = -2$ (least negative = nearest wall!). Subtracting it gives $0 - (-2) = \mathbf{+2\text{ cm}}$.
- **On a wall**, say $\mathbf{d} = (0, -3, -3)$: both Part A and Part B are $0 \implies D = \mathbf{0}$.

> **Socratic Check:** Why did we need Part B at all? What happens if you delete Part B and return `0` for every point inside the box?
> *Think first:* If inside points give `0` error, what stops the solver from setting $L = 10\text{ m}, W = 10\text{ m}, H = 10\text{ m}$? Nothing! Every point in the scene would be inside that giant box and report `0.000` error. Part B penalizes hollow space between an interior point and the wall so the walls shrink inward to hug the points.

#### Step 4: Dynamic Multi-Box Association
With $K$ boxes, every point picks its closest box **inside every objective call**:
$$r_i(\mathbf{\theta}) = \min_{b \in \{1, \dots, K\}} D(\mathbf{p}_i, \mathbf{\theta}_b)$$

---

### 4.3 Answers to the Professor's 5 Defense Questions (Visualized)

1. **Can a box grow without the error increasing? In what situations and why? What did you do to solve it?**
   - **Socratic Check:** Suppose a box sits flat on the table ($z = 0$ fixed — **not** floating!), and the camera only saw its **top** and **front** wall, so its **back** wall has **zero** points (maybe it's occluded or pressed against another box in Level 3). What happens if we stretch that back wall $15\text{ cm}$ backward?
   ```
     Top points still sit right on top face (distance = 0!)
          ●●●●●●●●●●●●
          ┌──────────┬ - - - - - - ┐
   Front ●│          │   (empty!)  │  ◄── Unseen back wall stretched +15 cm!
   wall  ●│          │  0 points   │      Zero points inside -> error stays 0!
          └──────────┴ - - - - - - ┘
   ═══════════════════════════════════ Table (z = 0)
   ```
   - **Answer:** **Yes — it does NOT need to be floating!** Any time **at least one face has zero observed points** (an occluded back wall, a seam wall touching another box, or the hidden bottom face if $z$ is allowed to go negative into the table), pushing that unseen face outward changes the distance of **zero** points, so total error stays identical! (And of course, if you used solid-box distance instead of surface distance, *every* face could grow freely.)
   - **How to fix it:**
     1. Measure distance to the **surface** (include Part B for inside points).
     2. Enforce physical bounds ($z_b \ge 0$, or $z_b = 0$ when unstacked; $L, W, H > 0$).
     3. Add a tiny **volume / size regularization** term (e.g., $w_{\text{size}} \cdot (L_b + W_b + H_b)$ or $w_{\text{vol}} \sqrt{L_b W_b H_b}$) so unseen walls gently pull inward until they touch the edge points instead of drifting outward.

2. **Can two boxes occupy the same space? What happens in optimization if this is allowed?**
   - **Answer:** **Yes.** Without protection, two boxes can **collapse onto the exact same physical box** (both get near-zero residual on that one big box while leaving a smaller box unmatched as "outliers" under `soft_l1`), or **interpenetrate** slightly across a touching seam in Level 3.
   - **How to fix it:** Start boxes in distinct regions (e.g., split cluster via K-means or greedy residual peeling) and add a **pairwise overlap penalty** residual whenever two boxes intersect in 3D.

3. **Why does this fitting avoid an outer loop alternating between associating points and optimizing?**
   - **Answer:** Because `min_b D(p_i, theta_b)` is recomputed **on every residual evaluation inside `least_squares`**. When `least_squares` nudges box A's edge during Jacobian estimation, boundary points smoothly switch ownership between box A and box B on the fly — preventing an outer loop from locking in wrong initial cluster splits.

4. **Which points determine each parameter? Does a point in the middle of the box top help estimate its length?**
   - **Answer:** **No!** A point in the middle of the top face is closest to the **top face** (`d_z` dominates), so stretching `length` slightly leaves its distance to the top face completely unchanged ($\frac{\partial r_i}{\partial L} = 0$).
     - **Top face points** control `z + height`.
     - **Left/right wall points** (perpendicular to local $x$) control `x` and `length`.
     - **Front/back wall points** (perpendicular to local $y$) control `y` and `width`.
     - **Wall points away from wall centers** (giving rotational leverage) control `yaw`.

5. **Why is the bounding box so sensitive to noise? Which points does it actually use?**
   - **Answer:** Covered in Section 3.1 — enclosing bounding boxes use only the few outermost convex-hull points, whereas optimization averages over every point on every face.

### 4.4 API Mechanics & Minimal Isolated Snippet
Reference: [SciPy `least_squares` Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

```python
import numpy as np
from scipy.optimize import least_squares

def box_surface_distances(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Unsigned distance from N points (table frame) to 1 box [x, y, z, yaw, L, W, H]."""
    x, y, z, yaw, L, W, H = params
    c = np.array([x, y, z + H / 2.0])
    ca, sa = np.cos(yaw), np.sin(yaw)
    R_T = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])
    q = (points - c) @ R_T.T
    d = np.abs(q) - np.array([L / 2.0, W / 2.0, H / 2.0])
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    return outside - inside

# Isolated Demonstration: Fit one synthetic noisy box
rng = np.random.default_rng(42)
top = np.column_stack([rng.uniform(-0.15, 0.15, 200), rng.uniform(-0.10, 0.10, 200), np.full(200, 0.15)])
wall_x = np.column_stack([np.full(150, 0.15), rng.uniform(-0.10, 0.10, 150), rng.uniform(0.0, 0.15, 150)])
wall_y = np.column_stack([rng.uniform(-0.15, 0.15, 150), np.full(150, 0.10), rng.uniform(0.0, 0.15, 150)])
pts = np.vstack([top, wall_x, wall_y]) + rng.normal(0, 0.002, (500, 3))

x0 = np.array([0.02, -0.02, 0.0, 0.1, 0.25, 0.25, 0.12])
lb = [-1.0, -1.0, 0.0, -np.pi, 0.02, 0.02, 0.02]
ub = [ 1.0,  1.0, 1.0,  np.pi, 1.00, 1.00, 1.00]
res = least_squares(lambda th: box_surface_distances(pts, th), x0, bounds=(lb, ub), loss="soft_l1", f_scale=0.01)
print("Fitted [x, y, z, yaw, L, W, H]:", np.round(res.x, 4))
```

---

## 5. Difficulty Levels 1–4 & Identifiability Limits

### 5.1 Levels 1 & 2 (Single & Separated Boxes)
- **Level 1:** At zero noise, bounding box $\approx$ optimization. With realistic sensor noise (`depth_std`, `flying_pixels`), hull bounding boxes grow outward while `least_squares` averages across faces to keep millimeter precision.
- **Level 2:** DBSCAN separates $2$–$4$ boxes cleanly; optimization refines every face simultaneously.

### 5.2 Level 3: Touching Same-Color Boxes (What Is Solvable vs. Impossible?)
When same-color boxes touch, DBSCAN merges them into **one blob**:

```
SOLVABLE (geometry reveals seam):        IMPOSSIBLE (zero seam information!):
  ┌──────┐                                 ┌────────┬────────┐
  │ Box1 │┌──────┐  ◄── height step        │  Box1  │  Box2  │ ◄── identical height,
  │      ││ Box2 │                         │  (L1)  │  (L2)  │     aligned, same color!
  └──────┴┴──────┘                         └────────┴────────┘
                                            Where is the dividing line?
                                            (L1=10, L2=30) gives the EXACT SAME
                                            outer point cloud as (L1=20, L2=20)!
```

1. **Different heights:** **Solvable** (step on the top surface marks the seam).
2. **Same height, different widths / L-shapes:** **Solvable** (step along the side silhouette marks the seam).
3. **Rotated relative to each other:** **Solvable** (non-parallel walls reveal two distinct angles).
4. **Identical, perfectly aligned boxes flush side-by-side:** **IMPOSSIBLE.** Because the interior seam is hidden and all outer faces align seamlessly, infinitely many splits (`L1 + L2 = 2L`) produce the **exact same 3D point cloud**.
5. **Stacked boxes:** **Solvable** if the top box leaves part of the lower box's top visible; **partially impossible** if the top box sits flush on all 4 sides or overhangs completely (looks like one single tall box).

### 5.3 Level 4: Touching Different-Color Boxes (3 Approaches)
1. **Color segmentation + Bounding Box:** Fast, but `misalignment_px` bleeds neighbor colors onto edges, dragging the bounding box across the seam.
2. **Color segmentation + Optimization:** Better, because `soft_l1` ignores stray miscolored edge points, though bad initial color clustering can still hurt.
3. **Color inside the Objective Function:** Each box has a color $\mathbf{c}_b \in [0, 1]^3$, and each point minimizes combined geometric + color error:
   $$e_{i,b} = \sqrt{ D(\mathbf{p}_i, \mathbf{\theta}_b)^2 + \left( \lambda_{\text{color}} \, \|\mathbf{y}_i - \mathbf{c}_b\|_2 \right)^2 }$$
   - **Why Approach 3 wins on color noise:** A miscolored pixel sitting $10\text{ cm}$ away from box B still pays a huge geometric penalty $D$ if assigned to box B, so geometry overrides isolated wrong colors!
   - **Weighting $\lambda_{\text{color}}$:** Since geometry $D$ is in **meters** ($\sim 0.005\text{ m}$ noise) and RGB differences are in $[0, 1]$ ($\sim 0.3$–$1.0$ between distinct colors), setting $\lambda_{\text{color}} \approx 0.02\text{ to }0.05$ makes a strong color mismatch cost roughly $2$–$5\text{ cm}$ of geometric error.

---

## 6. Extras Overview

- **Photometric Correction (`--exposure_changes`):** Each camera $k$ scales/shifts brightness ($I_k \approx s_k I_0 + b_k$). Find points seen by multiple cameras, anchor camera `000` ($s_0 = 1, b_0 = 0$), solve $(s_k, b_k)$ per camera with `least_squares`, then correct RGB images before Level 4 color matching.
- **Real RGB-D Data:** Capture a real scene, measure boxes with calipers, and check whether systematic depth bias/specular reflections change the simulator trade-offs.
- **Joint Optimization / Unknown $K$:** Simultaneously refine camera poses + boxes, or incrementally spawn boxes on unexplained residual clusters.
