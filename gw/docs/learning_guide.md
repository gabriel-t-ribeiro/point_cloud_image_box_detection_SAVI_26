# SAVI - Trabalho Prático 1 (SAVIBoxes): Master's Engineering & Computer Vision Learning Guide

This is a self-study engineering guide for **Trabalho Prático 1 (SAVIBoxes)**, bridging the physical intuition, mathematical derivations, conceptual defense answers, and isolated Open3D / OpenCV / SciPy API mechanics behind every stage of the pipeline.

---

## 0. The Big Picture & Core Mathematical Foundations

### 0.1 Why This Pipeline Exists
An RGB-D camera looking at boxes on a table from a single viewpoint suffers from **self-occlusion** (you cannot see the back walls of a box) and **perspective foreshortening**. To measure box dimensions and poses to **millimeter accuracy**, a perception system must:
1. Fuse multiple views around the table into one dense, metrically consistent 3D point cloud (**Registration**).
2. Anchor the scene to a physical reference frame (the tabletop) and isolate foreground objects from background clutter (**Segmentation**).
3. Fit exact parametric 3D box models to noisy, potentially touching or partially occluded point measurements (**Optimization**).

```
[Multi-View RGB-D Clouds: 000.ply ... K.ply]
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ Part 1: Multi-View Registration                        │
│  Filter & Downsample ──► FPFH + RANSAC (Coarse)        │
│                     ──► Full-Cloud Point-to-Plane ICP  │
│                     ──► Aggregate in camera0 Frame     │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ Part 2: Tabletop Segmentation                          │
│  RANSAC Plane Fit ──► Construct Table Frame T_c0_table │
│  Crop Above Table ──► Euclidean Clustering (DBSCAN)    │
│                   ──► Optional Color Separation (Lv 4) │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ Part 3: Box Pose & Dimension Estimation                │
│  Baseline: OBB / 5-Plane Bounding Box (Initial Guess)  │
│  Main:     Joint Nonlinear Least Squares (least_squares)│
│            with Dynamic Point-to-Box Association       │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ Evaluation: 3D IoU, Position/Height/Dim/Yaw Errors     │
└────────────────────────────────────────────────────────┘
```

---

### 0.2 Coordinate Frames & Homogeneous Transformations in $\mathbf{SE}(3)$

Every rigid transformation $\mathbf{T}_{A, B} \in \mathbf{SE}(3)$ maps a point $\mathbf{p}_B$ expressed in frame $B$ into frame $A$:

$$\begin{bmatrix} \mathbf{p}_A \\ 1 \end{bmatrix} = \mathbf{T}_{A, B} \begin{bmatrix} \mathbf{p}_B \\ 1 \end{bmatrix} = \begin{bmatrix} \mathbf{R}_{A, B} & \mathbf{t}_{A, B} \\ \mathbf{0}^T & 1 \end{bmatrix} \begin{bmatrix} \mathbf{p}_B \\ 1 \end{bmatrix}$$

where:
- $\mathbf{R}_{A, B} \in \mathbf{SO}(3)$ ($\mathbf{R}^T\mathbf{R} = \mathbf{I}$, $\det(\mathbf{R}) = +1$) has columns equal to the $x, y, z$ unit axes of frame $B$ expressed in frame $A$.
- $\mathbf{t}_{A, B} \in \mathbb{R}^3$ is the origin of frame $B$ expressed in frame $A$.
- **Composition & Inversion:**
  $$\mathbf{T}_{A, C} = \mathbf{T}_{A, B} \, \mathbf{T}_{B, C}, \qquad \mathbf{T}_{B, A} = \mathbf{T}_{A, B}^{-1} = \begin{bmatrix} \mathbf{R}_{A, B}^T & -\mathbf{R}_{A, B}^T \mathbf{t}_{A, B} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

In SAVIBoxes, four frames interact:
1. **`camera_k`:** Sensor frame of view $k$ ($x$ right, $y$ down, $z$ forward).
2. **`camera0`:** Reference view (`000`). All final poses in `result.json` are expressed in `camera0`.
3. **`table`:** Orthonormal frame with $z_{\text{table}}$ perpendicular to the tabletop pointing **up**, and $z=0$ at the tabletop surface.
4. **`box_b`:** Centered at the geometric center of box $b$, with $x$ along `length` ($L$), $y$ along `width` ($W$), and $z$ along `height` ($H$, parallel to $z_{\text{table}}$). Because `z` in the table frame denotes the **base height** of the box above the table, the box center in table coordinates is $\mathbf{c}_b = [x_b,\; y_b,\; z_b + H_b/2]^T$.

---

### 0.3 Core Error Metrics & Box Symmetries

#### Registration Error ($\mathbf{E} = \mathbf{T}_{\text{est}}^T \mathbf{T}_{\text{gt}}$ in relative form)
Given estimated pose $\mathbf{T}_{\text{est}}$ and ground-truth $\mathbf{T}_{\text{gt}}$:
- **Translation error:** $e_t = \|\mathbf{t}_{\text{est}} - \mathbf{t}_{\text{gt}}\|_2 \times 1000\;\text{[mm]}$ (or $\|\mathbf{t}_E\|_2$ from $\mathbf{E} = \mathbf{T}_{\text{gt}}^{-1}\mathbf{T}_{\text{est}}$).
- **Rotation error:** From $\mathbf{R}_E = \mathbf{R}_{\text{est}}^T \mathbf{R}_{\text{gt}}$, Euler's rotation theorem gives:
  $$e_r = \arccos\left(\operatorname{clip}\left(\frac{\operatorname{tr}(\mathbf{R}_E) - 1}{2},\; -1,\; 1\right)\right) \quad \text{[rad]} \longrightarrow \text{[deg]}$$

#### Box Symmetry & 3D Intersection over Union ($\text{IoU}_{3\text{D}}$)
Because a box's $z$-axis always points up, rotating its local frame by $180^\circ$ around $z$ leaves the physical box unchanged, and swapping `length` $\leftrightarrow$ `width` while adding $90^\circ$ to yaw produces the exact same occupied volume. Therefore:
- **Yaw & Dimension Errors:** Compare each estimate against both $(\psi_{\text{gt}}, L_{\text{gt}}, W_{\text{gt}})$ and $(\psi_{\text{gt}} + \frac{\pi}{2}, W_{\text{gt}}, L_{\text{gt}})$ with yaw difference wrapped modulo $\pi$ into $[-\frac{\pi}{2}, +\frac{\pi}{2})$, selecting the swap with minimum $|\Delta \psi|$.
- **3D IoU (Symmetry-Invariant):** For $z$-aligned boxes, the 3D intersection volume factors cleanly into a 2D rotated-rectangle intersection area $A_{\cap}$ in the $(x, y)$ plane times a 1D height-interval overlap $\Delta z_{\cap}$:
  $$\Delta z_{\cap} = \max\!\left(0,\; \min(z_1 + H_1,\; z_2 + H_2) - \max(z_1,\; z_2)\right)$$
  $$V_{\cap} = A_{\cap} \cdot \Delta z_{\cap}, \qquad \text{IoU}_{3\text{D}} = \frac{V_{\cap}}{L_1 W_1 H_1 + L_2 W_2 H_2 - V_{\cap}} \in [0, 1]$$

---

### 0.4 Nonlinear Least Squares Engine (`scipy.optimize.least_squares`)

Given parameter vector $\mathbf{\theta} \in \mathbb{R}^P$ and residual vector $\mathbf{r}(\mathbf{\theta}) \in \mathbb{R}^M$, nonlinear least squares minimizes:

$$\min_{\mathbf{\theta}} \frac{1}{2} \sum_{i=1}^{M} \rho\!\left(r_i(\mathbf{\theta})^2\right)$$

where $\rho(\cdot)$ is a **robust loss function** (e.g., $\rho(u) = u$ for standard L2, or Huber / Soft-L1 / Cauchy to suppress outliers). At each step, the solver approximates $\mathbf{r}(\mathbf{\theta} + \Delta\mathbf{\theta}) \approx \mathbf{r}(\mathbf{\theta}) + \mathbf{J}\mathbf{\Delta\theta}$ with Jacobian $\mathbf{J} = \frac{\partial \mathbf{r}}{\partial \mathbf{\theta}}$ and solves the damped Gauss–Newton (Levenberg–Marquardt / Trust Region Reflective) system:

$$(\mathbf{J}^T \mathbf{W} \mathbf{J} + \lambda \mathbf{I})\,\mathbf{\Delta\theta} = -\mathbf{J}^T \mathbf{W} \mathbf{r}$$

---

## 1. Part 1: Multi-View Point Cloud Registration

### 1.1 Physical & Engineering Intuition
Individual RGB-D views are captured in their own camera frames (`000`, `001`, ...), initially separated by unknown rotations up to $360^\circ$ around the table.
- **Why ICP alone fails:** ICP matches each point to its *nearest current neighbor*. If two views start $45^\circ$ apart, nearest neighbors pair the wrong box sides with the table edge, trapping ICP in a bad local minimum.
- **Why Coarse-to-Fine works:**
  1. **Coarse alignment (FPFH + RANSAC):** Computes rotation-invariant 3D local geometry descriptors (Fast Point Feature Histograms) on a voxel-downsampled cloud and uses RANSAC to find a rigid consensus transform $\mathbf{T}^{(0)}$ unaffected by arbitrary initial pose.
  2. **Fine alignment (Point-to-Plane ICP):** Refines $\mathbf{T}^{(0)}$ on dense clouds using tangent-plane projections down to millimeter accuracy.

### 1.2 Rigorous Mathematical Formulation

1. **FPFH Feature Matching + RANSAC (Coarse):**
   For each point $\mathbf{p}_i$ with estimated normal $\mathbf{n}_i$, FPFH encodes a 33-bin histogram of relative Darboux-frame angles $(\alpha, \phi, \theta)$ to neighbors within radius $r_{\text{feat}}$. RANSAC repeatedly samples $3$ correspondences $(\mathbf{p}_{s,k}, \mathbf{p}_{t,k})$, computes the Kabsch-Umeyama rigid transform $\mathbf{T}$, and scores the number of inliers satisfying $\|\mathbf{T}\mathbf{p}_{s,i} - \mathbf{p}_{t,j}\|_2 < \tau_{\text{ransac}}$.

2. **Point-to-Plane ICP (Fine):**
   Given current correspondences $\mathcal{K} = \{(\mathbf{p}_i, \mathbf{q}_i)\}$ within threshold $d_{\max}$:
   $$E_{\text{p2l}}(\mathbf{R}, \mathbf{t}) = \sum_{(\mathbf{p}_i, \mathbf{q}_i) \in \mathcal{K}} \rho\!\left( \left[ \left(\mathbf{R}\mathbf{p}_i + \mathbf{t} - \mathbf{q}_i\right) \cdot \mathbf{n}_{\mathbf{q}_i} \right]^2 \right)$$
   Multi-stage ICP decreases $d_{\max}$ across passes (e.g., $8\text{ cm} \to 3\text{ cm} \to 1\text{ cm}$) so early passes pull in misaligned structures while final passes reject flying-pixel edge noise.

### 1.3 Answers to the Professor's Conceptual Questions

1. **What happens to ICP if you remove the background and leave only the table and boxes? Why?**
   - **Answer:** Registration degrades severely or drifts along the table plane. A flat tabletop + small symmetric boxes have **few unique 3D geometric constraints** (a plane constrains translation only along its normal and two tilt angles, leaving **in-plane translation and yaw around the table normal weakly constrained or ambiguous**). Including the surrounding room (walls, corners, chairs, floor) provides orthogonal planes in all three directions ($x, y, z$) that lock all $6$ DoF of $\mathbf{SE}(3)$.
2. **What happens to the error when chaining many images? Does the last image align with the first?**
   - **Answer:** **Error accumulates (drifts)** across sequential pairwise transforms: $\mathbf{T}_{0, k} = \mathbf{T}_{0, 1}\mathbf{T}_{1, 2}\cdots\mathbf{T}_{k-1, k}$. Even a $0.3^\circ$ rotation error per pair becomes $>2^\circ$ after $7$–$8$ hops, causing a **loop-closure gap** where the last camera (`007`) fails to align cleanly with camera `000` (seen as double walls or ghosted box edges). Registering directly against the **growing aggregated target cloud** (or performing global multi-way / pose-graph refinement) anchors every new view to `000` and all previously aligned views, suppressing drift.
3. **What effect do registration errors have on the aggregated point cloud?**
   - **Answer:** Misalignment creates **thickened or doubled surfaces ("ghosting")** on box walls and corners. Because downstream bounding boxes and surface distances interpret those ghosted double layers as wider objects or noisy outliers, a $10\text{ mm}$ registration drift directly inflates estimated box dimensions and lowers 3D IoU.

### 1.4 API Mechanics & Minimal Isolated Snippet
Reference: [Open3D Global Registration](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html) & [ICP Registration](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html).
- `pcd.voxel_down_sample(voxel_size)`: Reduces density uniformly in 3D space.
- `pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))`: PCA surface normals.
- `o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param)`: 33-D FPFH descriptors.
- `o3d.pipelines.registration.registration_ransac_based_on_feature_matching(...)`: Coarse alignment.
- `o3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance, init, estimation_method)`: Fine alignment.

```python
import numpy as np
import open3d as o3d

# Isolated Demonstration: Coarse FPFH+RANSAC followed by Point-to-Plane ICP
src = o3d.geometry.PointCloud()
src.points = o3d.utility.Vector3dVector(np.random.rand(500, 3))
tgt = o3d.geometry.PointCloud(src)  # Copy and apply a known shift
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
The aggregated cloud contains floor, walls, table legs, tabletop, and boxes. Before fitting boxes:
1. **Find the tabletop plane:** In a room scene, walls may actually have *more* points than the table! Naive "take the single largest RANSAC plane" fails if a wall dominates. We must run **sequential RANSAC** and select the largest candidate whose normal aligns with the camera's approximate "up" direction (and reject floors too far below/above).
2. **Build the table frame $\mathbf{T}_{\text{camera0}, \text{table}}$:** Transforming everything into the table frame turns a tilted 3D problem into a clean **$z$-up** problem where $z = 0$ is the table surface.
3. **Crop & Cluster:** Keeping points with $z_{\min} < z < z_{\max}$ inside the table's $(x, y)$ footprint removes both the table itself and the surrounding room, leaving isolated islands of points that DBSCAN separates into object clusters.

### 2.2 Rigorous Mathematical Formulation

1. **Plane Equation & Signed Orientation:**
   A plane $\Pi$ satisfies $\mathbf{n}^T \mathbf{p} + d = 0$ with $\|\mathbf{n}\|_2 = 1$. To ensure $\mathbf{n}$ points **upward** (toward the camera / opposite gravity $\mathbf{g}_{\text{up}}$), flip sign if $\mathbf{n} \cdot \mathbf{u}_{\text{guess}} < 0$:
   $$(\mathbf{n}, d) \leftarrow (-\mathbf{n}, -d) \quad \text{if } \mathbf{n}^T \mathbf{u}_{\text{guess}} < 0$$
2. **Orthonormal Table Basis from Plane Inliers:**
   Let $\mathbf{z}_t = \mathbf{n}$ and $\mathbf{p}_0$ be the centroid of table inliers projected onto $\Pi$. Choose an arbitrary horizontal reference vector (or principal PCA axis of the table inliers) $\mathbf{v}$, then construct a right-handed basis via cross products:
   $$\mathbf{x}_t = \frac{\mathbf{v} - (\mathbf{v} \cdot \mathbf{z}_t)\mathbf{z}_t}{\|\mathbf{v} - (\mathbf{v} \cdot \mathbf{z}_t)\mathbf{z}_t\|_2}, \qquad \mathbf{y}_t = \mathbf{z}_t \times \mathbf{x}_t, \qquad \mathbf{T}_{\text{camera0}, \text{table}} = \begin{bmatrix} \mathbf{x}_t & \mathbf{y}_t & \mathbf{z}_t & \mathbf{p}_0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
3. **Euclidean Clustering (DBSCAN):**
   Groups points whose $\varepsilon$-neighborhoods ($\|\mathbf{p}_i - \mathbf{p}_j\|_2 \le \varepsilon$) form density-connected components of at least `min_points`.
4. **Precision & Recall Evaluation (Isolated from Registration Error):**
   Using the **true** camera transforms, a point $\mathbf{p}$ lies inside ground-truth box $b$ (with tolerance margin $m \approx 5\text{ mm}$) iff in box $b$'s local frame:
   $$|q_x| \le \frac{L_b}{2} + m \;\land\; |q_y| \le \frac{W_b}{2} + m \;\land\; |q_z| \le \frac{H_b}{2} + m$$
   For predicted cluster $C$ matched to ground-truth point set $G_b$:
   $$\text{Precision} = \frac{|C \cap G_b|}{|C|}, \qquad \text{Recall} = \frac{|C \cap G_b|}{|G_b|}$$

### 2.3 API Mechanics & Minimal Isolated Snippet
Reference: [Open3D Plane Segmentation](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Plane-segmentation) & [DBSCAN Clustering](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#DBSCAN-clustering).
- `pcd.segment_plane(distance_threshold, ransac_n=3, num_iterations=1000) -> (plane_model, inliers)`
- `pcd.cluster_dbscan(eps, min_points) -> IntVector` (returns `-1` for noise points)

```python
import numpy as np
import open3d as o3d

# Isolated Demonstration: Plane removal + DBSCAN clustering
rng = np.random.default_rng(0)
table = np.column_stack([rng.uniform(-0.5, 0.5, 400), rng.uniform(-0.5, 0.5, 400), np.zeros(400)])
box1  = rng.uniform([-0.2, -0.1, 0.02], [-0.05, 0.1, 0.15], size=(120, 3))
box2  = rng.uniform([0.1, -0.1, 0.02], [0.25, 0.1, 0.12], size=(120, 3))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.vstack([table, box1, box2]))

plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=500)
objects_pcd = pcd.select_by_index(inliers, invert=True)

labels = np.array(objects_pcd.cluster_dbscan(eps=0.04, min_points=20))
n_clusters = labels.max() + 1
print(f"Plane: {np.round(plane_model, 3)} | Found {n_clusters} object clusters")
```

---

## 3. Part 3A: Base Method — Oriented & 5-Plane Bounding Boxes

### 3.1 Physical & Engineering Intuition
Before building a nonlinear optimizer, a **bounding box baseline** establishes both a benchmark and an **initial guess** $\mathbf{\theta}_0$:
- **Open3D OBB (`get_oriented_bounding_box` / `get_minimal_oriented_bounding_box`):** Wraps an arbitrary 3D box around the convex hull. Its axes are **not** constrained to keep $z$ perpendicular to the table, so slight noise tilts the box and corrupts `z` and `height`.
- **2D `cv2.minAreaRect` + Height Percentiles:** Projects points onto the table $(x, y)$ plane, finds the minimum-area enclosing rectangle, and uses percentiles (e.g., 1st and 99th) for height.
- **5-Plane Normal-Based Bounding Box:** Estimates yaw from side-wall normals (`|n_z| < 0.3`) and wall positions via medians of wall-facing points (ignoring corners/outliers), plus a `z`-histogram peak for the top face.

### 3.2 Answers to the Professor's Conceptual Questions

- **Why is the bounding box so sensitive to noise? Which points does it actually use?**
  - **Answer:** A minimum enclosing bounding box (`minAreaRect`, `get_minimal_oriented_bounding_box`, or min/max) is determined **exclusively by the few outermost extreme points on the convex hull** — not by the thousands of interior face points! A single noisy flying pixel sticking $2\text{ cm}$ out from an edge drags the entire box face outward by $2\text{ cm}$ and can rotate `minAreaRect` to align with that outlier. Even with outlier removal, extreme-value estimators have high variance compared to fitting all surface points simultaneously.

### 3.3 API Mechanics & Minimal Isolated Snippet
Reference: [Open3D Bounding Volumes](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Bounding-volumes) & [OpenCV `minAreaRect`](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html).
- `pcd.get_minimal_oriented_bounding_box()` -> `o3d.geometry.OrientedBoundingBox` (`.center`, `.R`, `.extent`).
- `cv2.minAreaRect(points_2d.astype(np.float32))` -> `((cx, cy), (w, h), angle_deg)`.

```python
import cv2
import numpy as np

# Isolated Demonstration: 2D z-up footprint via cv2.minAreaRect
pts_xy = np.random.uniform([-0.15, -0.10], [0.15, 0.10], size=(200, 2)).astype(np.float32)
theta = np.radians(25.0)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
rotated_xy = pts_xy @ R.T

(cx, cy), (length, width), angle_deg = cv2.minAreaRect(rotated_xy)
print(f"Center=({cx:.3f}, {cy:.3f}), Size=({length:.3f}, {width:.3f}), Angle={angle_deg:.1f} deg")
```

---

## 4. Part 3B: Main Method — Nonlinear Optimization Box Fitting

This is the **core of Trabalho 1**.

### 4.1 Physical & Engineering Intuition
Instead of trusting only extreme hull points (bounding box) or committing prematurely to cluster boundaries (which fail whenever boxes touch in Level 3), we parameterize all $K$ boxes simultaneously in a single parameter vector $\mathbf{\theta} \in \mathbb{R}^{7K}$ and let **every point in the scene** pull the box surfaces toward the data.

Crucially, **every point chooses its best-explaining box on every single residual evaluation**. This turns hard clustering + fitting into a single continuous optimization problem.

### 4.2 Rigorous Mathematical Formulation

#### 1. Parameterization ($\mathbf{\theta} \in \mathbb{R}^{7K}$)
Each box $b \in \{1, \dots, K\}$ in the table frame has $7$ parameters:
$$\mathbf{\theta}_b = [x_b,\; y_b,\; z_b,\; \psi_b,\; L_b,\; W_b,\; H_b]^T$$
where $z_b \ge 0$ is the base altitude above the table ($z_b = 0$ if resting on the table, or $H_{\text{lower}}$ if stacked).

#### 2. Point-to-Box Surface Distance (Exact Analytical Form)
For any 3D point $\mathbf{p}_i$ in the table frame, transform it into box $b$'s centered local frame:
$$\mathbf{c}_b = \begin{bmatrix} x_b \\ y_b \\ z_b + H_b / 2 \end{bmatrix}, \qquad \mathbf{q}_{i,b} = \mathbf{R}_z(\psi_b)^T (\mathbf{p}_i - \mathbf{c}_b)$$

Let $\mathbf{h}_b = [L_b/2,\; W_b/2,\; H_b/2]^T$ be the half-extents, and compute component-wise signed excess:
$$\mathbf{d} = |\mathbf{q}_{i,b}| - \mathbf{h}_b = \begin{bmatrix} |q_x| - L_b/2 \\ |q_y| - W_b/2 \\ |q_z| - H_b/2 \end{bmatrix}$$

The **unsigned distance from point $\mathbf{p}_i$ to the 6 faces (surface) of box $b$** is:
$$D(\mathbf{p}_i, \mathbf{\theta}_b) = \underbrace{\left\| \max(\mathbf{d}, \mathbf{0}) \right\|_2}_{\text{outside distance}} \;-\; \underbrace{\min\!\left(\max(d_x, d_y, d_z),\; 0\right)}_{\text{inside distance to nearest face}}$$

> **Why subtract $\min(\max(d_x, d_y, d_z), 0)$?**
> If you only used $\|\max(\mathbf{d}, \mathbf{0})\|_2$ (distance to the *solid* box), every point strictly *inside* the box would yield $0$ error! The optimizer would immediately blow up a single box to cover the whole room and get $0$ residual everywhere. Penalizing interior points by their depth to the nearest face forces the box walls to hug the actual surface points from both inside and outside.

#### 3. Multi-Box Objective with Dynamic Association
For $K$ boxes, each point $i \in \{1, \dots, N\}$ is assigned inside the residual call to whichever box explains it best:
$$r_i(\mathbf{\theta}) = \min_{b \in \{1, \dots, K\}} D(\mathbf{p}_i, \mathbf{\theta}_b)$$

With `scipy.optimize.least_squares(..., loss='soft_l1', f_scale=0.01)`, outliers (flying pixels, stray registration noise) receive linear rather than quadratic influence once $|r_i| > f_{\text{scale}}$.

### 4.3 Answers to the Professor's 5 Defense Questions

1. **Can a box grow without the error increasing? In what situations and why? What did you do to solve it?**
   - **Answer:** **Yes.** Two classic situations allow zero-penalty growth:
     1. **Occluded / unseen face:** If no points lie on the back wall (or bottom face on the table), pushing that face outward moves zero points into the interior, so the surface distance of all observed points stays unchanged!
     2. **Solid-distance mistake:** If the distance function returned $0$ for interior points instead of distance-to-surface, any expansion leaves residuals at $0$.
   - **Mitigation:** (a) Always measure distance to the **box surface** (including interior points), (b) enforce $z_b \ge 0$ (or $z_b = 0$ for non-stacked boxes) via bounds, and (c) add a small **minimum-volume / size-regularization residual** $w_{\text{vol}} \sqrt{L_b W_b H_b}$ (or gentle shrinkage prior on $L_b, W_b, H_b$) so unobserved faces collapse snugly instead of drifting outward.

2. **Can two boxes occupy the same space? What happens in optimization if this is allowed?**
   - **Answer:** **Yes, if unpenalized.** Two phenomena occur:
     1. **Box collapse / duplication:** Two boxes converge onto the exact same large physical box (sharing its points with low error), leaving another smaller box unrepresented (or absorbed as outliers if robust loss is active).
     2. **Penetration at touching faces (Level 3):** Boxes interpenetrate slightly to minimize local residuals on noisy touching edges.
   - **Mitigation:** Good initialization (e.g., farthest-point / sequential residual-peeling or split-cluster initialization so each box starts in a distinct occupied region) plus an explicit **3D overlap penalty** residual between every pair of boxes $(a, b)$ proportional to their intersection volume (or differentiable AABB/OBB overlap).

3. **Why does this fitting avoid an outer loop alternating between associating points and optimizing?**
   - **Answer:** Because $\min_{b} D(\mathbf{p}_i, \mathbf{\theta}_b)$ is evaluated **at every residual call inside `least_squares`**. As finite-difference Jacobians perturb each box parameter $\theta_j \to \theta_j + \delta$, points near the boundary between two touching boxes smoothly (piecewise-smoothly) switch ownership when one box moves closer than the other. An outer hard-assignment loop locks wrong initial associations during the continuous step and can oscillate or get stuck in poor local minima.

4. **Which points determine each parameter? Does a point in the middle of the box top help estimate its length?**
   - **Answer:** Only points whose **closest face** is controlled by a parameter produce a nonzero partial derivative $\frac{\partial r_i}{\partial \theta_j}$:
     - **Top face points** ($|q_z| - H/2$ largest) determine $z + H$ (height / top altitude). A point in the middle of the top face has $\frac{\partial r_i}{\partial L} = 0$, so it provides **zero** information about `length`, `width`, `x`, `y`, or `yaw`!
     - **Side wall points** perpendicular to local $x$ determine $x$ and $L$; side wall points perpendicular to local $y$ determine $y$ and $W$.
     - **Off-center side wall points** (away from wall midpoints) create torque/lever arm to determine `yaw` ($\psi$).

5. **Why is the bounding box so sensitive to noise? Which points does it actually use?**
   - **Answer:** Covered in Section 3.2 — it relies solely on the few extreme hull points rather than averaging across the entire point density of each face.

### 4.4 API Mechanics & Minimal Isolated Snippet
Reference: [SciPy `least_squares` Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
- `least_squares(fun, x0, bounds=(lb, ub), loss='soft_l1', f_scale=0.01, args=...)`:
  - `bounds`: Keeps $L, W, H > 0$ and $z \ge 0$.
  - `loss='soft_l1'`: Linearizes residuals larger than `f_scale` (in meters, e.g., `0.01` m = 1 cm) to reject outliers.

```python
import numpy as np
from scipy.optimize import least_squares

def box_surface_distances(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Unsigned distance from N points (table frame) to a single box [x, y, z, yaw, L, W, H]."""
    x, y, z, yaw, L, W, H = params
    c = np.array([x, y, z + H / 2.0])
    ca, sa = np.cos(yaw), np.sin(yaw)
    R_T = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])
    q = (points - c) @ R_T.T
    d = np.abs(q) - np.array([L / 2.0, W / 2.0, H / 2.0])
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    return outside - inside  # > 0 outside, 0 on surface, > 0 inside

# Isolated Demonstration: Fit one synthetic noisy box
rng = np.random.default_rng(42)
# Sample points on top and two visible walls of a 0.30 x 0.20 x 0.15 box centered at (0, 0)
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

## 5. Difficulty Levels 1–4 & Identifiability Analysis

### 5.1 Levels 1 & 2 (Single & Separated Boxes)
- **Level 1:** With zero noise, a bounding box matches optimization. Once realistic sensor noise (`depth_std`, `flying_pixels`, `edge_dropouts`) is enabled, extreme-point bounding boxes inflate dimensions by $5$–$15\text{ mm}$, whereas `least_squares` averages over all surface points and recovers millimeter accuracy.
- **Level 2:** DBSCAN cleanly separates $2$–$4$ non-touching boxes; fitting each cluster independently (or jointly) scales linearly.

### 5.2 Level 3: Touching Same-Color Boxes (Which Cases Are Impossible?)
When $2$–$4$ same-color boxes touch, they merge into **one single Euclidean cluster**. Only surface geometry separates them:
1. **Different heights:** **Solvable.** A clear height step on the top surface tells the optimizer where box A ends and box B begins.
2. **Same height, different widths / L-shapes:** **Solvable (if step edges are visible).** Step discontinuities in the side silhouettes constrain the shared boundary plane.
3. **Rotated relative to each other:** **Solvable.** Different wall orientations ($\psi_1 \neq \psi_2$) produce intersecting non-collinear lines on the top/side boundary.
4. **Identical, perfectly aligned boxes flush side-by-side:** **IMPOSSIBLE (Unobservable).**
   - *Why:* Two identical $L \times W \times H$ boxes placed flush along a face of identical height and orientation produce the **exact same external 3D point cloud** as one $2L \times W \times H$ box (or any split $L_1 + L_2 = 2L$ along that axis!). Because the interior seam is completely occluded, there is **zero geometric or color information** in the data to locate the dividing plane — infinitely many parameterizations yield identical zero residuals.
5. **Stacked boxes:** **Solvable** if the upper box is smaller or shifted/rotated so the lower box's top ledge is visible; **partially unobservable** in $z_{\text{upper}}$ vs. $H_{\text{lower}}$ if the upper box completely overhangs/covers the lower box or sits flush on all four sides (identical footprint $\implies$ looks like one tall box!).

### 5.3 Level 4: Touching Different-Color Boxes (3 Approaches)

When touching boxes have different RGB colors, color resolves Level 3's ambiguities:
1. **Approach 1 — Color Segmentation + Bounding Box:** Cluster points in RGB/HSV/Lab space (e.g., K-means or color-threshold + DBSCAN), then run bounding box per color cluster. Fast, but **fragile** to `misalignment_px` color bleeding at edges (an edge pixel painted with a neighbor's color stretches the bounding box across the boundary!).
2. **Approach 2 — Color Segmentation + Optimization:** Same color clustering, followed by `least_squares` with robust loss. Much better than Approach 1 because robust loss ignores isolated miscolored edge pixels, though severe clustering splits still hurt initialization.
3. **Approach 3 — Joint Geometric + Color Objective Function:**
   Each box $b$ also carries a representative color $\mathbf{c}^{\text{rgb}}_b \in [0, 1]^3$ (estimated from cluster modes or jointly optimized). For point $i$ with position $\mathbf{p}_i$ and color $\mathbf{y}_i$:
   $$e_{i,b} = \sqrt{ D(\mathbf{p}_i, \mathbf{\theta}_b)^2 + \left( \lambda_{\text{color}} \, \|\mathbf{y}_i - \mathbf{c}^{\text{rgb}}_b\|_2 \right)^2 }$$
   and point $i$ selects $b^* = \arg\min_b e_{i,b}$ on every evaluation!
   - **Why Approach 3 is most robust:** Even if a pixel on box A gets box B's color due to `misalignment_px`, if it lies $10\text{ cm}$ inside box A's geometry, the geometric term keeps it associated with box A (or if it lies near the seam, robust `loss` downweights it).
   - **How to choose $\lambda_{\text{color}}$:** Balance units so a meaningful color mismatch compares properly with geometric errors: if $D$ is in meters ($\sim 0.005\text{ m}$ noise, $\sim 0.05\text{ m}$ cross-box error) and RGB differences in $[0, 1]$ are $\sim 0.3$–$1.0$, then $\lambda_{\text{color}} \approx 0.02\text{ to }0.05\;\text{[m / unit RGB]}$ makes a strong color mismatch equivalent to a $2$–$5\text{ cm}$ geometric shift without overpowering solid geometry.

---

## 6. Extras (Photometric Correction, Real Data & Beyond)

### 6.1 Photometric Gain & Bias Correction (`--exposure_changes`)
When auto-exposure varies between views, the true color $I_0$ at a 3D point seen by both camera $0$ and camera $k$ undergoes an affine radiometric transform:
$$I_k \approx s_k \cdot I_0 + b_k \implies I_{k, \text{corrected}} = \frac{I_k - b_k}{s_k}$$
Using registered overlapping 3D points across views (projecting them into both `rgb/000.png` and `rgb/k.png`, or directly comparing aligned cloud colors via nearest neighbors), `scipy.optimize.least_squares` with robust `loss` estimates $(s_k, b_k)$ per image relative to image `000` (anchored at $s_0 = 1, b_0 = 0$) before running Level 4 color segmentation.

### 6.2 Real RGB-D Data & Additional Extras
- **Real data:** Real depth sensors exhibit structured systematic depth bias, flying pixels around sharp corners, and specular highlights; comparing caliper ground truth against both methods tests whether simulation advantages transfer to real hardware.
- **Joint camera-box optimization:** Optimizing camera $\mathbf{SE}(3)$ poses and box parameters together removes residual registration drift on the boxes themselves.
- **Unknown $K$:** Incrementally spawn new boxes on unexplained residual clusters (e.g., greedy residual peeling + BIC/AIC or minimum-inlier stopping criterion).
