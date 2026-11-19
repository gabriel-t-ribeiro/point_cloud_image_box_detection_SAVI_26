# Results: baseline vs registration fixes

All scenes generated with the simulator configs (levels 1-4 from `SAVI_26-27/Trabalho1/simulator/configs`,
level 0 from `gw/configs/level_0_empty.json`), with noise. Full logs and `result.json` per scene in the two folders.

- **baseline_4b3f844**: code at commit `4b3f844` (git tag `baseline-before-registration-fixes`)
- **registration_fixes**: last camera also RANSAC-matched against camera 000 (ring closure);
  camera 000 refined too in the refinement passes and poses rebased on it; `REFINEMENT_PASSES = 3`

Registration error = mean over cameras vs ground truth. IoU = mean 3D IoU (missing boxes count 0).

| Scene | Boxes | Registration (baseline) | Registration (fixes) | BBox IoU (baseline → fixes) | Optimization IoU (baseline → fixes) | Time (s) |
|---|---|---|---|---|---|---|
| level_0_empty | 0 | 13.60 mm / 0.561° | 6.11 mm / 0.215° | – | – | 107 → 149 |
| level_1 | 1 | 5.91 mm / 0.257° | 3.11 mm / 0.102° | 0.970 → 0.977 | 0.971 → 0.977 | 90 → 128 |
| level_2 | 3 | 7.11 mm / 0.395° | 2.99 mm / 0.156° | 0.972 → 0.985 | 0.971 → 0.984 | 73 → 129 |
| level_3_aligned | 2 | 9.08 mm / 0.451° | 4.09 mm / 0.160° | 0.334 → 0.336 | 0.631 → 0.637 | 93 → 130 |
| level_3_heights | 2 | 100.13 mm / 0.376° | 3.61 mm / 0.125° | 0.000 → 0.484 | 0.000 → 0.976 | 80 → 134 |
| level_3_rotated | 2 | 7.89 mm / 0.358° | 2.46 mm / 0.067° | 0.277 → 0.279 | 0.964 → 0.977 | 75 → 118 |
| level_3_widths | 2 | 6.61 mm / 0.303° | 2.47 mm / 0.090° | 0.304 → 0.309 | 0.951 → 0.967 | 88 → 85 |
| level_3_stacked | 2 | 119.99 mm / 0.467° | 4.36 mm / 0.186° | 0.326 → 0.329 | 0.327 → 0.968 | 75 → 81 |
| level_4 | 3 | 8.34 mm / 0.357° | 2.69 mm / 0.080° | 0.196 → 0.201 | **0.505 → 0.142** | 91 → 109 |

Open problems: level_4 optimization got worse (box splitting, not registration: registration improved),
level_3_aligned optimization stays at 0.637.
