# Color calibration

Object meshes and scene splats are built from iPhone captures, while the policy
sees frames from the RealSense D435 head camera. A 3x3 matrix in linear RGB maps
the iPhone's color response to the D435's. It is stored as
`M_IPHONE_TO_REALSENSE` in `scripts/main/renderer/iphone_to_realsense.py` and is
applied once to every composited frame in `scripts/data/offline_renderer.py`
(`APPLY_IPHONE_TO_REALSENSE`).

## Inputs

`data/calibration/color_chart/`:

| file | camera | size |
|---|---|---|
| `iphone.jpg` | iPhone photo | 4032x3024 |
| `realsense.jpg` | D435 color stream, AE/AWB off, exposure 10 ms, white balance 3000 K | 640x480 |

Both show the same 24-patch ColorChecker Classic under the same lighting.

## Refit

```bash
conda activate legs
python scripts/calibration/fit_color_matrix.py
```

The script detects the chart with `cv2.mcc`, takes per-patch medians,
linearizes them, and solves the 3x3 by least squares. It prints the refit next
to the shipped matrix and a per-patch CIE76 Delta E table. On the files above
it reproduces the shipped matrix to 4.1e-9 per element (the shipped literal is
rounded to 8 decimals). With the default inputs it exits non-zero if the
difference exceeds `--tol` (default 1e-8).

| mapping | Delta E mean | Delta E max |
|---|---|---|
| none (iPhone vs D435) | 17.92 | 47.01 |
| linear 3x3 (shipped) | 4.69 | 10.73 |

`--compare-models` also reports an affine 3x4 (5.19 / 12.78) and a 13-term
root-polynomial fit (3.59 / 9.49), scored on the patches they were fit to.
Only the linear 3x3 is used.

To calibrate a different camera pair, photograph the chart with both under
the same light, then

```bash
python scripts/calibration/fit_color_matrix.py \
    --iphone my_iphone.jpg --realsense my_d435.jpg --out ccm.json
```

and paste the printed matrix into `M_IPHONE_TO_REALSENSE`.
