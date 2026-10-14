"""Apply the fitted iPhone -> RealSense color calibration to an image.

The transform is a 3x3 color correction matrix M applied in linear (gamma-decoded)
sRGB space:

    rgb_linear_realsense = rgb_linear_iphone @ M

M is a least-squares fit on the 24-patch ColorChecker Classic, photographed
under the same light by the iPhone (source) and by the RealSense D435 color
camera (target) with AE/AWB off, exposure 100 µs and white balance 3000 K (see
the paper's appendix on color calibration). The fit uses the per-patch medians
of data/calibration/color_chart/{iphone,realsense}.jpg after sRGB
linearization. To refit, or to check the matrix below against the photos:

    python scripts/calibration/fit_color_matrix.py

Accuracy on the chart (CIE76 Delta E vs RealSense, in sRGB-Lab):
    mean = 4.69, max = 10.73  (uncorrected: mean 17.92, max 47.01)

The gsplat scene and the object meshes are both in the iPhone's color domain,
so scripts/data/offline_renderer.py applies this once per frame, to the
depth-composited image, before the optional HSV jitter.

Only depends on numpy.
"""
from __future__ import annotations

import numpy as np

# Fitted 3x3 CCM. Applied as: rgb_lin_out = rgb_lin_in @ M_IPHONE_TO_REALSENSE
M_IPHONE_TO_REALSENSE = np.array([
    [ 0.63123419,  0.00730587, -0.00969861],
    [ 0.02874080,  0.63693131,  0.23368252],
    [-0.01403240,  0.04908152,  0.62365492],
], dtype=np.float64)


def _srgb_to_linear(srgb_01: np.ndarray) -> np.ndarray:
    s = np.asarray(srgb_01, dtype=np.float64)
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(lin_01: np.ndarray) -> np.ndarray:
    l = np.clip(np.asarray(lin_01, dtype=np.float64), 0, None)
    out = np.where(l <= 0.0031308, l * 12.92, 1.055 * (l ** (1 / 2.4)) - 0.055)
    return np.clip(out, 0, 1)


def calibrate_iphone_to_realsense(image: np.ndarray) -> np.ndarray:
    """Map an iPhone-captured image to RealSense color response.

    Parameters
    ----------
    image : np.ndarray
        Input image in sRGB encoding with **RGB channel order** (not BGR).
        - uint8 (0-255), float32/float64 in [0, 1], or float in [0, 255]: all accepted.
        - Shape (H, W, 3) or (N, 3).
        - 4-channel inputs (RGBA) keep the alpha channel untouched.

    Returns
    -------
    np.ndarray
        Calibrated image, same shape and dtype as the input.

    Notes
    -----
    Input is treated as sRGB (the usual JPEG/PNG encoding). If you have a linear
    RGB image already, multiply by M_IPHONE_TO_REALSENSE directly and skip the
    gamma encode/decode.

    If you have a BGR image (e.g. straight from `cv2.imread`), do:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = calibrate_iphone_to_realsense(rgb)
        bgr_out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    """
    arr = np.asarray(image)
    if arr.ndim < 2 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"expected (...,3) or (...,4) array, got shape {arr.shape}")

    # Detect & normalize the input range so output matches input dtype/range.
    in_dtype = arr.dtype
    if np.issubdtype(in_dtype, np.integer):
        max_val = float(np.iinfo(in_dtype).max)
        scale_back = "uint"
    else:
        max_val = 255.0 if arr.max() > 1.5 else 1.0
        scale_back = "float"

    has_alpha = arr.shape[-1] == 4
    rgb = arr[..., :3].astype(np.float64) / max_val

    # sRGB -> linear -> apply 3x3 -> linear -> sRGB
    flat = rgb.reshape(-1, 3)
    lin = _srgb_to_linear(flat)
    out_lin = lin @ M_IPHONE_TO_REALSENSE
    out_lin = np.clip(out_lin, 0, 1)
    out_srgb = _linear_to_srgb(out_lin).reshape(rgb.shape)

    out = out_srgb * max_val
    if scale_back == "uint":
        out = np.rint(out).clip(0, max_val).astype(in_dtype)
    else:
        out = out.astype(in_dtype, copy=False)

    if has_alpha:
        out = np.concatenate([out, arr[..., 3:4]], axis=-1)
    return out


if __name__ == "__main__":
    # Example: calibrate an iPhone JPEG and write the result.
    import sys
    from PIL import Image

    if len(sys.argv) != 3:
        print("usage: python iphone_to_realsense.py <input.jpg> <output.png>")
        sys.exit(0 if sys.argv[1:] in (["-h"], ["--help"]) else 1)

    img = np.array(Image.open(sys.argv[1]).convert("RGB"))
    out = calibrate_iphone_to_realsense(img)
    Image.fromarray(out).save(sys.argv[2])
    print(f"wrote {sys.argv[2]}")
