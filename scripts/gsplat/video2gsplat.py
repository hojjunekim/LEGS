"""
Video → Gaussian Splat training pipeline via nerfstudio.

Pipeline:
  1. Extract frames from video (uniform sampling via ffmpeg)
  2. Run SfM via nerfstudio (hloc)
  3. Train splatfacto gaussian splat
  4. Export the splat to <output-dir>/<scene_name>/export/splat.ply

Usage (in the separate gsplat env, `conda env create -f gsplat.yml`):
    conda activate legs-gsplat
    python scripts/gsplat/video2gsplat.py <scene_name> \
        --video <path_to_video.mov> \
        --config iphone16e \
        --output-dir scripts/gsplat/output

The script prints the steps that turn the export into a simulator map.
"""

import argparse
import json
import subprocess
from pathlib import Path

from nerfstudio.process_data.images_to_nerfstudio_dataset import ImagesToNerfstudioDataset

SCRIPT_DIR = Path(__file__).parent


def _ffmpeg_extract_all(video_path: str, tmp_dir: Path) -> int:
    """Extract all frames from video using ffmpeg (avoids OpenCV segfaults)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-q:v", "2", str(tmp_dir / "all_%05d.png")]
    subprocess.run(cmd, check=True, capture_output=True)
    return len(list(tmp_dir.glob("all_*.png")))


def extract_frames(video_path: str, images_dir: Path, num_images: int) -> None:
    """Extract uniformly-spaced frames from video using ffmpeg."""
    # Decode all frames via ffmpeg into a temp directory
    tmp_dir = images_dir.parent / "_tmp_frames"
    print("Decoding video with ffmpeg...")
    Ntot = _ffmpeg_extract_all(video_path, tmp_dir)
    print(f"Total frames: {Ntot}")

    # Select uniformly-spaced frames
    step = max(1, Ntot // num_images)
    selected = list(range(1, Ntot + 1, step))[:num_images]
    print(f"Selecting {len(selected)} frames (step={step})...")

    import shutil
    for out_idx, frame_idx in enumerate(selected):
        src = tmp_dir / f"all_{frame_idx:05d}.png"
        dst = images_dir / f"frame_{out_idx + 1:05d}.png"
        shutil.copy2(str(src), str(dst))

    shutil.rmtree(tmp_dir)
    print(f"Extracted {len(selected)} frames to {images_dir}")




# ── Main pipeline ─────────────────────────────────────────────────────────────

def find_video(scene_name: str, capture_dir: Path) -> str:
    matches = list(capture_dir.glob(f"*{scene_name}*"))
    if not matches:
        raise FileNotFoundError(f"No video matching '{scene_name}' in {capture_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for '{scene_name}': {matches}")
    return str(matches[0])


def run_pipeline(scene_name: str, video_path: str, config_path: Path,
                 output_dir: Path) -> None:
    with open(config_path) as f:
        cfg = json.load(f)
    num_images = cfg["extractor"]["num_images"]

    work_dir = (output_dir / scene_name).resolve()  # ns-train runs with cwd=work_dir
    images_dir = work_dir / "images"
    sfm_dir = work_dir / "sfm"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract frames
    print("\n=== Step 1: Extracting frames ===")
    if any(images_dir.glob("*.png")):
        print(f"Images already exist in {images_dir}, skipping extraction.")
    else:
        extract_frames(video_path, images_dir, num_images)

    # Step 2: SfM via nerfstudio
    print("\n=== Step 2: Running SfM (hloc) ===")
    sfm_tfm = sfm_dir / "transforms.json"
    if sfm_tfm.exists():
        print(f"SfM already done ({sfm_tfm}), skipping.")
    else:
        ns_obj = ImagesToNerfstudioDataset(
            data=images_dir, output_dir=sfm_dir,
            camera_type="perspective", matching_method="exhaustive",
            sfm_tool="hloc", gpu=True,
        )
        ns_obj.main()

    # Step 3: Train splatfacto
    print("\n=== Step 3: Training splatfacto ===")
    ns_output_dir = work_dir / "nerfstudio"
    cmd = [
        "ns-train", "splatfacto",
        "--data", str(sfm_dir),
        "--viewer.quit-on-train-completion", "True",
        "--output-dir", str(ns_output_dir),
        "--pipeline.model.camera-optimizer.mode", "SO3xR3",
        "nerfstudio-data",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(work_dir))
    if result.returncode != 0:
        raise SystemExit(f"ns-train failed with exit code {result.returncode}")
    print("Splatfacto training complete!")

    # Step 4: Export the trained splat (latest run) to a .ply
    print("\n=== Step 4: Exporting splat ===")
    configs = sorted(ns_output_dir.glob("*/splatfacto/*/config.yml"))
    if not configs:
        raise SystemExit(f"No splatfacto config.yml under {ns_output_dir}")
    export_dir = work_dir / "export"
    cmd = ["ns-export", "gaussian-splat", "--load-config", str(configs[-1]),
           "--output-dir", str(export_dir)]
    print(f"Running: {' '.join(cmd)}")
    if subprocess.run(cmd, cwd=str(work_dir)).returncode != 0:
        raise SystemExit("ns-export failed")

    print(f"""
Exported {export_dir / 'splat.ply'}. To use it as a simulator map:
  1. cp {export_dir / 'splat.ply'} data/gsplat/{scene_name}.ply
  2. cd scripts/main && python simulate_object.py --ply {scene_name}
     then align the splat to the sim world in the "GSplat Calibration" panel and
     press Save Calibration (writes data/gsplat/{scene_name}/default.json, which
     the offline renderer requires).
  Do not copy nerfstudio's dataparser_transforms.json into data/gsplat/: the
  renderer would apply it to every splat in that directory.""")


def main():
    parser = argparse.ArgumentParser(description="Video → Gaussian Splat via nerfstudio")
    parser.add_argument("scene_name", help="Name of the scene (used for directory naming and video search)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (auto-searches capture-dir if not set)")
    parser.add_argument("--capture-dir", type=str, default=None, help="Directory to search for video matching scene_name")
    parser.add_argument("--config", type=str, default="iphone16e", help="Camera config name (default: iphone16e)")
    parser.add_argument("--output-dir", type=str, default=str(SCRIPT_DIR / "output"), help="Output directory")
    args = parser.parse_args()

    config_path = SCRIPT_DIR / f"{args.config}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if args.video:
        video_path = args.video
    else:
        capture_dir = Path(args.capture_dir) if args.capture_dir else SCRIPT_DIR / "captures"
        video_path = find_video(args.scene_name, capture_dir)

    print(f"Scene: {args.scene_name}")
    print(f"Video: {video_path}")
    print(f"Config: {config_path}")
    print(f"Output: {args.output_dir}")

    run_pipeline(args.scene_name, video_path, config_path, Path(args.output_dir))


if __name__ == "__main__":
    main()
