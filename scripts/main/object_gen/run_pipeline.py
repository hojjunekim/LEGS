"""End-to-end orchestrator for object_gen: iPhone JPG -> sim-ready mesh.

Runs sam_mask -> sam3d_mesh -> color_refine_ccm -> render_test in sequence.
Each step is also importable/runnable on its own — use `--steps` to skip
ahead when iterating.

The `--name` arg controls the output directory:
    <data-root>/<name>/

Default <data-root> is `<repo>/data/objects/sam3d_runs`.

`--source DIR` takes the inputs from a source dir instead, e.g.
data/objects/sources/<name>/ (iphone.jpg, click.json, mask.png, mesh_raw.glb,
ccm.json, config.json). A step reads what an earlier step wrote in the same
run, else the source dir; outputs always go to <data-root>/<name>/, so the
sources are never overwritten. mesh_raw.glb is fetched by
`bash scripts/download_assets.sh sources`.

Usage:
    # full pipeline
    python run_pipeline.py --image mug.jpg --click 612,840 --name mug

    # just remask, keeping existing mesh
    python run_pipeline.py --image mug.jpg --click 700,820 --name mug \\
        --steps mask

    # rerun everything downstream of an existing mask
    python run_pipeline.py --name mug --steps mesh,refine,render

    # rebuild a shipped object from its sources (refit, or bake the stored fit)
    python run_pipeline.py --source data/objects/sources/orange --steps refine,render
    python run_pipeline.py --source data/objects/sources/plate --steps refine,render \\
        --apply-ccm data/objects/sources/plate/ccm.json
"""
from __future__ import annotations

# cv2 must load before trimesh (imported below through sam3d_mesh). trimesh's compiled pip dependencies load the
# system libstdc++, and on hosts where that is older than the env's (Ubuntu 22.04), cv2 then fails with
# "CXXABI_1.3.15 not found". Loaded first, cv2 brings in the env's libstdc++ for all.
import cv2  # noqa: F401
import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sam_mask  # noqa: E402
import sam3d_mesh  # noqa: E402
import color_refine  # noqa: E402
import color_refine_ccm  # noqa: E402
import render_test  # noqa: E402


_DEFAULT_DATA_ROOT = _HERE.parents[2] / "data" / "objects" / "sam3d_runs"
_VALID_STEPS = ("mask", "mesh", "refine", "render")


def _parse_click(s: str):
    parts = s.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--click expects 'x,y', got {s!r}")
    return int(parts[0]), int(parts[1])


def _parse_steps(s: str) -> list[str]:
    steps = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in steps if x not in _VALID_STEPS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown steps {bad}; valid: {list(_VALID_STEPS)}"
        )
    return steps


def _load_clicks(path: Path):
    """click.json -> (positive, negative). Also reads the older {x, y} form."""
    meta = json.loads(path.read_text())
    if "positive" in meta:
        return ([tuple(c) for c in meta["positive"]],
                [tuple(c) for c in meta.get("negative", [])])
    return [(meta["x"], meta["y"])], []


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=None,
                   help="object id; determines out_dir (default: --source dir name)")
    p.add_argument("--source", default=None,
                   help="dir with iphone.jpg, click.json, mask.png, mesh_raw.glb "
                        "(e.g. data/objects/sources/orange)")
    p.add_argument("--image", default=None, help="iPhone JPG (required for 'mask')")
    p.add_argument("--click", type=_parse_click, action="append", default=None,
                   help="positive point 'x,y' (repeatable; required for 'mask')")
    p.add_argument("--neg-click", type=_parse_click, action="append", default=[],
                   help="negative point 'x,y' (repeatable; SAM excludes this region)")
    p.add_argument("--data-root", default=str(_DEFAULT_DATA_ROOT),
                   help=f"output root dir (default: {_DEFAULT_DATA_ROOT})")
    p.add_argument("--steps", type=_parse_steps,
                   default=list(_VALID_STEPS),
                   help=f"comma-separated subset of {list(_VALID_STEPS)}")
    p.add_argument("--no-refine", action="store_true",
                   help="skip color refinement (alias for omitting 'refine')")
    p.add_argument("--refine-method", choices=["ccm", "reinhard"], default="ccm",
                   help="ccm (default; post-render per-channel scale in linear RGB, "
                        "see color_refine_ccm.py) or reinhard (vertex-space Lab "
                        "mean/std match, see color_refine.py)")
    p.add_argument("--apply-ccm", default=None, metavar="CCM_JSON",
                   help="ccm only: bake this stored ccm.json instead of refitting")
    p.add_argument("--sam3d-root", default=None,
                   help="path to submodules/sam-3d-objects (default: auto)")
    p.add_argument("--fov", type=float, default=0.74)
    p.add_argument("--cam-distance", type=float, default=0.4)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--seed", type=int, default=None,
                   help="SAM3D seed (default: --source config.json, else 42)")
    p.add_argument("--real-height", type=float, default=None,
                   help="meters; scale mesh.glb so its Y-extent (GLB up) matches this — "
                        "SAM3D outputs are ~unit-cube, sim needs metric "
                        "(default: --source config.json)")
    args = p.parse_args()

    steps = list(args.steps)
    if args.no_refine and "refine" in steps:
        steps.remove("refine")
    if args.apply_ccm and args.refine_method != "ccm":
        p.error("--apply-ccm requires --refine-method ccm")

    src = Path(args.source) if args.source else None
    if src is not None and not src.is_dir():
        p.error(f"--source {src} is not a directory")
    name = args.name or (src.name if src is not None else None)
    if not name:
        p.error("--name is required unless --source is given")
    config = {}
    if src is not None and (src / "config.json").exists():
        config = json.loads((src / "config.json").read_text())
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    real_height = args.real_height if args.real_height is not None else config.get("real_height_m")

    out_dir = Path(args.data_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "iphone": out_dir / "iphone.jpg",
        "mask": out_dir / "mask.png",
        "mesh": out_dir / "mesh.glb",
        "mesh_refined": out_dir / "mesh_refined.glb",
        "mesh_refined_ccm": out_dir / "mesh_refined_ccm.glb",
    }
    # Inputs: what an earlier step writes in this run, else --source, else out_dir.
    # sam_mask re-encodes iphone.jpg, so the source photo is preferred over that copy.
    inputs = dict(paths)
    if src is not None:
        inputs["iphone"] = src / "iphone.jpg"
        if "mask" not in steps:
            inputs["mask"] = src / "mask.png"
        if "mesh" not in steps:
            inputs["mesh"] = src / "mesh_raw.glb"

    def _require_mesh():
        if inputs["mesh"].exists():
            return
        if src is not None and "mesh" not in steps:
            sys.exit(f"missing {inputs['mesh']} — fetch it with "
                     "`bash scripts/download_assets.sh sources`, or add 'mesh' to --steps")
        sys.exit(f"missing {inputs['mesh']} — run 'mesh' first")

    if "mask" in steps:
        image = args.image or (str(inputs["iphone"]) if src is not None else None)
        clicks, neg_clicks = args.click, args.neg_click
        if not clicks and src is not None and (src / "click.json").exists():
            clicks, neg_clicks = _load_clicks(src / "click.json")
        if not image or not clicks:
            sys.exit("--image and --click (or --source with click.json) are required "
                     "when step 'mask' is run")
        print(f"[1/4] mask  -> {out_dir}")
        sam_mask.run(image, clicks, str(out_dir), negative=neg_clicks)

    if "mesh" in steps:
        if not inputs["iphone"].exists() or not inputs["mask"].exists():
            sys.exit(f"missing {inputs['iphone']} or {inputs['mask']} — run 'mask' first")
        print(f"[2/4] mesh  -> {out_dir}")
        sam3d_mesh.run(
            str(inputs["iphone"]), str(inputs["mask"]), str(out_dir),
            args.sam3d_root or str(_HERE.parents[2] / "submodules" / "sam-3d-objects"),
            seed=seed,
            real_height_m=real_height,
        )

    if "refine" in steps:
        _require_mesh()
        print(f"[3/4] refine ({args.refine_method}) -> {out_dir}")
        if args.refine_method == "reinhard":
            color_refine.run(
                str(inputs["iphone"]), str(inputs["mask"]), str(inputs["mesh"]), str(out_dir),
            )
        elif args.apply_ccm:
            color_refine_ccm.apply(str(inputs["mesh"]), args.apply_ccm, str(out_dir))
        else:
            color_refine_ccm.run(
                str(inputs["iphone"]), str(inputs["mask"]), str(inputs["mesh"]), str(out_dir),
                width=args.width, height=args.height,
                fov=args.fov, cam_distance=args.cam_distance,
            )

    if "render" in steps:
        print(f"[4/4] render -> {out_dir}")
        # Pick the refined mesh matching the method we ran; fall back to whichever exists.
        refined_pref = (
            paths["mesh_refined_ccm"] if args.refine_method == "ccm"
            else paths["mesh_refined"]
        )
        refined_for_render = (
            refined_pref if refined_pref.exists()
            else (paths["mesh_refined"] if paths["mesh_refined"].exists()
                  else paths["mesh_refined_ccm"] if paths["mesh_refined_ccm"].exists()
                  else None)
        )
        render_test.run(
            str(out_dir),
            mesh_path=str(inputs["mesh"]) if inputs["mesh"].exists() else None,
            mesh_refined_path=str(refined_for_render) if refined_for_render else None,
            width=args.width, height=args.height,
            fov=args.fov, cam_distance=args.cam_distance,
        )

    print(f"\nDone. Artifacts in {out_dir}/")
    final_refined = (
        paths["mesh_refined_ccm"] if args.refine_method == "ccm" and paths["mesh_refined_ccm"].exists()
        else paths["mesh_refined"] if paths["mesh_refined"].exists()
        else None
    )
    if final_refined is not None:
        print(f"  -> next: cp {final_refined} {_HERE.parents[2] / 'data' / 'objects' / name}.glb")


if __name__ == "__main__":
    main()
