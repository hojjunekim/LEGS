"""
Scene configuration for MuJoCo + Viser synchronization.

Defines scene layouts (static props, manipulable objects, grasp info) as Python
dicts. Generates MuJoCo XML strings at runtime so both MuJoCo and Viser read
from the same source — no manual sync needed.

Schema:
    SCENES[name] = {
        "prompt": str,  # optional; language instruction recorded per episode
        "statics": [  # non-movable props (tables, shelves, walls, ...)
            {
                "name": str,              # unique body name
                "mesh": str,              # mesh filename under data/objects/
                "pos":  [x, y, z],
                "rpy":  [r, p, y],        # optional; degrees, initial body orientation
                "scale": float,
                "render_in_image": bool,  # optional; default True. False = visible in Viser + simulated by MuJoCo, but skipped in egocentric image render.
                # Collision is auto-decomposed via CoACD (see _ensure_collision_parts).
                # Tabletop Z for task primitives is measured from the mesh (static_top_offset).
            },
            ...
        ],
        "objects": [  # manipulable (freejoint) items
            {
                "name": str,
                "mesh": str,
                "pos":  [x, y, z],
                "rpy":  [r, p, y],                   # optional; degrees, initial body orientation
                "scale": float,
                "mass": float,
                "friction": [slide, spin, roll],
                "rgba": [r, g, b, a],
                "resting_on": str,                   # optional; static body the object sits on
                "grasp": {                           # optional; wrist pose target vs object center
                    "offset": [x, y, z],
                    "rotation_euler": [rx, ry, rz],  # optional; degrees, XYZ-intrinsic (right arm)
                },
            },
            ...
        ],
        "task_plan": [  # optional; sequential primitive list
            ("approach", "<entity>"),
            ("grasp",    "<object>"),
            ("place",    "<surface>"),
            ...
        ],
        "randomization": {  # optional; sim spawn-pose jitter per reset
            "robot":   {"x": [lo, hi], "y": [lo, hi], "yaw_deg": [lo, hi]},
            "objects": {"x": [lo, hi], "y": [lo, hi], "yaw_deg": [lo, hi]},
        },
        # Per-object override: set entry["randomization"] to a dict (same
        # keys as scene["randomization"]["objects"]) to override scene
        # defaults for that one object, or to None to pin it.
    }

    Randomization defaults match the previous global behavior in
    base_sim.py (robot ±0.05 m xy, no yaw; objects ±0.05 m xy, ±180° yaw).
    `write_scene_xml` writes a sidecar `<xml_stem>_random.json` next to the
    generated XML; base_sim consumes it at reset time.

Usage:
    from scene_config import get_scene, scene_to_xml

    cfg = get_scene("orange_to_plate_turn")
    xml_str = scene_to_xml(cfg)                  # -> MuJoCo
    for obj in cfg["objects"]: ...               # -> Viser
    for stat in cfg["statics"]: ...              # -> Viser
"""

import glob
import json
import os
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# All scene meshes (and their CoACD collision caches) live directly here.
_OBJECTS_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../data/objects"))

# Generated scene XML lives in gr00t's g1 model dir so MuJoCo can resolve the
# robot <include> and "meshdir=meshes" relative paths natively. Override the
# GR00T-WBC location via $GROOT_WBC_DIR, or the full XML path via
# $GROOT_WBC_GENERATED_XML, or `python scene_config.py <scene> --out <path>`.
_GROOT_WBC_DIR = os.environ.get(
    "GROOT_WBC_DIR",
    os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "submodules", "GR00T-WholeBodyControl")),
)
GENERATED_XML_PATH = os.environ.get(
    "GROOT_WBC_GENERATED_XML",
    os.path.join(
        _GROOT_WBC_DIR,
        "gear_sonic/data/robot_model/model_data/g1/generated_scene.xml",
    ),
)

# Bare filename — resolved by MuJoCo relative to the generated XML's directory.
# Assumes `g1_29dof_with_hand.xml` is a sibling of the generated scene file.
_ROBOT_XML = "g1_29dof_with_hand.xml"

# Per-reset spawn-pose randomization. Defaults match the previous global
# behavior baked into base_sim.py — robot xy ±0.05 m (no yaw, because the
# deploy snaps planner_facing_angle_ to 0 on goal-timeout), objects xy
# ±0.05 m and yaw full circle. Each scene can override under "randomization",
# and each object entry can override (or pin via None) under "randomization".
DEFAULT_RANDOMIZATION = {
    "robot": {
        "x":       [-0.05, 0.05],
        "y":       [-0.05, 0.05],
        "yaw_deg": [0.0, 0.0],
    },
    "objects": {
        "x":       [-0.05, 0.05],
        "y":       [-0.05, 0.05],
        "yaw_deg": [-180.0, 180.0],
    },
}

# ---------------------------------------------------------------------------
# Scene definitions — the SINGLE source of truth
# ---------------------------------------------------------------------------
SCENES = {
    # The high_table is centered 1.0 m in front of the robot (near edge at
    # x = 0.625 m), so the robot must walk a couple of steps before it can grasp.
    "orange_to_plate_locomani": {
        "prompt": "place the orange on the plate",
        # Pin the grasping arm: the orange sits right of center (y = -0.1), but
        # the xy + yaw spawn randomization can flip the closer-arm pick.
        "force_arm": "right",
        "episode_timeout": 50.0,
        "statics": [
            {
                "name": "high_table",
                "mesh": "high_table.stl",
                "pos": [1.0, 0.0, 0.365],  # centered on robot y=0; bottom at floor, top at world z = 0.730
                "rpy": [0, 0, -3.14/2],
                "scale": 1.4868,           # → 0.730 m tall (mesh z-extent 0.491)
                "render_in_image": False,    # table is visible in Viser + simulated in MuJoCo, but omitted from egocentric image render to reduce visual clutter and encourage object-centric perception.
            },
        ],
        "objects": [
            {
                "name": "plate",
                "mesh": "plate.stl",  # SAM3D-from-iPhone, already metric (22 cm diameter × 2.5 cm thick)
                "pos": [0.8, 0.1, 0.7424],  # high_table top 0.730 + half-thickness 0.0124
                "scale": 1.0,
                "mass": 0.4,
                "friction": [3.0, 1.0, 1.0],
                "rgba": [0.95, 0.95, 0.95, 1],
                "resting_on": "high_table",
            },
            {
                "name": "orange",
                "mesh": "orange.stl",  # SAM3D-from-iPhone, already metric (~8 cm tall)
                "pos": [0.75, -0.1, 0.770],  # high_table top 0.730 + half-height 0.040
                "scale": 1.0,
                "mass": 0.15,
                "friction": [3.0, 1.0, 0.1],
                "rgba": [1.0, 0.55, 0.0, 1],
                "resting_on": "high_table",
                # Orange is near-spherical: at default threshold (0.05)
                # CoACD collapses to one piece → ~2 vertices touch the
                # table → point-contact chatter. threshold=0.02 is the
                # sweet spot — gives ~4 convex parts in ~4 s without
                # tripping the exponential MCTS branching that smaller
                # thresholds cause on near-convex inputs.
                "coacd_threshold": 0.02,
                "grasp": {
                    "offset": [-0.08, -0.04, 0.02],
                },
            },
        ],
        "task_plan": [
            # First approach drives to the midpoint of orange + plate so the
            # robot stages between them before the hand-biased yaw_align picks
            # the actual grasp target.
            ("approach",       ["orange", "plate"]),
            ("perturb",        {"yaw_deg": 15.0, "duration": 0.3}),
            ("yaw_align",      "orange"),
            ("grasp",          "orange"),
            ("verify_held",    "orange"),
            ("yaw_align",      "plate"),
            # ("drive",          "plate"),
            ("perturb",        {"yaw_deg": 15.0, "duration": 0.3}),
            # ("yaw_align",      "plate"),
            ("place",          "plate"),
            ("verify_placed",  "plate"),
        ],
        # Wider spawn diversity than the default — robot must walk
        # toward the table, so we want the policy to handle off-axis
        # starting poses (xy + small yaw).
        "randomization": {
            "robot": {
                "x":       [-0.20, 0.40],
                "y":       [-0.20, 0.20],
                "yaw_deg": [-15.0, 15.0],
            },
        },
    },
    # Same source-side setup as orange_to_plate_locomani (orange on the
    # high_table in front), but the plate destination is moved onto a
    # lower mid_table on the robot's right. Pick → turn ~90° right →
    # squat (~13 cm drop from high_table top to mid_table top) → place.
    # Object scales match orange_to_plate_locomani.
    "orange_to_plate_turn": {
        "prompt": "pick the orange, turn right, and place it on the plate",
        "episode_timeout": 70.0,
        "statics": [
            {
                "name": "high_table",
                "mesh": "high_table.stl",
                "pos": [1.0, 0.0, 0.365],
                "rpy": [0, 0, -3.14/2],
                "scale": 1.4868,
                "render_in_image": False,
            },
            # Mid_table unrotated (long axis along Y, parallel to robot's
            # right side) extending south of high_table. Mid_table's NE
            # corner meets high_table's SW corner at world (0.625, -0.74),
            # so the two tops form an L with the shared corner pointing
            # outward (mid_table extends south + west from there).
            #   High_table footprint: X∈[0.625, 1.375], Y∈[-0.74, +0.74]
            #   Mid_table half-extents (unrotated, scale 0.653): X=0.27, Y=0.33
            #   Mid_table at (0.195, -1.105) → footprint X∈[-0.075, 0.465],
            #                                                Y∈[-1.435, -0.775]
            # Top z = 0.303 + 0.486 * 0.653 ≈ 0.60.
            {
                "name": "mid_table",
                "mesh": "mid_table.stl",
                "pos": [0.195, -1.105, 0.303],
                "rpy": [0, 0, 0],
                "scale": 0.653,
                "render_in_image": False,
            },
        ],
        "objects": [
            {
                "name": "plate",
                "mesh": "plate.stl",
                "pos": [0.195, -0.89, 0.6124],  # north end of mid_table top; z = 0.600 + 0.0124
                "scale": 1.0,
                "mass": 0.4,
                "friction": [3.0, 1.05, 1.0],
                "rgba": [0.95, 0.95, 0.95, 1],
                "resting_on": "mid_table",
                # Place target — keep tight so the plate stays well inside the
                # mid_table footprint (top is X∈[-0.075,0.465], Y∈[-1.435,-0.775]).
                "randomization": {
                    "x":       [-0.15, 0.15],
                    "y":       [-0.05, 0.05],
                    "yaw_deg": [-180.0, 180.0],
                },
            },
            {
                "name": "orange",
                "mesh": "orange.stl",
                "pos": [0.75, 0.0, 0.770],  # centered on high_table; z = 0.730 + 0.040
                "scale": 1.0,
                "mass": 0.15,
                "friction": [3.0, 1.0, 0.1],
                "rgba": [1.0, 0.55, 0.0, 1],
                "resting_on": "high_table",
                "coacd_threshold": 0.02,
                "grasp": {
                    "offset": [-0.08, -0.04, 0.02],
                },
                # Grasp target — high_table top is big (X∈[0.625,1.375],
                # Y∈[-0.74,+0.74]); ±0.05 m keeps the orange centered enough
                # for the grasp arm-selection logic. Yaw is free (sphere).
                "randomization": {
                    "x":       [-0.08, 0.08],
                    "y":       [-0.05, 0.05],
                    "yaw_deg": [-180.0, 180.0],
                },
            },
        ],
        # Explicit Stepback / Heading / Drive split: motion is identical to
        # ("approach", entity) → ... → ("approach", entity) but the post-grasp
        # back-step is its own primitive instead of being baked into the next
        # Approach's held-object branch. Heading does the rough yaw (no hand
        # bias) for a clean planner path; YawAlign right after picks the
        # active hand and applies the ±10° body bias for grasp/place.
        "task_plan": [
            ("heading",        "orange"),
            ("drive",          "orange"),
            ("yaw_align",      "orange"),
            ("grasp",          "orange"),
            ("verify_held",    "orange"),
            ("stepback",       None),
            ("heading",        "plate"),
            # Pre-drive perturb + paired yaw_align: kicks the body off-axis
            # after heading completed, then the inserted yaw_align recovers
            # toward the plate. The Perturb itself is skip-recorded, so only
            # the recovery action is logged as a label.
            # ("perturb",        {"yaw_deg": 30.0, "duration": 0.3}),
            # ("yaw_align",      "plate"),
            ("drive",          "plate"),
            ("yaw_align",      "plate"),
            ("place",          "plate"),
            ("verify_placed",  "plate"),
        ],
        "randomization": {
            "robot": {
                "x":       [-0.20, 0.40],
                "y":       [-0.20, 0.20],
                "yaw_deg": [-15.0, 15.0],
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Appearance variants (paper Sec. 4.4 / App. D)
# ---------------------------------------------------------------------------
# Render-only re-skins of a recorded task: record episodes once with the base
# scene, then re-render the same episode dir with `--scene <variant>`. Entity
# NAMES are kept ("orange", "plate", "high_table", ...) so recorded
# trajectories replay frame-for-frame; only the rendered appearance and the
# prompt change. Naming is <objects>_<task>[_blue]:
#
#   condition         turn                          locomani
#   default           orange_to_plate_turn          orange_to_plate_locomani
#   scene             orange_to_plate_turn_blue     orange_to_plate_locomani_blue
#   objects           apple_to_box_turn             apple_to_box_locomani
#   objects + scene   apple_to_box_turn_blue        apple_to_box_locomani_blue
#
# scene:   the blue-table capture. The gsplat renderer switches to its splat
#          ("gsplat": "blue"; an explicit --ply still wins), whose real tables
#          replace the default ones (tables are never mesh-rendered over a
#          splat). The mesh-only baseline swaps in the SAM3D meshes of that
#          capture's tables (blue high_table, wood mid_table).
# objects: orange -> apple, plate -> box, with the paper's prompts. The box is
#          drawn at the plate's recorded position but kept world-aligned.
# Assets beyond the core set: `scripts/download_assets.sh aug`.
import copy as _copy

_BLUE_TABLES = {"high_table": "high_table_blue_raw.stl", "mid_table": "mid_table_wood_raw.stl"}
_APPLE_BOX = {
    "orange": {"mesh": "apple.stl"},
    "plate":  {"mesh": "box.stl", "lock_world_orientation": True},
}


def _variant(name, base, prompt=None, objects=None, tables=None, gsplat=None):
    cfg = _copy.deepcopy(SCENES[base])
    cfg["render_only"] = True
    cfg["base_scene"] = base
    if prompt:
        cfg["prompt"] = prompt
    if gsplat:
        cfg["gsplat"] = gsplat
    for o in cfg["objects"]:
        o.update((objects or {}).get(o["name"], {}))
    for st in cfg["statics"]:
        st["mesh"] = (tables or {}).get(st["name"], st["mesh"])
    SCENES[name] = cfg


for _task, _prompt in (
    ("turn",     "pick the apple, turn right, and put it in the box"),
    ("locomani", "put the apple in the box"),
):
    _base = f"orange_to_plate_{_task}"
    _variant(f"{_base}_blue", _base, tables=_BLUE_TABLES, gsplat="blue")
    _variant(f"apple_to_box_{_task}", _base, prompt=_prompt, objects=_APPLE_BOX)
    _variant(f"apple_to_box_{_task}_blue", _base, prompt=_prompt, objects=_APPLE_BOX,
             tables=_BLUE_TABLES, gsplat="blue")
del _variant, _task, _prompt, _base

# Large assets are not stored in git: scripts/download_assets.sh fetches them,
# and scripts/assets.sha256 lists each one as `<group> <sha256> <bytes> <path>`.
_REPO_ROOT = os.path.realpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_ASSET_MANIFEST = os.path.join(_REPO_ROOT, "scripts", "assets.sha256")


def _repo_rel(path):
    p = os.path.realpath(path)
    return os.path.relpath(p, _REPO_ROOT) if p.startswith(_REPO_ROOT + os.sep) else None


def asset_group(path):
    """Download group that provides `path` (per scripts/assets.sha256), or None."""
    rel = _repo_rel(path)
    if rel is None:
        return None
    try:
        with open(_ASSET_MANIFEST) as f:
            for line in f:
                cols = line.split()
                if len(cols) == 4 and not cols[0].startswith("#") and cols[3] == rel:
                    return cols[0]
    except OSError:
        pass
    return None


def missing_asset_message(path):
    """One-line hint for a missing asset, naming the download group if any."""
    rel = _repo_rel(path)
    group = asset_group(path)
    if group:
        return f"{rel} is missing. Fetch it with: scripts/download_assets.sh {group}"
    if rel is None:
        return f"{path} not found."
    return f"{rel} is missing and is not a downloadable asset (scripts/assets.sha256)."


def require_core_assets():
    """Exit with a hint unless every file of the `core` download group is present.

    The simulator, the scene build and the renderers all need them (robot and
    object meshes, the shipped collision parts, the flightroom splat).
    """
    try:
        with open(_ASSET_MANIFEST) as f:
            rels = [c[3] for c in (line.split() for line in f) if len(c) == 4 and c[0] == "core"]
    except OSError:
        return
    missing = [r for r in rels if not os.path.isfile(os.path.join(_REPO_ROOT, r))]
    if missing:
        raise SystemExit(f"{len(missing)} of the {len(rels)} core asset files are missing "
                         f"(e.g. {missing[0]}). Fetch them with: scripts/download_assets.sh core")


DEFAULT_SCENE = "orange_to_plate_locomani"


def _fmt(vals):
    """Format a list of numbers as a space-separated string."""
    return " ".join(str(v) for v in vals)


def _s3(v):
    """Format a scalar as a 3-component scale string ('v v v')."""
    return f"{v} {v} {v}"


def _geom_pos_attr(entry):
    """Return ' pos="..."' for a geom that re-centers the body origin onto a
    mesh feature (e.g. mug cylinder center). The cfg's `mesh_offset` is the
    feature's position in raw, unscaled mesh coordinates; geom pos is its
    negation, scaled to body-local units. Empty string when not configured.
    """
    off = entry.get("mesh_offset")
    if off is None:
        return ""
    s = entry.get("scale", 1.0)
    return f' pos="{-off[0]*s} {-off[1]*s} {-off[2]*s}"'


def _static_xml(s):
    """Emit a <body> block for a static prop.

    Collision comes from CoACD-decomposed parts (auto-generated by
    _ensure_collision_parts). If parts are missing (e.g. coacd not installed),
    the body is visual-only.
    """
    name = s["name"]
    body_attrs = f'name="{name}" pos="{_fmt(s["pos"])}"'
    if "rpy" in s:
        body_attrs += f' euler="{_fmt(s["rpy"])}"'
    if "rgba" in s:
        appearance = f'rgba="{_fmt(s["rgba"])}"'
    else:
        appearance = 'material="prop_mat"'
    visual = (
        f'<geom name="{name}_visual" type="mesh" mesh="{name}_mesh" '
        f'{appearance} contype="0" conaffinity="0" group="1"/>'
    )
    col_parts = _find_collision_parts(s)
    if col_parts:
        col_geoms = [
            f'<geom name="{name}_col_{i}" type="mesh" mesh="{name}_col_{i}_mesh" '
            f'material="prop_mat" rgba="0.6 0.4 0.2 0.3" group="3"/>'
            for i in range(len(col_parts))
        ]
        collision = "\n      ".join(col_geoms)
        return (
            f'<body {body_attrs}>\n'
            f'      {visual}\n'
            f'      {collision}\n'
            f'    </body>'
        )
    return (
        f'<body {body_attrs}>\n'
        f'      {visual}\n'
        f'    </body>'
    )


def _collision_dir(entry):
    """Return the per-object collision directory under data/objects/."""
    return os.path.join(_OBJECTS_DIR, entry["name"])


def _find_collision_parts(entry):
    """Return sorted list of collision STL paths for an entry (e.g. <name>/<name>_col_0.stl, ...)."""
    pattern = os.path.join(_collision_dir(entry), f"{entry['name']}_col_*.stl")
    return sorted(glob.glob(pattern))


def _ensure_collision_parts(entry, threshold=0.05, max_faces_per_part=None,
                            mcts_iterations=None, mcts_max_depth=None,
                            preprocess_resolution=None, max_convex_hull=None):
    """Auto-decompose a mesh via CoACD if no collision parts exist yet.

    Loads <name>.stl from data/objects/, runs convex decomposition,
    and saves parts into data/objects/<name>/<name>_col_*.stl.
    Skips silently if parts already exist or trimesh/coacd are not installed.

    `max_faces_per_part`: if set, each CoACD part is post-decimated (Open3D
    quadric) to this face cap. CoACD preserves input topology when it can't
    decompose further, so a "convex" piece for a near-spherical object may
    still have tens of thousands of triangles — that creates redundant
    contact candidates and chatters MuJoCo's solver. Capping faces keeps
    contacts numerically clean.

    `mcts_iterations`, `mcts_max_depth`, `preprocess_resolution`: optional
    CoACD knobs. Defaults (None) use CoACD's own defaults (150, 3, 50). For
    near-convex inputs at low threshold, CoACD's MCTS branching explodes
    exponentially because no candidate cut clears the concavity threshold
    — capping these (e.g. 20 / 2 / 30) bounds runtime to seconds at the
    cost of decomposition quality.

    `max_convex_hull`: hard upper bound on the number of pieces returned.
    Default (None) → CoACD's -1 (unlimited), which can produce thousands of
    micro-pieces on near-spherical inputs at low threshold. Cap to ~8–16
    for round objects so MuJoCo doesn't get an avalanche of collision geoms.
    """
    if _find_collision_parts(entry):
        return

    name = entry["name"]
    stl_path = os.path.join(_OBJECTS_DIR, f"{name}.stl")
    if not os.path.exists(stl_path):
        print(f"[scene_config] No STL found for '{name}' at {stl_path}, skipping decomposition")
        return

    try:
        import trimesh
        import coacd
    except ImportError as e:
        print(f"[scene_config] {e} — skipping auto-decomposition for '{name}'. "
              "Install with: pip install coacd trimesh")
        return

    col_dir = _collision_dir(entry)
    os.makedirs(col_dir, exist_ok=True)

    coacd_kwargs = {"threshold": threshold}
    if mcts_iterations is not None:
        coacd_kwargs["mcts_iterations"] = int(mcts_iterations)
    if mcts_max_depth is not None:
        coacd_kwargs["mcts_max_depth"] = int(mcts_max_depth)
    if preprocess_resolution is not None:
        coacd_kwargs["preprocess_resolution"] = int(preprocess_resolution)
    if max_convex_hull is not None:
        coacd_kwargs["max_convex_hull"] = int(max_convex_hull)

    extras = [f"{k}={v}" for k, v in coacd_kwargs.items() if k != "threshold"]
    if max_faces_per_part:
        extras.append(f"max_faces_per_part={max_faces_per_part}")
    extras_str = (", " + ", ".join(extras)) if extras else ""
    print(f"[scene_config] Auto-decomposing '{name}' (threshold={threshold}{extras_str})...")
    mesh = trimesh.load(stl_path, force="mesh")
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(coacd_mesh, **coacd_kwargs)

    if max_faces_per_part:
        try:
            import open3d as o3d
        except ImportError as e:
            print(f"[scene_config] {e} — max_faces_per_part requires open3d; skipping decimation")
            max_faces_per_part = None

    for i, (verts, faces) in enumerate(parts):
        if max_faces_per_part and len(faces) > max_faces_per_part:
            o3 = o3d.geometry.TriangleMesh()
            o3.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
            o3.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
            dec = o3.simplify_quadric_decimation(target_number_of_triangles=max_faces_per_part)
            verts = np.asarray(dec.vertices)
            faces = np.asarray(dec.triangles)
        col_path = os.path.join(col_dir, f"{name}_col_{i}.stl")
        trimesh.Trimesh(verts, faces).export(col_path, file_type="stl")

    print(f"[scene_config] '{name}' -> {len(parts)} convex parts saved to {col_dir}/")


def _object_xml(o):
    """Emit a <body> block for a manipulable (freejoint) object.

    If CoACD collision parts exist (<name>_col_*.stl), uses a separate visual
    geom (no collision) and per-part collision geoms (invisible). Otherwise
    falls back to a single geom for both visual and collision.
    """
    name = o["name"]
    col_parts = _find_collision_parts(o)
    body_attrs = f'name="{name}" pos="{_fmt(o["pos"])}"'
    if "rpy" in o:
        body_attrs += f' euler="{_fmt(o["rpy"])}"'

    pos_attr = _geom_pos_attr(o)

    if col_parts:
        # Visual-only geom (no collision, no inertia — without mass="0"
        # MuJoCo falls back to density=1000 kg/m³ × mesh volume, which can
        # silently make the body 1000× heavier than the configured `mass`).
        visual = (
            f'<geom name="{name}_visual" type="mesh" mesh="{name}_mesh" '
            f'contype="0" conaffinity="0" mass="0" '
            f'rgba="{_fmt(o["rgba"])}" group="1"{pos_attr}/>'
        )
        # Per-part collision geoms (invisible, share mass equally)
        n = len(col_parts)
        part_mass = o["mass"] / n
        col_geoms = []
        for i in range(n):
            col_geoms.append(
                f'<geom name="{name}_col_{i}" type="mesh" mesh="{name}_col_{i}_mesh" '
                f'mass="{part_mass:.4f}" friction="{_fmt(o["friction"])}" '
                f'rgba="{_fmt(o["rgba"])}" condim="4" group="3"{pos_attr}/>'
            )
        col_block = "\n      ".join(col_geoms)
        return (
            f'<body {body_attrs}>\n'
            f'      <freejoint name="{name}_joint"/>\n'
            f'      {visual}\n'
            f'      {col_block}\n'
            f'    </body>'
        )
    else:
        # Fallback: single geom for visual + collision (convex hull)
        return (
            f'<body {body_attrs}>\n'
            f'      <freejoint name="{name}_joint"/>\n'
            f'      <geom name="{name}_geom" type="mesh" mesh="{name}_mesh" '
            f'mass="{o["mass"]}" friction="{_fmt(o["friction"])}" '
            f'rgba="{_fmt(o["rgba"])}" condim="4"{pos_attr}/>\n'
            f'    </body>'
        )


def scene_to_xml(cfg):
    """Generate a MuJoCo XML string from a scene config dict."""
    if cfg.get("render_only"):
        raise ValueError(
            "Appearance variants are render-only: record with the base scene "
            f"'{cfg['base_scene']}', then re-render that episode dir with --scene <variant>."
        )
    statics = cfg.get("statics", [])
    objects = cfg.get("objects", [])

    # Auto-decompose all meshes (statics + objects) that lack collision parts.
    # Per-entry `coacd_threshold` override controls decomposition coarseness;
    # higher threshold → fewer, larger parts (use for flat shapes like plates).
    for entry in list(statics) + list(objects):
        _ensure_collision_parts(
            entry,
            threshold=entry.get("coacd_threshold", 0.05),
            max_faces_per_part=entry.get("max_faces_per_part"),
            mcts_iterations=entry.get("coacd_mcts_iterations"),
            mcts_max_depth=entry.get("coacd_mcts_max_depth"),
            preprocess_resolution=entry.get("coacd_preprocess_resolution"),
            max_convex_hull=entry.get("coacd_max_convex_hull"),
        )

    mesh_assets = []
    for entry in list(statics) + list(objects):
        path = os.path.join(_OBJECTS_DIR, entry["mesh"])
        mesh_assets.append(
            f'<mesh name="{entry["name"]}_mesh" file="{path}" scale="{_s3(entry["scale"])}"/>'
        )
    # Add collision part meshes for all entries with CoACD decomposition
    for entry in list(statics) + list(objects):
        for col_path in _find_collision_parts(entry):
            i = col_path.rsplit("_", 1)[-1].replace(".stl", "")
            mesh_assets.append(
                f'<mesh name="{entry["name"]}_col_{i}_mesh" file="{col_path}" scale="{_s3(entry["scale"])}"/>'
            )
    assets_block = "\n    ".join(mesh_assets)

    bodies = [_static_xml(s) for s in statics] + [_object_xml(o) for o in objects]
    bodies_block = "\n\n    ".join(bodies)

    return f"""\
<!-- GENERATED from scene_config.py — do not hand-edit; changes will be overwritten. -->
<mujoco model="g1_scene">
  <include file="{_ROBOT_XML}"/>

  <statistic center="0 0 0.5" extent="2.0"/>
  <option cone="elliptic" impratio="10" noslip_iterations="5"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
    <material name="prop_mat" rgba="0.6 0.4 0.2 1"/>

    {assets_block}
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <site name="com_marker" pos="0.1 0 0" size="0.05" rgba="1 0 0 1" type="sphere"/>

    {bodies_block}
  </worldbody>

  <default>
    <geom friction="2.0"/>
  </default>
</mujoco>"""


def resolve_randomization(cfg):
    """Return a fully-resolved randomization spec for a scene.

    Output:
        {
            "robot":   {"x": [..], "y": [..], "yaw_deg": [..]},
            "objects": {<object_name>: {"x": [..], "y": [..], "yaw_deg": [..]}
                        | None},   # None = pinned, do not jitter
        }

    Resolution order (most → least specific):
        per-object override > scene["randomization"]["objects"] > DEFAULT
    Robot:
        scene["randomization"]["robot"] > DEFAULT
    """
    scene_rand = cfg.get("randomization") or {}
    robot = {**DEFAULT_RANDOMIZATION["robot"], **(scene_rand.get("robot") or {})}
    obj_default = {**DEFAULT_RANDOMIZATION["objects"], **(scene_rand.get("objects") or {})}

    object_specs = {}
    for o in cfg.get("objects", []):
        if "randomization" in o:
            override = o["randomization"]
            if override is None:
                object_specs[o["name"]] = None  # pinned
                continue
            object_specs[o["name"]] = {**obj_default, **override}
        else:
            object_specs[o["name"]] = obj_default

    return {"robot": robot, "objects": object_specs}


def _randomization_sidecar_path(xml_path):
    """Sidecar JSON path that lives next to a generated scene XML."""
    stem, _ = os.path.splitext(xml_path)
    return stem + "_random.json"


def write_scene_xml(cfg, output_path):
    """Write scene XML + randomization sidecar JSON.

    Sidecar lives at `<output_path stem>_random.json` so base_sim.py can
    pick up scene-specific spawn jitter without knowing the scene name.
    """
    xml = scene_to_xml(cfg)
    with open(output_path, "w") as f:
        f.write(xml)

    sidecar_path = _randomization_sidecar_path(output_path)
    with open(sidecar_path, "w") as f:
        json.dump(resolve_randomization(cfg), f, indent=2)

    return output_path


def get_scene(name=None):
    """Get a scene config by name. Defaults to DEFAULT_SCENE."""
    name = name or DEFAULT_SCENE
    if name not in SCENES:
        raise ValueError(f"Unknown scene '{name}'. Available: {list(SCENES.keys())}")
    return SCENES[name]


def find_static(scene_cfg, name):
    """Return the static cfg with matching name, or None."""
    for s in scene_cfg.get("statics", []):
        if s["name"] == name:
            return s
    return None


def find_object(scene_cfg, name):
    """Return the object cfg with matching name, or None."""
    for o in scene_cfg.get("objects", []):
        if o["name"] == name:
            return o
    return None


def find_entity(scene_cfg, name):
    """Return the cfg with matching name from either statics or objects, or None."""
    return find_static(scene_cfg, name) or find_object(scene_cfg, name)


_MESH_MAX_Z_CACHE = {}


def _mesh_max_z(entry):
    """Return the max-Z extent of an entry's mesh in body frame (cached).

    Cache key is the absolute resolved path under data/objects/.
    """
    path = os.path.join(_OBJECTS_DIR, entry["mesh"])
    if path in _MESH_MAX_Z_CACHE:
        return _MESH_MAX_Z_CACHE[path]
    try:
        import trimesh
        mesh = trimesh.load(path, force="mesh")
        max_z = float(mesh.bounds[1, 2])
    except Exception as e:
        print(f"[scene_config] Failed to measure '{entry['mesh']}' max-Z: {e}")
        max_z = 0.0
    _MESH_MAX_Z_CACHE[path] = max_z
    return max_z


def static_top_offset(static_cfg):
    """Body-frame Z offset of a static prop's top surface.

    Measured directly from the mesh's max-Z (body frame), multiplied by the
    entry's `scale`. For a static at world pose `(x,y,z)`, the top surface
    sits at `z + static_top_offset(cfg)`. Cached per resolved mesh path.
    """
    return _mesh_max_z(static_cfg) * static_cfg.get("scale", 1.0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dump a scene as MuJoCo XML.")
    parser.add_argument("scene", nargs="?", default=DEFAULT_SCENE,
                        help=f"Scene name. Available: {list(SCENES.keys())}")
    parser.add_argument("--out", default=GENERATED_XML_PATH,
                        help=f"Output path (default: {GENERATED_XML_PATH})")
    args = parser.parse_args()

    cfg = get_scene(args.scene)
    require_core_assets()
    path = write_scene_xml(cfg, args.out)
    print(f"Wrote scene '{args.scene}' -> {path}")
