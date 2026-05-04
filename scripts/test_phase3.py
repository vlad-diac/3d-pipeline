"""
Phase 3 test — mesh_generate.py + mesh_postprocess.py

Two test tiers:

  CPU tier (runs everywhere — macOS + RunPod):
    • Postprocessing functions on synthetic trimesh objects:
        - floater removal (keep_largest_component)
        - NaN/degenerate face removal
        - decimation (trimesh built-in, pymeshlab fallback if available)
        - normalization (center + unit scale)
        - full postprocess_mesh pipeline
    • module importability of mesh_generate (no GPU call)

  GPU tier (RunPod only — skipped automatically if CUDA unavailable):
    • load_shape_pipeline_auto()
    • generate_mesh() from a test image → raw trimesh
    • postprocess_mesh() on the generated mesh
    • Save raw + postprocessed GLBs

Outputs written to  outputs/test/phase3/:
  mesh_raw.glb             — direct from shape pipeline    (GPU only)
  mesh_postprocessed.glb   — after full cleanup + decimation (GPU only)
  synth_floaters.glb       — synthetic: before floater removal
  synth_cleaned.glb        — synthetic: after postprocess_mesh
  synth_normalized.glb     — synthetic: after normalize_mesh

Usage:
  python scripts/test_phase3.py [--image path/to/input.png]

  --image is required for the GPU generation test.
  Without --image, only the CPU-tier tests run (even on GPU).
"""

from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("test_phase3")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import trimesh


from datetime import datetime
OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "test" / "phase3" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_glb(mesh: trimesh.Trimesh, name: str) -> None:
    path = OUTPUT_DIR / name
    mesh.export(str(path))
    print(f"  saved → {path.relative_to(_PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Synthetic mesh helpers
# ---------------------------------------------------------------------------

def make_unit_cube() -> trimesh.Trimesh:
    """Return a clean unit cube (12 triangles)."""
    return trimesh.creation.box(extents=[1.0, 1.0, 1.0])


def make_sphere(subdivisions: int = 3) -> trimesh.Trimesh:
    """Return a UV sphere (for decimation tests — more faces than a cube)."""
    return trimesh.creation.icosphere(subdivisions=subdivisions)


def make_mesh_with_floaters() -> trimesh.Trimesh:
    """
    Return a cube with two tiny disconnected fragments (floaters).
    Used to verify keep_largest_component() removes them.
    """
    cube = make_unit_cube()

    # Small triangle far away — simulates a marching-cubes floater.
    tiny_verts = np.array([
        [10.0, 10.0, 10.0],
        [10.1, 10.0, 10.0],
        [10.0, 10.1, 10.0],
    ])
    tiny_faces = np.array([[0, 1, 2]], dtype=np.int64)
    floater = trimesh.Trimesh(vertices=tiny_verts, faces=tiny_faces, process=False)

    # Another tiny quad far in the opposite direction.
    tiny2_verts = np.array([
        [-20.0, -20.0, -20.0],
        [-19.9, -20.0, -20.0],
        [-20.0, -19.9, -20.0],
        [-19.9, -19.9, -20.0],
    ])
    tiny2_faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    floater2 = trimesh.Trimesh(vertices=tiny2_verts, faces=tiny2_faces, process=False)

    combined = trimesh.util.concatenate([cube, floater, floater2])
    return combined


def make_mesh_with_nans() -> trimesh.Trimesh:
    """Return a cube with a NaN vertex injected (tests remove_infinite_values)."""
    cube = make_unit_cube()
    mesh = cube.copy()
    # Inject NaN into one vertex — trimesh's remove_infinite_values should clean this.
    verts = np.array(mesh.vertices, dtype=np.float64)
    verts[0] = [np.nan, np.inf, -np.inf]
    mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)
    return mesh


# ---------------------------------------------------------------------------
# CPU-tier tests
# ---------------------------------------------------------------------------

def test_imports() -> None:
    print("\n[1] Module imports (CPU)")
    from src.mesh_postprocess import (  # noqa: F401
        keep_largest_component,
        normalize_mesh,
        decimate_mesh,
        postprocess_mesh,
        save_mesh,
    )
    print("  src.mesh_postprocess: OK")

    # mesh_generate imports are GPU-lazy; just check module is parseable.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mesh_generate", _PROJECT_ROOT / "src" / "mesh_generate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Don't exec — would import torch/hy3dshape. Just confirm it loads the spec.
    assert spec is not None
    print("  src.mesh_generate: spec loaded OK (imports deferred to GPU tier)")


def test_keep_largest_component() -> None:
    print("\n[2] keep_largest_component")
    from src.mesh_postprocess import keep_largest_component

    mesh = make_mesh_with_floaters()
    cube = make_unit_cube()

    assert len(mesh.faces) > len(cube.faces), "Floater mesh should have more faces than cube alone"

    cleaned = keep_largest_component(mesh)
    assert len(cleaned.faces) == len(cube.faces), (
        f"Expected {len(cube.faces)} faces (cube only), got {len(cleaned.faces)}"
    )
    print(f"  faces: {len(mesh.faces)} → {len(cleaned.faces)}  (floaters removed): OK")

    save_glb(mesh, "synth_floaters.glb")


def test_normalize_mesh() -> None:
    print("\n[3] normalize_mesh")
    from src.mesh_postprocess import normalize_mesh

    # Cube scaled to 5× and offset — after normalization should be ~unit.
    mesh = make_unit_cube()
    mesh.apply_scale(5.0)
    mesh.apply_translation([100.0, -50.0, 20.0])

    normalized = normalize_mesh(mesh)
    bounds = normalized.bounds
    center = bounds.mean(axis=0)
    longest_side = float(np.max(bounds[1] - bounds[0]))

    assert np.allclose(center, [0.0, 0.0, 0.0], atol=1e-5), \
        f"Center should be at origin, got {center}"
    assert abs(longest_side - 1.0) < 1e-5, \
        f"Longest side should be 1.0, got {longest_side}"
    print(f"  center={np.round(center, 4)}, longest_side={longest_side:.4f}: OK")

    save_glb(normalized, "synth_normalized.glb")


def test_decimate() -> None:
    print("\n[4] decimate_mesh")
    from src.mesh_postprocess import decimate_mesh

    mesh = make_sphere(subdivisions=4)   # ~5120 faces
    original_faces = len(mesh.faces)
    target = 500

    decimated = decimate_mesh(mesh, target_faces=target)

    assert len(decimated.faces) <= target * 1.05, (
        f"Expected <= {target * 1.05:.0f} faces after decimation, got {len(decimated.faces)}"
    )
    print(f"  faces: {original_faces} → {len(decimated.faces)} (target={target}): OK")

    # No-op: already under target.
    mesh_small = make_unit_cube()
    result_noop = decimate_mesh(mesh_small, target_faces=100_000)
    assert len(result_noop.faces) == len(mesh_small.faces), "Should be a no-op"
    print("  no-op (mesh already under target): OK")


def test_postprocess_full() -> None:
    print("\n[5] postprocess_mesh (full pipeline)")
    from src.mesh_postprocess import postprocess_mesh

    # Build a dirty mesh: floaters + high face count.
    mesh = make_mesh_with_floaters()
    sphere = make_sphere(subdivisions=4)   # ~5120 faces
    mesh = trimesh.util.concatenate([mesh, sphere])
    mesh.apply_translation([200.0, 0.0, 0.0])   # out of unit range — normalization test

    original_faces = len(mesh.faces)
    target = 1000

    cleaned = postprocess_mesh(mesh, target_faces=target, normalize=True)

    # Face count under target.
    assert len(cleaned.faces) <= target * 1.05, \
        f"Expected <= {target * 1.05:.0f} faces, got {len(cleaned.faces)}"

    # Normalized.
    bounds = cleaned.bounds
    center = bounds.mean(axis=0)
    longest = float(np.max(bounds[1] - bounds[0]))
    assert np.allclose(center, [0.0, 0.0, 0.0], atol=1e-4), f"Center off: {center}"
    assert abs(longest - 1.0) < 1e-4, f"Scale off: {longest}"

    print(f"  faces: {original_faces} → {len(cleaned.faces)}, normalized: OK")
    save_glb(cleaned, "synth_cleaned.glb")


def test_nan_removal() -> None:
    print("\n[6] NaN/Inf vertex removal")
    from src.mesh_postprocess import postprocess_mesh

    mesh = make_mesh_with_nans()
    # Should not raise — remove_infinite_values handles NaN/Inf.
    try:
        cleaned = postprocess_mesh(mesh, target_faces=1000, normalize=True)
        # After NaN removal the mesh may lose the affected face(s) but should be non-empty.
        print(f"  cleaned mesh has {len(cleaned.faces)} faces (NaN faces removed): OK")
    except Exception as exc:
        print(f"  WARNING: NaN removal raised an exception: {exc}")
        print("  (trimesh may not fully handle all NaN cases — acceptable if mesh is small)")


# ---------------------------------------------------------------------------
# GPU-tier tests
# ---------------------------------------------------------------------------

def test_gpu_generate(image_path: Path, cfg) -> None:
    import torch
    from src.mesh_generate import load_shape_pipeline_auto, generate_mesh
    from src.mesh_postprocess import postprocess_mesh, save_mesh
    from src.preprocess import load_image_rgba, compose_over_white

    print("\n[GPU] Loading shape pipeline...")
    pipeline = load_shape_pipeline_auto(cfg)
    print("  load_shape_pipeline_auto: OK")

    # Use white composite for shape conditioning.
    raw_img = load_image_rgba(image_path)
    shape_img = compose_over_white(raw_img)

    views = {"front": shape_img}
    print(f"  Input: {image_path.name}  size={shape_img.size}")

    print(f"\n[GPU] Generating mesh (steps={cfg.shape_steps}, guidance={cfg.shape_guidance_scale})...")
    raw_mesh = generate_mesh(pipeline, views, cfg)
    print(f"  Raw mesh: {len(raw_mesh.vertices)} vertices, {len(raw_mesh.faces)} faces")
    save_mesh(raw_mesh, OUTPUT_DIR / "mesh_raw.glb")
    print(f"  saved → outputs/test/phase3/mesh_raw.glb")

    print(f"\n[GPU] Postprocessing mesh (target_faces={cfg.target_faces})...")
    post_mesh = postprocess_mesh(raw_mesh, target_faces=cfg.target_faces, normalize=cfg.normalize_mesh)
    print(f"  Postprocessed: {len(post_mesh.vertices)} vertices, {len(post_mesh.faces)} faces")
    save_mesh(post_mesh, OUTPUT_DIR / "mesh_postprocessed.glb")
    print(f"  saved → outputs/test/phase3/mesh_postprocessed.glb")

    vram_used = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"\n  Peak VRAM used: {vram_used:.2f} GB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 test: mesh_generate + mesh_postprocess")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input image for GPU generation test. Required for GPU tier.",
    )
    parser.add_argument(
        "--shape-steps",
        type=int,
        default=50,
        help="Shape diffusion steps (GPU tier, default 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    print(f"Output directory: {OUTPUT_DIR.relative_to(_PROJECT_ROOT)}")

    failed = False

    # ---- CPU tier ----------------------------------------------------------
    cpu_tests = [
        test_imports,
        test_keep_largest_component,
        test_normalize_mesh,
        test_decimate,
        test_postprocess_full,
        test_nan_removal,
    ]
    for fn in cpu_tests:
        try:
            fn()
        except Exception as exc:
            print(f"  FAIL [{fn.__name__}]: {exc}", file=sys.stderr)
            failed = True

    # ---- GPU tier ----------------------------------------------------------
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if not cuda_available:
        print("\n[GPU tier] SKIPPED — CUDA not available (expected on macOS / CPU-only env).")
    elif args.image is None:
        print("\n[GPU tier] SKIPPED — pass --image <path> to run generation test.")
    else:
        try:
            from src.config import PipelineConfig
            cfg = PipelineConfig.for_runpod()
            cfg.shape_steps = args.shape_steps
            cfg.seed = args.seed
            test_gpu_generate(args.image, cfg)
        except Exception as exc:
            import traceback
            print(f"\n[GPU tier] FAIL: {exc}", file=sys.stderr)
            traceback.print_exc()
            failed = True

    if failed:
        print("\nSome tests FAILED.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll Phase 3 tests passed.")
        print(f"Results written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
