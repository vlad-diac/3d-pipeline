"""
Phase 4 test — render_multiview.py + paint_multiview.py

Two test tiers:

  CPU tier (runs everywhere — macOS + RunPod):
    • Module import spec checks for render_multiview and paint_multiview.
    • uv_unwrap_mesh on synthetic trimesh objects:
        - Unit cube UV unwrap (verifies xatlas is installed and working)
        - UV unwrap result has valid UV coords (0–1 range, correct count)
        - Scene input is handled (concatenated before unwrap)
        - Over-limit face count raises ValueError
    • upscale_views (Lanczos) on tiny PIL images — no GPU needed.
    • delight_reference with missing delight dir returns gray composite.

  GPU tier (RunPod only — skipped automatically if CUDA unavailable):
    • PaintPipeline init (requires compiled custom_rasterizer CUDA extension).
    • Load UV-unwrapped mesh into renderer.
    • render_conditioning_maps — produce normal + position PIL images.
    • Save normal_00.png … normal_05.png and position_00.png … position_05.png.
    • (Optional) MultiviewDiffusionNet + upscale if paint model is downloaded.
    • Save albedo_00.png … albedo_05.png and mr_00.png … mr_05.png.

Outputs written to  outputs/test/phase4/<timestamp>/:
  synth_cube_uv.glb        — synthetic UV-unwrapped cube
  normal_00.png …          — conditioning normals  (GPU only)
  position_00.png …        — conditioning positions (GPU only)
  albedo_00.png …          — diffusion albedo      (GPU + models only)
  mr_00.png …              — diffusion MR          (GPU + models only)

Usage:
  python scripts/test_phase4.py [--image path/to/input.png] [--paint-steps N]
                                [--seed S] [--skip-paint]

  --image        Required for GPU tier.  Input image used to generate a mesh
                 (from Phase 3) before running Phase 4 operations.
  --glb          Supply a pre-generated postprocessed GLB directly (skips
                 mesh generation, faster for iterating on Phase 4).
  --paint-steps  Number of diffusion steps (default 10).
  --skip-paint   Skip MultiviewDiffusionNet even if models are present.
  --seed         Random seed (default: random, logged for reproducibility).
"""

from __future__ import annotations

import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("test_phase4")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import trimesh
from PIL import Image

from scripts.test_utils import RunLogger, resolve_seed

OUTPUT_DIR = (
    _PROJECT_ROOT
    / "outputs"
    / "test"
    / "phase4"
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log = RunLogger(OUTPUT_DIR, phase="Phase 4")


def save_images(images, prefix: str) -> None:
    for i, img in enumerate(images):
        path = OUTPUT_DIR / f"{prefix}_{i:02d}.png"
        img.save(str(path))
    print(f"      saved {len(images)} images → {prefix}_*.png in {OUTPUT_DIR.relative_to(_PROJECT_ROOT)}")


def save_glb(mesh: trimesh.Trimesh, name: str) -> None:
    path = OUTPUT_DIR / name
    mesh.export(str(path))
    print(f"      saved → {path.relative_to(_PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Synthetic mesh helpers
# ---------------------------------------------------------------------------

def make_unit_cube() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[1.0, 1.0, 1.0])


def make_sphere(subdivisions: int = 2) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=subdivisions)


# ---------------------------------------------------------------------------
# CPU-tier tests
# ---------------------------------------------------------------------------

def test_imports() -> None:
    print("\n[1] Module imports (CPU)")
    import importlib.util

    with log.step("spec-check src.render_multiview", tier="CPU"):
        spec = importlib.util.spec_from_file_location(
            "render_multiview", _PROJECT_ROOT / "src" / "render_multiview.py"
        )
        assert spec is not None, "Could not find render_multiview.py"

    with log.step("spec-check src.paint_multiview", tier="CPU"):
        spec = importlib.util.spec_from_file_location(
            "paint_multiview", _PROJECT_ROOT / "src" / "paint_multiview.py"
        )
        assert spec is not None, "Could not find paint_multiview.py"

    with log.step("import src.render_multiview (deferred GPU deps)", tier="CPU"):
        from src.render_multiview import uv_unwrap_mesh  # noqa: F401

    with log.step("import src.paint_multiview (deferred GPU deps)", tier="CPU"):
        from src.paint_multiview import upscale_views, delight_reference  # noqa: F401


def test_uv_unwrap_cube() -> None:
    print("\n[2] uv_unwrap_mesh — unit cube")
    from src.render_multiview import uv_unwrap_mesh

    cube = make_unit_cube()
    n_verts_before = len(cube.vertices)
    n_faces_before = len(cube.faces)

    with log.step("uv_unwrap_mesh (unit cube)", tier="CPU"):
        uv_mesh = uv_unwrap_mesh(cube)

        # UV coords must be present
        assert hasattr(uv_mesh.visual, "uv"), "uv_unwrap_mesh: .visual.uv not set"
        assert uv_mesh.visual.uv is not None, "uv_unwrap_mesh: .visual.uv is None"

        uvs = np.asarray(uv_mesh.visual.uv)
        assert uvs.ndim == 2 and uvs.shape[1] == 2, (
            f"Expected UV shape (N, 2), got {uvs.shape}"
        )
        assert uvs.min() >= -1e-6 and uvs.max() <= 1.0 + 1e-6, (
            f"UV coords out of [0, 1] range: min={uvs.min():.4f}  max={uvs.max():.4f}"
        )
        # UV count must match vertex count (xatlas may split vertices)
        assert len(uvs) == len(uv_mesh.vertices), (
            f"UV count {len(uvs)} != vertex count {len(uv_mesh.vertices)}"
        )
        # Face count must be preserved
        assert len(uv_mesh.faces) == n_faces_before, (
            f"Face count changed: {n_faces_before} → {len(uv_mesh.faces)}"
        )

        log.metric("uv_verts_before",  n_verts_before,       unit="verts")
        log.metric("uv_verts_after",   len(uv_mesh.vertices), unit="verts")
        log.metric("uv_faces",         len(uv_mesh.faces),    unit="faces")
        log.metric("uv_coords",        len(uvs),              unit="coords")

    save_glb(uv_mesh, "synth_cube_uv.glb")


def test_uv_unwrap_sphere() -> None:
    print("\n[3] uv_unwrap_mesh — sphere")
    from src.render_multiview import uv_unwrap_mesh

    sphere = make_sphere(subdivisions=3)

    with log.step(f"uv_unwrap_mesh (sphere, {len(sphere.faces)} faces)", tier="CPU"):
        uv_mesh = uv_unwrap_mesh(sphere)
        uvs = np.asarray(uv_mesh.visual.uv)
        assert uvs.min() >= -1e-6 and uvs.max() <= 1.0 + 1e-6

        log.metric("sphere_uv_verts", len(uv_mesh.vertices), unit="verts")
        log.metric("sphere_uv_faces", len(uv_mesh.faces),    unit="faces")


def test_uv_unwrap_scene() -> None:
    print("\n[4] uv_unwrap_mesh — trimesh.Scene input")
    from src.render_multiview import uv_unwrap_mesh

    cube = make_unit_cube()
    scene = trimesh.scene.scene.Scene(geometry={"cube": cube})

    with log.step("uv_unwrap_mesh (Scene → concatenate)", tier="CPU"):
        uv_mesh = uv_unwrap_mesh(scene)
        assert isinstance(uv_mesh, trimesh.Trimesh)
        assert hasattr(uv_mesh.visual, "uv") and uv_mesh.visual.uv is not None


def test_uv_unwrap_face_limit() -> None:
    print("\n[5] uv_unwrap_mesh — face limit guard")
    from src.render_multiview import uv_unwrap_mesh
    import numpy as np

    # Build a fake mesh that reports > 500M faces via monkey-patch
    cube = make_unit_cube()

    class _BigMesh(trimesh.Trimesh):
        @property
        def faces(self):
            return np.zeros((500_000_001, 3), dtype=np.int64)

    big = _BigMesh(vertices=cube.vertices, faces=cube.faces, process=False)

    with log.step("uv_unwrap_mesh raises ValueError for >500M faces", tier="CPU"):
        raised = False
        try:
            uv_unwrap_mesh(big)
        except ValueError:
            raised = True
        assert raised, "Expected ValueError for mesh with >500M faces"


def test_upscale_views_lanczos() -> None:
    print("\n[6] upscale_views — Lanczos")
    from src.paint_multiview import upscale_views

    images = [Image.new("RGB", (128, 128), color=(i * 30, 100, 200)) for i in range(6)]
    target = 512

    with log.step(f"upscale_views Lanczos {images[0].size[0]}px → {target}px × {len(images)}", tier="CPU"):
        upscaled = upscale_views(images, target_size=target, use_realesrgan=False)
        assert len(upscaled) == len(images)
        for img in upscaled:
            assert img.size == (target, target), f"Expected ({target}, {target}), got {img.size}"
        log.metric("upscale_input_px",  images[0].size[0], unit="px")
        log.metric("upscale_output_px", target,             unit="px")
        log.metric("upscale_count",     len(upscaled),      unit="images")

    with log.step("upscale_views empty list → empty list", tier="CPU"):
        result = upscale_views([], target_size=target)
        assert result == []


def test_delight_no_model() -> None:
    print("\n[7] delight_reference — missing delight dir returns gray composite")
    from src.paint_multiview import delight_reference
    from src.config import PipelineConfig

    cfg = PipelineConfig.for_macos_dev()
    # Ensure delight dir does NOT exist for this test
    cfg.delight_model_dir = _PROJECT_ROOT / "models" / "_nonexistent_delight_"

    front = Image.new("RGBA", (128, 128), (200, 150, 100, 255))

    with log.step("delight_reference (no model) → gray composite RGB", tier="CPU"):
        result = delight_reference(front, cfg)
        assert result.mode == "RGB", f"Expected RGB, got {result.mode}"
        assert result.size == front.size, f"Size changed: {front.size} → {result.size}"
        log.metric("delight_result_mode", result.mode)
        log.metric("delight_result_size", f"{result.size[0]}×{result.size[1]}", unit="px")


# ---------------------------------------------------------------------------
# GPU-tier tests
# ---------------------------------------------------------------------------

def _load_or_generate_postprocessed_mesh(args, cfg) -> "trimesh.Trimesh":
    """
    Either load a GLB from --glb or generate one via Phase 3 pipeline.
    Returns a postprocessed, normalized mesh ready for UV unwrap.
    """
    if args.glb:
        logger.info("Loading mesh from %s", args.glb)
        loaded = trimesh.load(str(args.glb))
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        return loaded

    # Generate via Phase 3
    from src.mesh_generate import load_shape_pipeline_auto, generate_mesh
    from src.mesh_postprocess import postprocess_mesh
    from src.preprocess import load_image_rgba, compose_over_white

    with log.step("load_image + white composite", tier="GPU"):
        raw_img = load_image_rgba(args.image)
        shape_img = compose_over_white(raw_img)

    with log.step("load_shape_pipeline_auto", tier="GPU"):
        pipeline = load_shape_pipeline_auto(cfg)

    with log.step(f"generate_mesh (steps={cfg.shape_steps})", tier="GPU"):
        raw_mesh = generate_mesh(pipeline, {"front": shape_img}, cfg)
        log.metric("raw_verts", len(raw_mesh.vertices), unit="verts")
        log.metric("raw_faces", len(raw_mesh.faces),    unit="faces")

    with log.step(f"postprocess_mesh (target={cfg.target_faces})", tier="GPU"):
        post_mesh = postprocess_mesh(raw_mesh, target_faces=cfg.target_faces, normalize=True)
        log.metric("post_verts", len(post_mesh.vertices), unit="verts")
        log.metric("post_faces", len(post_mesh.faces),    unit="faces")

    return post_mesh


def test_gpu_render(args, cfg) -> "trimesh.Trimesh | None":
    """
    GPU tier — UV unwrap, init PaintPipeline, render conditioning maps.

    Returns the UV-unwrapped mesh and the initialized PaintPipeline so that
    the optional paint tier can reuse them.
    """
    import torch
    from src.render_multiview import uv_unwrap_mesh, PaintPipeline, render_conditioning_maps

    print("\n[GPU-Render] UV unwrap + conditioning maps")

    post_mesh = _load_or_generate_postprocessed_mesh(args, cfg)

    with log.step("uv_unwrap_mesh", tier="GPU"):
        uv_mesh = uv_unwrap_mesh(post_mesh)
        log.metric("uv_verts", len(uv_mesh.vertices), unit="verts")
        log.metric("uv_faces", len(uv_mesh.faces),    unit="faces")

    uv_glb_path = OUTPUT_DIR / "mesh_uv.glb"
    uv_mesh.export(str(uv_glb_path))
    print(f"      saved → {uv_glb_path.relative_to(_PROJECT_ROOT)}")

    with log.step("PaintPipeline init (MeshRender + ViewProcessor)", tier="GPU"):
        paint_pipeline = PaintPipeline(cfg)

    with log.step("load_mesh into renderer", tier="GPU"):
        paint_pipeline.load_mesh(uv_mesh)

    with log.step("render_conditioning_maps (6 views)", tier="GPU"):
        normal_maps, position_maps = render_conditioning_maps(paint_pipeline, cfg)
        assert len(normal_maps) == len(cfg.camera.azimuths), (
            f"Expected {len(cfg.camera.azimuths)} normal maps, got {len(normal_maps)}"
        )
        assert len(position_maps) == len(cfg.camera.azimuths), (
            f"Expected {len(cfg.camera.azimuths)} position maps, got {len(position_maps)}"
        )
        log.metric("normal_maps",   len(normal_maps),   unit="images")
        log.metric("position_maps", len(position_maps), unit="images")

    save_images(normal_maps,   "normal")
    save_images(position_maps, "position")

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    log.metric("peak_vram_render_gb", round(peak_vram, 2), unit="GB")

    return uv_mesh, paint_pipeline, normal_maps, position_maps


def test_gpu_paint(args, cfg, uv_mesh, paint_pipeline, normal_maps, position_maps) -> None:
    """
    GPU tier (optional) — MultiviewDiffusionNet + upscale.
    Skipped if paint model weights are not downloaded.
    """
    import torch
    from src.paint_multiview import (
        MultiviewDiffusionNet,
        delight_reference,
        upscale_views,
    )
    from src.preprocess import load_image_rgba

    paint_model_dir = cfg.models_dir / "hunyuan3d-paintpbr-v2-1"
    if not paint_model_dir.exists():
        print(
            f"\n[GPU-Paint] SKIPPED — paint model not found at {paint_model_dir}.\n"
            "  Run:  python scripts/download_models.py  to download weights."
        )
        return

    print("\n[GPU-Paint] Delight + MultiviewDiffusionNet + upscale")

    # Prepare reference image
    with log.step("delight_reference", tier="GPU"):
        front_rgba = load_image_rgba(args.image)
        reference = delight_reference(front_rgba, cfg)
        reference.save(str(OUTPUT_DIR / "reference_delighted.png"))

    with log.step("MultiviewDiffusionNet init (~21 GB VRAM)", tier="GPU"):
        mvd = MultiviewDiffusionNet(cfg)

    with log.step(f"paint diffusion (steps={cfg.paint_steps})", tier="GPU"):
        paint_out = mvd(reference, normal_maps, position_maps, cfg)
        albedo_views = paint_out["albedo"]
        mr_views     = paint_out["mr"]
        log.metric("albedo_views", len(albedo_views), unit="images")
        log.metric("mr_views",     len(mr_views),     unit="images")

    save_images(albedo_views, "albedo")
    save_images(mr_views,     "mr")

    with log.step(f"upscale_views Lanczos → {cfg.render_size}px", tier="GPU"):
        albedo_up = upscale_views(albedo_views, target_size=cfg.render_size)
        mr_up     = upscale_views(mr_views,     target_size=cfg.render_size)
        log.metric("upscaled_albedo_px", albedo_up[0].size[0] if albedo_up else 0, unit="px")

    save_images(albedo_up, "albedo_upscaled")
    save_images(mr_up,     "mr_upscaled")

    mvd.unload()
    torch.cuda.empty_cache()

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    log.metric("peak_vram_paint_gb", round(peak_vram, 2), unit="GB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 test: render_multiview + paint_multiview")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input image for mesh generation (GPU tier). Required unless --glb is given.",
    )
    parser.add_argument(
        "--glb",
        type=Path,
        default=None,
        help="Pre-generated postprocessed GLB (skips Phase 3 mesh generation).",
    )
    parser.add_argument(
        "--shape-steps",
        type=int,
        default=50,
        help="Shape diffusion steps when generating mesh from --image (default 50).",
    )
    parser.add_argument(
        "--paint-steps",
        type=int,
        default=10,
        help="Paint diffusion steps (default 10).",
    )
    parser.add_argument(
        "--skip-paint",
        action="store_true",
        help="Skip MultiviewDiffusionNet even if the paint model is downloaded.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Omit to use a random seed (logged for reproducibility).",
    )
    args = parser.parse_args()

    seed = resolve_seed(args.seed)
    log.metric("seed", seed, note="pass --seed to reproduce")

    print(f"Output directory: {OUTPUT_DIR.relative_to(_PROJECT_ROOT)}")
    print(f"Seed: {seed}{' (random)' if args.seed is None else ' (fixed)'}")

    failed = False

    # ---- CPU tier ----------------------------------------------------------
    cpu_tests = [
        test_imports,
        test_uv_unwrap_cube,
        test_uv_unwrap_sphere,
        test_uv_unwrap_scene,
        test_uv_unwrap_face_limit,
        test_upscale_views_lanczos,
        test_delight_no_model,
    ]
    for fn in cpu_tests:
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"  FAIL [{fn.__name__}]: {exc}", file=sys.stderr)
            traceback.print_exc()
            failed = True

    # ---- GPU tier ----------------------------------------------------------
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if not cuda_available:
        print("\n[GPU tier] SKIPPED — CUDA not available (expected on macOS / CPU-only env).")
    elif args.image is None and args.glb is None:
        print("\n[GPU tier] SKIPPED — pass --image <path> (or --glb <path>) to run GPU tests.")
    else:
        try:
            from src.config import PipelineConfig

            cfg = PipelineConfig.for_runpod()
            cfg.shape_steps  = args.shape_steps
            cfg.paint_steps  = args.paint_steps
            cfg.seed         = seed

            uv_mesh, paint_pipeline, normal_maps, position_maps = test_gpu_render(args, cfg)

            if not args.skip_paint:
                test_gpu_paint(args, cfg, uv_mesh, paint_pipeline, normal_maps, position_maps)
            else:
                print("\n[GPU-Paint] SKIPPED — --skip-paint flag set.")

        except Exception as exc:
            import traceback
            print(f"\n[GPU tier] FAIL: {exc}", file=sys.stderr)
            traceback.print_exc()
            failed = True

    log.save()

    if failed:
        print("\nSome tests FAILED.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll Phase 4 tests passed.")


if __name__ == "__main__":
    main()
