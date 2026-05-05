"""
Phase 3 test — mesh_generate.py + mesh_postprocess.py

Runs the mesh generation and postprocessing pipeline in sequence.
Requires an input image and a GPU with the shape model downloaded.

Outputs written to  outputs/test/phase3/<timestamp>/:
  mesh_raw.glb           — direct from shape pipeline
  mesh_postprocessed.glb — after cleanup + decimation

Usage:
  python scripts/test_phase3.py --image path/to/input.png [--shape-steps N] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.test_utils import RunLogger, resolve_seed

OUTPUT_DIR = (
    _PROJECT_ROOT / "outputs" / "test" / "phase3"
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log = RunLogger(OUTPUT_DIR, phase="Phase 3")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args, cfg) -> None:
    import torch
    from src.preprocess import load_image_rgba, compose_over_white
    from src.mesh_generate import load_shape_pipeline_auto, generate_mesh
    from src.mesh_postprocess import postprocess_mesh, save_mesh

    with log.step("load_image + white composite"):
        raw_img = load_image_rgba(args.image)
        shape_img = compose_over_white(raw_img)
        log.metric("input_image", f"{args.image.name} {raw_img.size[0]}×{raw_img.size[1]}", unit="px")

    with log.step("load_shape_pipeline_auto"):
        pipeline = load_shape_pipeline_auto(cfg)

    with log.step(f"generate_mesh (steps={cfg.shape_steps})"):
        raw_mesh = generate_mesh(pipeline, {"front": shape_img}, cfg)
        log.metric("raw_vertices", len(raw_mesh.vertices), unit="verts")
        log.metric("raw_faces",    len(raw_mesh.faces),    unit="faces")

    save_mesh(raw_mesh, OUTPUT_DIR / "mesh_raw.glb")
    print(f"      saved → outputs/test/phase3/mesh_raw.glb")

    with log.step(f"postprocess_mesh (target={cfg.target_faces})"):
        post_mesh = postprocess_mesh(
            raw_mesh, target_faces=cfg.target_faces, normalize=cfg.normalize_mesh
        )
        log.metric("post_vertices", len(post_mesh.vertices), unit="verts")
        log.metric("post_faces",    len(post_mesh.faces),    unit="faces")

    save_mesh(post_mesh, OUTPUT_DIR / "mesh_postprocessed.glb")
    print(f"      saved → outputs/test/phase3/mesh_postprocessed.glb")

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    log.metric("peak_vram_gb", round(peak_vram, 2), unit="GB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 test: mesh_generate + mesh_postprocess")
    parser.add_argument("--image", type=Path, required=True,
                        help="Input image for mesh generation.")
    parser.add_argument("--shape-steps", type=int, default=50,
                        help="Shape diffusion steps (default 50).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (logged for reproducibility).")
    args = parser.parse_args()

    seed = resolve_seed(args.seed)
    log.metric("seed", seed, note="pass --seed to reproduce")
    print(f"Output directory: {OUTPUT_DIR.relative_to(_PROJECT_ROOT)}")

    from src.config import PipelineConfig
    cfg = PipelineConfig.for_runpod()
    cfg.shape_steps = args.shape_steps
    cfg.seed = seed

    failed = False
    try:
        run_pipeline(args, cfg)
    except Exception as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        traceback.print_exc()
        failed = True

    log.save()
    if failed:
        sys.exit(1)
    else:
        print("\nPhase 3 pipeline completed.")


if __name__ == "__main__":
    main()
