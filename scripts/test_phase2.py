"""
Phase 2 test — config.py + preprocess.py

Runs the config and preprocessing pipeline stages in sequence.
Steps that fail due to missing dependencies are caught and logged.

Outputs written to  outputs/test/phase2/<timestamp>/:
  front_rgba.png      — RGBA after background removal (if --image given)
  front_shape.png     — composited over white  (DiT input)
  front_paint.png     — composited over gray   (paint reference)
  synthetic_rgba.png  — synthetically generated test image (no --image)
  synthetic_shape.png
  synthetic_paint.png

Usage:
  python scripts/test_phase2.py [--image path/to/input.png] [--seed S]

  If --image is not given a synthetic 512×512 RGBA image is used.
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image, ImageDraw
import numpy as np
from scripts.test_utils import RunLogger, resolve_seed

OUTPUT_DIR = (
    _PROJECT_ROOT / "outputs" / "test" / "phase2"
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log = RunLogger(OUTPUT_DIR, phase="Phase 2")


def save(img: Image.Image, name: str) -> None:
    path = OUTPUT_DIR / name
    img.save(path)
    print(f"      saved → {path.relative_to(_PROJECT_ROOT)}")


def make_synthetic_rgba(size: int = 512) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = size // 8
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=(180, 100, 60, 255),
        outline=(255, 255, 255, 255),
        width=4,
    )
    draw.rectangle([margin * 2, margin * 2, margin * 4, margin * 4], fill=(60, 140, 200, 200))
    draw.rectangle(
        [size - margin * 4, size - margin * 4, size - margin * 2, size - margin * 2],
        fill=(200, 60, 80, 200),
    )
    return img


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_config() -> None:
    print("\n[1] Config dataclasses")
    from src.config import CameraConfig, PipelineConfig

    with log.step("CameraConfig default"):
        cam = CameraConfig()
        assert len(cam.azimuths) == len(cam.elevations) == len(cam.weights) == 6
        assert cam.ortho_scale == 1.0

    with log.step("PipelineConfig.for_macos_dev()"):
        cfg_dev = PipelineConfig.for_macos_dev()
        assert cfg_dev.device == "cpu"
        assert cfg_dev.models_dir == _PROJECT_ROOT / "models"

    with log.step("PipelineConfig.for_runpod(texture_size=2048)"):
        cfg_pod = PipelineConfig.for_runpod(texture_size=2048)
        assert cfg_pod.device == "cuda"
        assert cfg_pod.render_size >= cfg_pod.texture_size

    with log.step("render_size < texture_size raises UserWarning"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = PipelineConfig(render_size=512, texture_size=4096)
            assert len(w) == 1 and issubclass(w[0].category, UserWarning)

    with log.step("CameraConfig length mismatch raises ValueError"):
        raised = False
        try:
            CameraConfig(azimuths=[0, 90], elevations=[0], weights=[1.0, 0.5])
        except ValueError:
            raised = True
        assert raised


def run_preprocess(input_path: "Path | None") -> None:
    print("\n[2] Image loading + compositing")
    from src.preprocess import (
        load_image_rgba,
        collect_views,
        compose_over_gray,
        compose_over_white,
        erode_alpha,
        maybe_remove_background,
        preprocess_all_views,
    )

    if input_path is not None:
        with log.step(f"load_image_rgba({input_path.name})"):
            img = load_image_rgba(input_path)
            log.metric("input_size", f"{img.size[0]}×{img.size[1]}", unit="px")
    else:
        with log.step("make_synthetic_rgba (no --image given)"):
            img = make_synthetic_rgba()
            save(img, "synthetic_rgba.png")
            log.metric("synthetic_size", img.size[0], unit="px")

    with log.step("compose_over_gray"):
        gray_rgb = compose_over_gray(img, gray=127)
        assert gray_rgb.mode == "RGB"
        arr = np.array(gray_rgb)
        assert arr[0, 0, 0] == 127
        label = "front_paint.png" if input_path else "synthetic_paint.png"
        save(gray_rgb, label)

    with log.step("compose_over_white"):
        white_rgb = compose_over_white(img)
        assert white_rgb.mode == "RGB"
        arr_w = np.array(white_rgb)
        assert arr_w[0, 0, 0] == 255
        label = "front_shape.png" if input_path else "synthetic_shape.png"
        save(white_rgb, label)

    with log.step("maybe_remove_background (remove_bg=False)"):
        result = maybe_remove_background(img.copy(), remove_bg=False)
        assert result.mode == "RGBA"

    with log.step("maybe_remove_background (remove_bg=True, stub)"):
        stub_fired: list[bool] = []
        def _stub(i: Image.Image) -> Image.Image:
            stub_fired.append(True)
            return i.convert("RGBA")
        maybe_remove_background(img.copy(), remove_bg=True, background_remover=_stub)
        assert stub_fired

    with log.step("maybe_remove_background (real rembg)"):
        from hy3dshape.rembg import BackgroundRemover  # type: ignore[import]
        remover = BackgroundRemover()
        result_r = maybe_remove_background(img.copy(), remove_bg=True, background_remover=remover)
        assert result_r.mode == "RGBA"
        if input_path:
            save(result_r, "front_rgba.png")

    with log.step("erode_alpha (px=0 no-op)"):
        eroded_noop = erode_alpha(img.copy(), px=0)
        assert np.array_equal(np.array(eroded_noop), np.array(img))

    with log.step("erode_alpha (px=4 shrinks mask)"):
        eroded = erode_alpha(img.copy(), px=4)
        orig_cov = int(np.count_nonzero(np.array(img)[..., 3]))
        ero_cov = int(np.count_nonzero(np.array(eroded)[..., 3]))
        assert ero_cov < orig_cov
        log.metric("alpha_before", orig_cov, unit="px")
        log.metric("alpha_after", ero_cov, unit="px")

    with log.step("collect_views + preprocess_all_views"):
        if input_path:
            views = collect_views(image=str(input_path))
        else:
            # Save synthetic to a temp file so collect_views can load it
            tmp = OUTPUT_DIR / "_synth_tmp.png"
            img.save(tmp)
            views = collect_views(image=str(tmp))
        processed = preprocess_all_views(views, remove_bg=False)
        assert "front" in processed
        for key in ("rgba", "shape", "paint"):
            assert key in processed["front"]
        save(processed["front"]["shape"], "front_shape.png" if input_path else "synthetic_shape_pv.png")

    with log.step("load_image_rgba missing file → FileNotFoundError"):
        raised = False
        try:
            load_image_rgba(Path("nonexistent_xyz.png"))
        except FileNotFoundError:
            raised = True
        assert raised

    with log.step("collect_views no args → ValueError"):
        raised = False
        try:
            collect_views()
        except ValueError:
            raised = True
        assert raised


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 test: config + preprocess")
    parser.add_argument("--image", type=Path, default=None,
                        help="Input image path. If omitted, a synthetic image is used.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (logged for reproducibility).")
    args = parser.parse_args()

    seed = resolve_seed(args.seed)
    log.metric("seed", seed, note="pass --seed to reproduce")
    print(f"Output directory: {OUTPUT_DIR.relative_to(_PROJECT_ROOT)}")

    failed = False

    for stage in (lambda: run_config(), lambda: run_preprocess(args.image)):
        try:
            stage()
        except Exception as exc:
            print(f"\nFAIL: {exc}", file=sys.stderr)
            traceback.print_exc()
            failed = True

    log.save()
    if failed:
        sys.exit(1)
    else:
        print("\nAll Phase 2 tests passed.")


if __name__ == "__main__":
    main()
