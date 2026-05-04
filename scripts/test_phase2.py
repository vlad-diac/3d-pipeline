"""
Phase 2 test — config.py + preprocess.py (CPU-safe, runs on macOS and RunPod).

What it tests:
  1. PipelineConfig and CameraConfig instantiate correctly.
  2. for_macos_dev() and for_runpod() constructors work.
  3. render_size < texture_size triggers a UserWarning.
  4. load_image_rgba() loads a real file or a synthetic RGBA image.
  5. collect_views() builds a single-view dict.
  6. compose_over_gray() and compose_over_white() produce correct RGB images.
  7. maybe_remove_background() skips removal when alpha channel is already present.
  8. preprocess_all_views() returns shape + paint + rgba variants for every view.

Outputs written to  outputs/test/phase2/ :
  front_rgba.png         — RGBA after simulated background removal (alpha kept)
  front_shape.png        — composited over white   (DiT input)
  front_paint.png        — composited over gray    (paint reference)
  synthetic_rgba.png     — synthetically generated test image
  synthetic_shape.png
  synthetic_paint.png

Usage:
  python scripts/test_phase2.py [--image path/to/input.png]

  If --image is not given the script generates a synthetic 512×512 test image.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Allow running from the project root without installing the package.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image, ImageDraw
import numpy as np


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

from datetime import datetime
OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "test" / "phase2" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save(img: Image.Image, name: str) -> None:
    path = OUTPUT_DIR / name
    img.save(path)
    print(f"  saved → {path.relative_to(_PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Synthetic test image (used when no --image is provided)
# ---------------------------------------------------------------------------

def make_synthetic_rgba(size: int = 512) -> Image.Image:
    """
    Create a simple RGBA test image: a coloured circle on a transparent background.
    Lets us exercise the preprocessing without needing a real asset.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Coloured disc — simulates a foreground object with existing alpha mask.
    margin = size // 8
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=(180, 100, 60, 255),
        outline=(255, 255, 255, 255),
        width=4,
    )
    # Add a few colour patches so the gray/white composites are visually distinct.
    draw.rectangle([margin * 2, margin * 2, margin * 4, margin * 4], fill=(60, 140, 200, 200))
    draw.rectangle([size - margin * 4, size - margin * 4, size - margin * 2, size - margin * 2],
                   fill=(200, 60, 80, 200))
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_config() -> None:
    print("\n[1] Config dataclasses")
    from src.config import CameraConfig, PipelineConfig

    cam = CameraConfig()
    assert len(cam.azimuths) == len(cam.elevations) == len(cam.weights) == 6
    assert cam.ortho_scale == 1.0
    print("  CameraConfig default: OK")

    cfg_dev = PipelineConfig.for_macos_dev()
    assert cfg_dev.device == "cpu"
    assert cfg_dev.models_dir == _PROJECT_ROOT / "models"
    assert cfg_dev.third_party_dir == _PROJECT_ROOT / "third_party"
    assert cfg_dev.delight_model_dir == _PROJECT_ROOT / "models" / "delight"
    print("  PipelineConfig.for_macos_dev(): OK")

    cfg_pod = PipelineConfig.for_runpod(texture_size=2048)
    assert cfg_pod.device == "cuda"
    assert cfg_pod.texture_size == 2048
    assert cfg_pod.render_size >= cfg_pod.texture_size
    print("  PipelineConfig.for_runpod(texture_size=2048): OK")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = PipelineConfig(render_size=512, texture_size=4096)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "render_size" in str(w[0].message)
    print("  render_size < texture_size warning: OK")

    bad_cam_raised = False
    try:
        CameraConfig(azimuths=[0, 90], elevations=[0], weights=[1.0, 0.5])
    except ValueError:
        bad_cam_raised = True
    assert bad_cam_raised, "Expected ValueError for mismatched lengths"
    print("  CameraConfig length mismatch raises ValueError: OK")


def test_load_and_compose(input_path: Path | None) -> None:
    print("\n[2] Image loading + compositing")
    from src.preprocess import (
        load_image_rgba,
        collect_views,
        compose_over_gray,
        compose_over_white,
        maybe_remove_background,
        preprocess_all_views,
    )

    # ---- synthetic image (always tested) -----------------------------------
    synth = make_synthetic_rgba()
    save(synth, "synthetic_rgba.png")

    gray_rgb = compose_over_gray(synth, gray=127)
    assert gray_rgb.mode == "RGB", f"Expected RGB, got {gray_rgb.mode}"
    arr = np.array(gray_rgb)
    # Corners (transparent in source) should be exactly gray=127.
    assert arr[0, 0, 0] == 127 and arr[0, 0, 1] == 127 and arr[0, 0, 2] == 127, \
        f"Corner pixel should be gray=127, got {arr[0, 0]}"
    save(gray_rgb, "synthetic_paint.png")
    print("  compose_over_gray: OK")

    white_rgb = compose_over_white(synth)
    assert white_rgb.mode == "RGB"
    arr_w = np.array(white_rgb)
    assert arr_w[0, 0, 0] == 255 and arr_w[0, 0, 1] == 255 and arr_w[0, 0, 2] == 255, \
        f"Corner pixel should be white=255, got {arr_w[0, 0]}"
    save(white_rgb, "synthetic_shape.png")
    print("  compose_over_white: OK")

    # ---- alpha-check: existing mask skips rembg ----------------------------
    result = maybe_remove_background(synth.copy(), remove_bg=True, background_remover=None)
    # synth already has transparency so rembg should be skipped;
    # the call should not raise even without rembg installed.
    assert result.mode == "RGBA"
    print("  maybe_remove_background (existing alpha, skip rembg): OK")

    # ---- fully opaque image triggers rembg path ----------------------------
    opaque = Image.new("RGBA", (64, 64), (200, 100, 50, 255))
    try:
        from hy3dshape.rembg import BackgroundRemover  # type: ignore[import]
        result_opaque = maybe_remove_background(opaque, remove_bg=True)
        assert result_opaque.mode == "RGBA"
        print("  maybe_remove_background (opaque → rembg): OK")
    except ImportError:
        print("  maybe_remove_background (opaque → rembg): SKIPPED (hy3dshape not installed)")

    # ---- real image (optional) --------------------------------------------
    if input_path is not None:
        real_img = load_image_rgba(input_path)
        print(f"  load_image_rgba({input_path.name}): OK  size={real_img.size}")

        views = collect_views(image=str(input_path))
        assert "front" in views
        print("  collect_views (single-image mode): OK")

        processed = preprocess_all_views(views, remove_bg=False)
        assert "front" in processed
        for key in ("rgba", "shape", "paint"):
            assert key in processed["front"], f"Missing key '{key}' in processed['front']"

        save(processed["front"]["rgba"], "front_rgba.png")
        save(processed["front"]["shape"], "front_shape.png")
        save(processed["front"]["paint"], "front_paint.png")
        print("  preprocess_all_views: OK  (outputs saved)")

    # ---- FileNotFoundError ------------------------------------------------
    raised = False
    try:
        load_image_rgba(Path("nonexistent_file_xyz.png"))
    except FileNotFoundError:
        raised = True
    assert raised, "Expected FileNotFoundError"
    print("  load_image_rgba (missing file → FileNotFoundError): OK")

    # ---- collect_views: no args raises ValueError -------------------------
    raised = False
    try:
        collect_views()
    except ValueError:
        raised = True
    assert raised, "Expected ValueError"
    print("  collect_views (no args → ValueError): OK")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 test: config + preprocess")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional input image path. If omitted, a synthetic image is used.",
    )
    args = parser.parse_args()

    print(f"Output directory: {OUTPUT_DIR.relative_to(_PROJECT_ROOT)}")

    try:
        test_config()
        test_load_and_compose(args.image)
        print(f"\nAll Phase 2 tests passed.")
        print(f"Results written to: {OUTPUT_DIR}")
    except AssertionError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
