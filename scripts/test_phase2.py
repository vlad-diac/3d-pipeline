"""
Phase 2 test — config.py + preprocess.py (CPU-safe, runs on macOS and RunPod).

What it tests:
  1. PipelineConfig and CameraConfig instantiate correctly.
  2. for_macos_dev() and for_runpod() constructors work.
  3. render_size < texture_size triggers a UserWarning.
  4. load_image_rgba() loads a real file or a synthetic RGBA image.
  5. collect_views() builds a single-view dict.
  6. compose_over_gray() and compose_over_white() produce correct RGB images.
  7. maybe_remove_background() always calls rembg regardless of existing alpha.
  8. erode_alpha() shrinks the alpha mask (or is a no-op at px=0).
  9. preprocess_all_views() returns shape + paint + rgba variants for every view.

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
from datetime import datetime
from scripts.test_utils import RunLogger, resolve_seed


# ---------------------------------------------------------------------------
# Output directory + logger
# ---------------------------------------------------------------------------

OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "test" / "phase2" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log = RunLogger(OUTPUT_DIR, phase="Phase 2")


def save(img: Image.Image, name: str) -> None:
    path = OUTPUT_DIR / name
    img.save(path)
    print(f"      saved → {path.relative_to(_PROJECT_ROOT)}")


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

    with log.step("CameraConfig default", tier="CPU"):
        cam = CameraConfig()
        assert len(cam.azimuths) == len(cam.elevations) == len(cam.weights) == 6
        assert cam.ortho_scale == 1.0

    with log.step("PipelineConfig.for_macos_dev()", tier="CPU"):
        cfg_dev = PipelineConfig.for_macos_dev()
        assert cfg_dev.device == "cpu"
        assert cfg_dev.models_dir == _PROJECT_ROOT / "models"
        assert cfg_dev.third_party_dir == _PROJECT_ROOT / "third_party"
        assert cfg_dev.delight_model_dir == _PROJECT_ROOT / "models" / "delight"

    with log.step("PipelineConfig.for_runpod(texture_size=2048)", tier="CPU"):
        cfg_pod = PipelineConfig.for_runpod(texture_size=2048)
        assert cfg_pod.device == "cuda"
        assert cfg_pod.texture_size == 2048
        assert cfg_pod.render_size >= cfg_pod.texture_size

    with log.step("render_size < texture_size warning", tier="CPU"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = PipelineConfig(render_size=512, texture_size=4096)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "render_size" in str(w[0].message)

    with log.step("CameraConfig length mismatch raises ValueError", tier="CPU"):
        raised = False
        try:
            CameraConfig(azimuths=[0, 90], elevations=[0], weights=[1.0, 0.5])
        except ValueError:
            raised = True
        assert raised, "Expected ValueError for mismatched lengths"


def test_load_and_compose(input_path: Path | None) -> None:
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

    with log.step("make_synthetic_rgba", tier="CPU"):
        synth = make_synthetic_rgba()
        save(synth, "synthetic_rgba.png")
        log.metric("synthetic_size_px", synth.size[0], unit="px")

    with log.step("compose_over_gray", tier="CPU"):
        gray_rgb = compose_over_gray(synth, gray=127)
        assert gray_rgb.mode == "RGB", f"Expected RGB, got {gray_rgb.mode}"
        arr = np.array(gray_rgb)
        assert arr[0, 0, 0] == 127 and arr[0, 0, 1] == 127 and arr[0, 0, 2] == 127, \
            f"Corner pixel should be gray=127, got {arr[0, 0]}"
        save(gray_rgb, "synthetic_paint.png")

    with log.step("compose_over_white", tier="CPU"):
        white_rgb = compose_over_white(synth)
        assert white_rgb.mode == "RGB"
        arr_w = np.array(white_rgb)
        assert arr_w[0, 0, 0] == 255 and arr_w[0, 0, 1] == 255 and arr_w[0, 0, 2] == 255, \
            f"Corner pixel should be white=255, got {arr_w[0, 0]}"
        save(white_rgb, "synthetic_shape.png")

    with log.step("maybe_remove_background (remove_bg=False)", tier="CPU"):
        # Master switch off — rembg must never be called and the image is
        # returned as-is (no import of BackgroundRemover needed).
        result_off = maybe_remove_background(synth.copy(), remove_bg=False)
        assert result_off.mode == "RGBA"

    with log.step("maybe_remove_background (stub, transparent input)", tier="CPU"):
        # rembg must fire on ANY input — even images that already have
        # transparency.  Pass a no-op stub so this runs without hy3dshape.
        stub_fired: list[bool] = []
        def _stub(img: Image.Image) -> Image.Image:  # noqa: E306
            stub_fired.append(True)
            return img.convert("RGBA")
        maybe_remove_background(synth.copy(), remove_bg=True, background_remover=_stub)
        assert stub_fired, "background_remover was not called on a transparent input"

    with log.step("maybe_remove_background (stub, opaque input)", tier="CPU"):
        # rembg must also fire on fully opaque inputs (existing behaviour).
        stub_fired2: list[bool] = []
        def _stub2(img: Image.Image) -> Image.Image:  # noqa: E306
            stub_fired2.append(True)
            return img.convert("RGBA")
        opaque = Image.new("RGBA", (64, 64), (200, 100, 50, 255))
        maybe_remove_background(opaque, remove_bg=True, background_remover=_stub2)
        assert stub_fired2, "background_remover was not called on an opaque input"

    try:
        from hy3dshape.rembg import BackgroundRemover  # type: ignore[import]
        remover = BackgroundRemover()
        opaque2 = Image.new("RGBA", (64, 64), (200, 100, 50, 255))
        with log.step("maybe_remove_background (real rembg, opaque)", tier="CPU"):
            result_opaque = maybe_remove_background(opaque2, remove_bg=True, background_remover=remover)
            assert result_opaque.mode == "RGBA"
        with log.step("maybe_remove_background (real rembg, transparent)", tier="CPU"):
            result_transp = maybe_remove_background(synth.copy(), remove_bg=True, background_remover=remover)
            assert result_transp.mode == "RGBA"
    except ImportError:
        print("      maybe_remove_background (real rembg): SKIPPED (hy3dshape not installed)")

    with log.step("erode_alpha (px=0 no-op)", tier="CPU"):
        eroded_noop = erode_alpha(synth.copy(), px=0)
        assert np.array_equal(np.array(eroded_noop), np.array(synth)), \
            "erode_alpha(px=0) must return the image unchanged"

    with log.step("erode_alpha (px=4 shrinks mask)", tier="CPU"):
        eroded = erode_alpha(synth.copy(), px=4)
        assert eroded.mode == "RGBA"
        orig_coverage = int(np.count_nonzero(np.array(synth)[..., 3]))
        ero_coverage = int(np.count_nonzero(np.array(eroded)[..., 3]))
        assert ero_coverage < orig_coverage, (
            f"Erosion should reduce alpha coverage ({ero_coverage} >= {orig_coverage})"
        )
        log.metric("alpha_coverage_before", orig_coverage, unit="px")
        log.metric("alpha_coverage_after", ero_coverage, unit="px")

    if input_path is not None:
        with log.step(f"load_image_rgba({input_path.name})", tier="CPU"):
            real_img = load_image_rgba(input_path)
            log.metric("input_image_size", f"{real_img.size[0]}×{real_img.size[1]}", unit="px")

        with log.step("collect_views (single-image)", tier="CPU"):
            views = collect_views(image=str(input_path))
            assert "front" in views

        with log.step("preprocess_all_views", tier="CPU"):
            processed = preprocess_all_views(views, remove_bg=False)
            assert "front" in processed
            for key in ("rgba", "shape", "paint"):
                assert key in processed["front"], f"Missing key '{key}'"
            save(processed["front"]["rgba"],  "front_rgba.png")
            save(processed["front"]["shape"], "front_shape.png")
            save(processed["front"]["paint"], "front_paint.png")

    with log.step("load_image_rgba missing file → FileNotFoundError", tier="CPU"):
        raised = False
        try:
            load_image_rgba(Path("nonexistent_file_xyz.png"))
        except FileNotFoundError:
            raised = True
        assert raised, "Expected FileNotFoundError"

    with log.step("collect_views no args → ValueError", tier="CPU"):
        raised = False
        try:
            collect_views()
        except ValueError:
            raised = True
        assert raised, "Expected ValueError"


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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Omit to generate a random seed (logged for reproducibility).",
    )
    args = parser.parse_args()

    seed = resolve_seed(args.seed)
    log.metric("seed", seed, note="pass --seed to reproduce")

    print(f"Output directory: {OUTPUT_DIR.relative_to(_PROJECT_ROOT)}")
    print(f"Seed: {seed}{' (random)' if args.seed is None else ' (fixed)'}")

    failed = False
    try:
        test_config()
        test_load_and_compose(args.image)
    except AssertionError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        failed = True

    log.save()

    if failed:
        sys.exit(1)
    else:
        print(f"\nAll Phase 2 tests passed.")


if __name__ == "__main__":
    main()
