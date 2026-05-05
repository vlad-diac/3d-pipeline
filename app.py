"""
Gradio UI for testing the Hunyuan3D multiview pipeline.

Sections:
  1. Test Name  — sets the output folder name (combined with timestamp).
  2. Dataset & Stages — input folder, auto-detected view count, stage
     checkboxes, run button.
  3. Results — streaming HTML table (one row per completed stage) and a
     Model3D viewer that updates whenever a new mesh is produced.

Usage:
    python app.py
"""

from __future__ import annotations

import base64
import io
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

_VIEW_ORDER = ["front", "left", "right", "back"]
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"})

STAGE_IDS = [
    "remove_background",    # rembg / BiRefNet + erode + center + resize → clean RGBA
    "canonical_multiview",  # optional: MV-Adapter → front/right/back/left RGBA
    "preprocess",           # compositing only: white (shape) + gray (paint)
    "mesh_generate",
    "mesh_postprocess",
    "render_multiview",
    "paint_multiview",
    "bake_texture",
    "inpaint_texture",
    "export_glb",
]

STAGE_LABELS = {
    "remove_background":   "Remove Background",
    "canonical_multiview": "Canonical Multiview",
    "preprocess":          "Preprocess",
    "mesh_generate":       "Mesh Generate",
    "mesh_postprocess":    "Mesh Postprocess",
    "render_multiview":    "Render Multiview",
    "paint_multiview":     "Paint Multiview",
    "bake_texture":        "Bake Texture",
    "inpaint_texture":     "Inpaint Texture",
    "export_glb":          "Export GLB",
}

# canonical_multiview requires GPU (SDXL + MV-Adapter, ~14 GB VRAM)
GPU_REQUIRED = {
    "canonical_multiview",
    "mesh_generate", "mesh_postprocess",
    "render_multiview", "paint_multiview",
    "bake_texture", "export_glb",
}

# Stages selected by default (canonical_multiview opt-in only)
_DEFAULT_STAGE_IDS = [s for s in STAGE_IDS if s != "canonical_multiview"]

# ─── View / image helpers ────────────────────────────────────────────────────

def _find_single_image(folder: Path) -> Optional[Path]:
    candidates = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in _IMAGE_EXTS and not p.name.startswith(".")
    )
    return candidates[0] if candidates else None


def scan_views(folder_str: str) -> tuple[str, dict[str, Path]]:
    """
    Detect single-view or multiview images in a folder.

    Returns:
        (mode, view_paths)
        mode      — "multiview", "single", or "none"
        view_paths — dict of {view_name: Path}
    """
    folder = Path(folder_str.strip())
    if not folder.is_dir():
        return "none", {}

    try:
        from src.preprocess import scan_multiview_folder
        paths = scan_multiview_folder(folder)
        return "multiview", paths
    except (ValueError, NotADirectoryError):
        pass
    except ImportError:
        # src not importable — fall through to manual scan
        found: dict[str, Path] = {}
        for p in sorted(folder.iterdir()):
            if p.name.startswith(".") or p.suffix.lower() not in _IMAGE_EXTS:
                continue
            stem = p.stem.lower()
            for orient in _VIEW_ORDER:
                if stem.endswith(f"-{orient}") and orient not in found:
                    found[orient] = p
        if all(o in found for o in _VIEW_ORDER):
            return "multiview", {o: found[o] for o in _VIEW_ORDER}

    single = _find_single_image(folder)
    if single:
        return "single", {"front": single}

    return "none", {}


def _pil_to_b64(img: Image.Image, max_px: int = 280) -> str:
    """Return a base64 PNG data-URI thumbnail of a PIL image."""
    img = img.copy()
    img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _path_to_b64(path: Path, max_px: int = 280) -> Optional[str]:
    """Load an image from disk and return a base64 data-URI thumbnail."""
    try:
        img = Image.open(path).convert("RGB")
        return _pil_to_b64(img, max_px=max_px)
    except Exception:
        return None


# ─── Table HTML ──────────────────────────────────────────────────────────────

_CSS = """
<style>
.pp-table {
  border-collapse: collapse;
  width: 100%;
  font-family: ui-monospace, monospace;
  font-size: 12px;
  table-layout: fixed;
}
.pp-table th {
  background: #1e1e2e;
  color: #cdd6f4;
  padding: 6px 8px;
  border: 1px solid #45475a;
  text-align: center;
}
.pp-table td {
  border: 1px solid #45475a;
  padding: 6px 8px;
  vertical-align: top;
}
.pp-table td.stage-cell {
  font-weight: 600;
  white-space: nowrap;
  color: #cba6f7;
  width: 130px;
}
.pp-table td.img-cell {
  text-align: center;
  background: #181825;
}
.pp-table td.metrics-cell {
  color: #a6adc8;
  font-size: 11px;
  width: 160px;
  line-height: 1.6;
}
.pp-table td.error-cell {
  background: #3d1010;
  color: #f38ba8;
  font-size: 11px;
  word-break: break-word;
}
.pp-table img {
  max-width: 220px;
  max-height: 220px;
  display: block;
  margin: auto;
  border-radius: 4px;
}
.pp-mesh-icon { font-size: 52px; display: block; text-align: center; }
.pp-glb-note  { font-size: 10px; color: #89b4fa; text-align: center; }
.pp-spin      { color: #f9e2af; }
.pp-ok        { color: #a6e3a1; }
.pp-fail      { color: #f38ba8; }
.pp-cost      { font-size: 12px; line-height: 1.7; }
.pp-cost-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 6px 20px; }
.pp-cost-item { }
.pp-cost-item .cost-label { color: #9399b2; font-size: 11px; }
.pp-cost-item .cost-value { color: #cdd6f4; font-weight: bold; }
</style>
"""


def _img_cell(b64: Optional[str]) -> str:
    if b64:
        return f'<td class="img-cell"><img src="{b64}"/></td>'
    return '<td class="img-cell" style="color:#585b70;">—</td>'


def _span_cell(content: str, n: int, cls: str = "img-cell") -> str:
    return f'<td class="{cls}" colspan="{n}">{content}</td>'


def _metrics_cell(m: dict, error: Optional[str] = None, running: bool = False) -> str:
    if running:
        return '<td class="metrics-cell"><span class="pp-spin">⏳ running…</span></td>'
    if error:
        snippet = (error[:320] + "…") if len(error) > 320 else error
        return f'<td class="error-cell"><span class="pp-fail">✗</span> {snippet}</td>'
    lines = [f"<b>{k}</b>: {v}" for k, v in m.items()]
    inner = "<br/>".join(lines) if lines else "—"
    return f'<td class="metrics-cell">{inner}</td>'


def build_table_html(rows: list[dict], n_views: int) -> str:
    """
    Render the results table as an HTML string.

    Each row dict must contain:
      stage   : str
      type    : "input" | "images" | "mesh" | "atlas" | "running" | "error"
      views   : list[Optional[str]]  — base64 URIs, one per view column
      atlas   : Optional[str]        — base64 URI for a single full-width image
      glb_path: Optional[Path]       — path shown under the mesh icon
      metrics : dict[str, str]
      error   : Optional[str]
      running : bool
    """
    view_labels = (["Front", "Left", "Right", "Back"][:n_views]
                   if n_views <= 4 else [f"View {i}" for i in range(n_views)])

    header = (
        "<tr>"
        + '<th style="width:130px">Stage</th>'
        + "".join(f"<th>{lbl}</th>" for lbl in view_labels)
        + '<th style="width:160px">Metrics</th>'
        + "</tr>"
    )

    body_rows: list[str] = []
    for row in rows:
        stage      = row.get("stage", "")
        rtype      = row.get("type", "images")
        error      = row.get("error")
        running    = row.get("running", False)
        metrics    = row.get("metrics", {})
        views      = row.get("views") or [None] * n_views
        atlas      = row.get("atlas")
        glb_path   = row.get("glb_path")

        stage_td   = f'<td class="stage-cell">{stage}</td>'
        metrics_td = _metrics_cell(metrics, error=error, running=running)

        if rtype in ("input", "images"):
            padded = list(views) + [None] * max(0, n_views - len(views))
            img_tds = "".join(_img_cell(v) for v in padded[:n_views])
            body_rows.append(f"<tr>{stage_td}{img_tds}{metrics_td}</tr>")

        elif rtype == "mesh":
            name = glb_path.name if glb_path else "mesh.glb"
            icon_html = (
                f'<span class="pp-mesh-icon">🧊</span>'
                f'<div class="pp-glb-note">📁 {name}</div>'
            )
            body_rows.append(
                f"<tr>{stage_td}"
                f"{_span_cell(icon_html, n_views)}"
                f"{metrics_td}</tr>"
            )

        elif rtype == "atlas":
            inner = (
                f'<img src="{atlas}"/>' if atlas
                else '<span style="color:#585b70;">—</span>'
            )
            body_rows.append(
                f"<tr>{stage_td}"
                f"{_span_cell(inner, n_views)}"
                f"{metrics_td}</tr>"
            )

        elif rtype == "running":
            spin = '<span class="pp-spin">⏳ running…</span>'
            body_rows.append(
                f"<tr>{stage_td}"
                f"{_span_cell(spin, n_views)}"
                f"{metrics_td}</tr>"
            )

        elif rtype == "error":
            fail = '<span class="pp-fail">✗ Stage failed — see Metrics</span>'
            body_rows.append(
                f"<tr>{stage_td}"
                f"{_span_cell(fail, n_views)}"
                f"{metrics_td}</tr>"
            )

        elif rtype == "cost":
            cost = row.get("cost", {})
            items_html = "".join(
                f'<div class="pp-cost-item">'
                f'<div class="cost-label">{k}</div>'
                f'<div class="cost-value">{v}</div>'
                f'</div>'
                for k, v in cost.items()
            )
            cost_html = f'<div class="pp-cost"><div class="pp-cost-grid">{items_html}</div></div>'
            body_rows.append(
                f"<tr>{stage_td}"
                f'{_span_cell(cost_html, n_views + 1)}'
                f"</tr>"
            )

    body = "\n".join(body_rows)
    return (
        f"{_CSS}"
        f'<table class="pp-table">'
        f"<thead>{header}</thead>"
        f"<tbody>{body}</tbody>"
        f"</table>"
    )


# ─── Individual stage runners ────────────────────────────────────────────────
# Each runner signature: (state, save_dir, cfg, log) → (row_dict, Optional[Path])
# Runners update `state` in-place with objects needed by later stages.


def _stage_remove_background(
    state: dict,
    save_dir: Path,
    cfg,
    log,
) -> tuple[list[dict], Optional[Path]]:
    """
    Stage 1 — Subject isolation.

    Loads raw images, runs background removal (rembg / hy3dshape), optional
    alpha erosion, centering, and resize.  Writes clean RGBA images to
    ``state["rgba_views"]``.

    Responsibility: *isolation only* — no compositing happens here.
    """
    from src.preprocess import (
        collect_views,
        maybe_remove_background,
        erode_alpha,
        center_and_pad,
        resize_to_size,
        compose_over_gray,
        _build_background_remover,  # type: ignore[attr-defined]
    )

    view_paths: dict[str, Path] = state["view_paths"]
    view_keys  = list(view_paths.keys())
    n_cols     = state["n_views"]

    remove_bg      = state.get("remove_bg", True)
    erode_px       = getattr(cfg, "rembg_erode_px", 0)
    center_subject = state.get("center_subject", True)
    target_size    = state.get("preprocess_target_size")

    with log.step("collect_views"):
        raw_views = collect_views(**{k: str(v) for k, v in view_paths.items()})

    state["raw_views"] = raw_views

    # Build background remover once; reuse across views.
    bg_remover = None
    if remove_bg:
        bg_remover = _build_background_remover()
        if bg_remover is None:
            remove_bg = False

    # Track per-view intermediates for display rows.
    intermediates: dict[str, dict[str, Image.Image]] = {}
    for name, img in raw_views.items():
        entry: dict[str, Image.Image] = {"raw": img.convert("RGBA")}

        with log.step(f"remove_bg({name})") if remove_bg else _noop_ctx():
            rgba = maybe_remove_background(
                entry["raw"], remove_bg=remove_bg, background_remover=bg_remover
            )
            rgba = erode_alpha(rgba, px=erode_px)
        entry["rgba"] = rgba

        centered = center_and_pad(rgba) if center_subject else rgba
        entry["centered"] = centered

        resized = resize_to_size(centered, target_size) if target_size is not None else centered
        entry["resized"] = resized

        intermediates[name] = entry

    # Store clean RGBA (after all isolation steps) in state.
    state["rgba_views"] = {name: d["resized"] for name, d in intermediates.items()}
    log.metric("views", len(intermediates))

    # Save all intermediate images to disk.
    for orient, entry in intermediates.items():
        for key, img in entry.items():
            img.save(str(save_dir / f"{key}_{orient}.png"))

    # --- Build display rows ---
    def _b64_on_gray(key: str) -> list[Optional[str]]:
        b64s: list[Optional[str]] = []
        for orient in view_keys:
            entry = intermediates.get(orient)
            if entry and key in entry:
                pil = entry[key]
                display = compose_over_gray(pil) if pil.mode == "RGBA" else pil
                b64s.append(_pil_to_b64(display))
            else:
                b64s.append(None)
        while len(b64s) < n_cols:
            b64s.append(None)
        return b64s[:n_cols]

    rows: list[dict] = []
    rows.append({
        "stage":   "RB: Raw",
        "type":    "images",
        "views":   _b64_on_gray("raw"),
        "metrics": {"step": "loaded RGBA"},
    })
    if remove_bg:
        rows.append({
            "stage":   "RB: Rembg",
            "type":    "images",
            "views":   _b64_on_gray("rgba"),
            "metrics": {"step": "bg removed"},
        })
    if center_subject:
        rows.append({
            "stage":   "RB: Center+Pad",
            "type":    "images",
            "views":   _b64_on_gray("centered"),
            "metrics": {"step": "centered & padded"},
        })
    if target_size is not None:
        rows.append({
            "stage":   f"RB: Resize→{target_size}",
            "type":    "images",
            "views":   _b64_on_gray("resized"),
            "metrics": {"step": f"→ {target_size}×{target_size} px"},
        })

    return rows, None


def _stage_canonical_multiview(
    state: dict,
    save_dir: Path,
    cfg,
    log,
) -> tuple[dict, Optional[Path]]:
    """
    Stage 2 — Canonical view synthesis (optional).

    Takes the clean RGBA from ``remove_background`` and uses MV-Adapter i2mv
    SDXL to generate consistent front / right / back / left canonical views.

    Overwrites ``state["rgba_views"]`` so that the downstream ``preprocess``
    stage composites the canonical views instead of the original user photos.
    Also sets ``state["canonical_views"]`` which is kept for ``paint_multiview``
    to use as the albedo source.
    """
    if "rgba_views" not in state:
        raise RuntimeError("Remove Background must run before Canonical Multiview.")

    from src.canonical_multiview import generate_canonical_views
    from src.config import CameraConfig

    front_rgba = state["rgba_views"].get("front")
    if front_rgba is None:
        raise RuntimeError("No 'front' view found in rgba_views.")

    with log.step("generate_canonical_views"):
        canonical_views, controls = generate_canonical_views(
            front_rgba, cfg.canonical,
            save_dir=save_dir,
            extra_views=state["rgba_views"],  # pass all labeled views for per-view maps
        )

    # Overwrite rgba_views so preprocess composites the canonical RGBA.
    state["rgba_views"]      = canonical_views
    state["canonical_views"] = canonical_views
    state["mode"]            = "multiview"
    state["n_views"]         = 4

    # Switch bake camera layout to 4 cardinal views.
    cfg.camera = CameraConfig.four_cardinal()

    log.metric("gen_size", f"{cfg.canonical.gen_size}px")
    log.metric("steps",    str(cfg.canonical.steps))

    # Column order matches the table headers produced by build_table_html:
    # "Front", "Left", "Right", "Back" (n_views=4).
    # All rows use this same order so each depth/canny map sits directly below
    # the canonical view it was used to condition.
    TABLE_COL_ORDER = ["front", "left", "right", "back"]
    n_cols = state["n_views"]

    view_b64s = [_pil_to_b64(canonical_views[k]) for k in TABLE_COL_ORDER]
    row_canonical = {
        "stage":   STAGE_LABELS["canonical_multiview"],
        "type":    "images",
        "views":   view_b64s,
        "metrics": {
            "gen_size": f"{cfg.canonical.gen_size}px",
            "steps":    str(cfg.canonical.steps),
        },
    }

    rows: list[dict] = [row_canonical]

    # One sub-row per control type.  Controls are keyed by view name so we can
    # look up each map using TABLE_COL_ORDER to guarantee column alignment.
    for ctrl_name, view_maps in controls.items():
        ctrl_b64s = [
            _pil_to_b64(view_maps[k]) if k in view_maps else None
            for k in TABLE_COL_ORDER
        ]
        rows.append({
            "stage":   f"↳ {ctrl_name}",
            "type":    "images",
            "views":   ctrl_b64s,
            "metrics": {},
        })

    return rows if len(rows) > 1 else row_canonical, None


def _stage_preprocess(
    state: dict,
    save_dir: Path,
    cfg,
    log,
) -> tuple[list[dict], Optional[Path]]:
    """
    Stage 3 — Model formatting (compositing only).

    Composites clean RGBA views (from ``remove_background`` or
    ``canonical_multiview``) over white (shape DiT) and gray (paint diffusion).
    No background removal runs here.
    """
    if "rgba_views" not in state:
        raise RuntimeError("Remove Background must run before Preprocess.")

    from src.preprocess import composite_views

    rgba_views = state["rgba_views"]
    view_keys  = list(rgba_views.keys())
    n_cols     = state["n_views"]

    with log.step("composite_views"):
        processed = composite_views(rgba_views)

    state["processed"]   = processed
    state["shape_views"] = {k: v["shape"] for k, v in processed.items()}
    state["paint_views"] = {k: v["paint"] for k, v in processed.items()}
    log.metric("views", len(processed))

    # Save to disk.
    for orient, entry in processed.items():
        for key, img in entry.items():
            img.save(str(save_dir / f"{key}_{orient}.png"))

    def _b64(key: str) -> list[Optional[str]]:
        b64s: list[Optional[str]] = []
        for orient in view_keys:
            entry = processed.get(orient)
            if entry and key in entry:
                b64s.append(_pil_to_b64(entry[key]))
            else:
                b64s.append(None)
        while len(b64s) < n_cols:
            b64s.append(None)
        return b64s[:n_cols]

    rows: list[dict] = [
        {
            "stage":   "Pre: Paint ref",
            "type":    "images",
            "views":   _b64("paint"),
            "metrics": {"step": "gray composite (paint)"},
        },
        {
            "stage":   "Pre: Shape ref",
            "type":    "images",
            "views":   _b64("shape"),
            "metrics": {"step": "white composite (shape)"},
        },
    ]
    return rows, None


class _noop_ctx:
    """No-op context manager used as a drop-in when a step should be skipped."""
    def __enter__(self):  return self
    def __exit__(self, *_): pass


def _stage_mesh_generate(
    state: dict,
    save_dir: Path,
    cfg,
    log,
) -> tuple[dict, Optional[Path]]:
    if "shape_views" not in state:
        raise RuntimeError("Preprocess must run before Mesh Generate.")

    from src.mesh_generate import load_shape_pipeline_auto, generate_mesh

    shape_views = state["shape_views"]
    cfg.use_multiview_shape = (state["mode"] == "multiview")

    with log.step("load_shape_pipeline_auto"):
        shape_pipe = load_shape_pipeline_auto(cfg)

    with log.step(f"generate_mesh (steps={cfg.shape_steps})"):
        raw_mesh = generate_mesh(shape_pipe, shape_views, cfg)
        log.metric("raw_vertices", len(raw_mesh.vertices), unit="verts")
        log.metric("raw_faces",    len(raw_mesh.faces),    unit="faces")

    mesh_path = state["mesh_saver"].save(raw_mesh, save_dir, "mesh_raw")
    state["raw_mesh"] = raw_mesh

    return {
        "stage":    STAGE_LABELS["mesh_generate"],
        "type":     "mesh",
        "glb_path": mesh_path,
        "metrics": {
            "verts": f"{len(raw_mesh.vertices):,}",
            "faces": f"{len(raw_mesh.faces):,}",
        },
    }, mesh_path


def _stage_mesh_postprocess(
    state: dict,
    save_dir: Path,
    cfg,
    log,
) -> tuple[dict, Optional[Path]]:
    if "raw_mesh" not in state:
        raise RuntimeError("Mesh Generate must run before Mesh Postprocess.")

    from src.mesh_postprocess import postprocess_mesh

    with log.step(f"postprocess_mesh (target={cfg.target_faces})"):
        post_mesh = postprocess_mesh(
            state["raw_mesh"],
            target_faces=cfg.target_faces,
            normalize=cfg.normalize_mesh,
        )
        log.metric("post_vertices", len(post_mesh.vertices), unit="verts")
        log.metric("post_faces",    len(post_mesh.faces),    unit="faces")

    mesh_path = state["mesh_saver"].save(post_mesh, save_dir, "mesh_postprocessed")
    state["post_mesh"] = post_mesh

    return {
        "stage":    STAGE_LABELS["mesh_postprocess"],
        "type":     "mesh",
        "glb_path": mesh_path,
        "metrics": {
            "verts": f"{len(post_mesh.vertices):,}",
            "faces": f"{len(post_mesh.faces):,}",
        },
    }, mesh_path


def _stage_render_multiview(
    state: dict,
    save_dir: Path,
    cfg,
    log,
) -> tuple[dict, Optional[Path]]:
    if "post_mesh" not in state:
        raise RuntimeError("Mesh Postprocess must run before Render Multiview.")

    from src.render_multiview import uv_unwrap_mesh, PaintPipeline, render_conditioning_maps

    with log.step("uv_unwrap_mesh"):
        uv_mesh = uv_unwrap_mesh(state["post_mesh"])
        log.metric("uv_verts", len(uv_mesh.vertices), unit="verts")

    uv_path = state["mesh_saver"].save(uv_mesh, save_dir, "mesh_uv")

    with log.step("PaintPipeline init"):
        paint_pipeline = PaintPipeline(cfg)

    with log.step("load_mesh"):
        paint_pipeline.load_mesh(uv_mesh)

    with log.step("render_conditioning_maps"):
        normal_maps, position_maps = render_conditioning_maps(paint_pipeline, cfg)
        log.metric("normal_maps",   len(normal_maps))
        log.metric("position_maps", len(position_maps))

    for i, img in enumerate(normal_maps):
        img.save(str(save_dir / f"normal_{i:02d}.png"))
    for i, img in enumerate(position_maps):
        img.save(str(save_dir / f"position_{i:02d}.png"))

    state["paint_pipeline"] = paint_pipeline
    state["uv_mesh"]        = uv_mesh
    state["normal_maps"]    = normal_maps
    state["position_maps"]  = position_maps

    n_cols = state["n_views"]
    # Show first n_cols normal maps in the table columns
    view_b64s = [_pil_to_b64(img) for img in normal_maps[:n_cols]]
    while len(view_b64s) < n_cols:
        view_b64s.append(None)

    return {
        "stage":   STAGE_LABELS["render_multiview"],
        "type":    "images",
        "views":   view_b64s,
        "metrics": {"normal_maps": str(len(normal_maps))},
    }, uv_path


def _stage_paint_multiview(
    state: dict,
    save_dir: Path,
    cfg,
    log,
) -> tuple[dict, Optional[Path]]:
    required = ("paint_pipeline", "normal_maps", "position_maps")
    missing = [k for k in required if k not in state]
    if missing:
        raise RuntimeError(f"Render Multiview must run first. Missing: {missing}")

    from src.paint_multiview import MultiviewDiffusionNet, delight_reference, upscale_views
    from src.preprocess import load_image_rgba

    paint_pipeline   = state["paint_pipeline"]
    canonical_views  = state.get("canonical_views")  # set by canonical_multiview stage

    # Use canonical front view as delight reference if available (it's the
    # MV-Adapter-generated anchor which is already clean and well-lit).
    if canonical_views is not None:
        front_rgba = canonical_views["front"].convert("RGBA")
    else:
        front_path = state["view_paths"]["front"]
        front_rgba = load_image_rgba(front_path)

    with log.step("delight_reference"):
        reference = delight_reference(front_rgba, cfg)
        reference.save(str(save_dir / "reference_delighted.png"))

    with log.step("MultiviewDiffusionNet init"):
        mvd = MultiviewDiffusionNet(cfg)

    with log.step(f"paint diffusion (steps={cfg.paint_steps})"):
        paint_out    = mvd(reference, state["normal_maps"], state["position_maps"], cfg)
        mr_views     = paint_out["mr"]
        log.metric("mr_views", len(mr_views))

    # When canonical views are available, use them as albedo and take only the
    # MR channel from HunyuanPaint (which is conditioned on normals/positions).
    with log.step("upscale_views"):
        if canonical_views is not None:
            canon_order = ["front", "right", "back", "left"]
            albedo_up = upscale_views(
                [canonical_views[k].convert("RGB") for k in canon_order],
                target_size=cfg.render_size,
            )
            mr_up = upscale_views(mr_views[:4], target_size=cfg.render_size)
            log.metric("albedo_source", "canonical_views")
        else:
            albedo_views = paint_out["albedo"]
            albedo_up = upscale_views(albedo_views, target_size=cfg.render_size)
            mr_up     = upscale_views(mr_views,     target_size=cfg.render_size)
            log.metric("albedo_source",  "hunyuan_paint")
            log.metric("albedo_views",   len(albedo_views))

    for i, img in enumerate(albedo_up):
        img.save(str(save_dir / f"albedo_upscaled_{i:02d}.png"))
    for i, img in enumerate(mr_up):
        img.save(str(save_dir / f"mr_upscaled_{i:02d}.png"))

    mvd.unload()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    state["albedo_views"] = albedo_up
    state["mr_views"]     = mr_up

    n_cols = state["n_views"]
    view_b64s = [_pil_to_b64(img) for img in albedo_up[:n_cols]]
    while len(view_b64s) < n_cols:
        view_b64s.append(None)

    return {
        "stage":   STAGE_LABELS["paint_multiview"],
        "type":    "images",
        "views":   view_b64s,
        "metrics": {"upscaled_px": f"{albedo_up[0].size[0]}px" if albedo_up else "—"},
    }, None


def _stage_bake_texture(
    state: dict,
    save_dir: Path,
    cfg,
    log,
) -> tuple[dict, Optional[Path]]:
    required = ("paint_pipeline", "albedo_views", "mr_views")
    missing = [k for k in required if k not in state]
    if missing:
        raise RuntimeError(f"Paint Multiview must run first. Missing: {missing}")

    from src.bake_texture import bake_multiview_textures

    with log.step("bake_multiview_textures"):
        tex_alb, mask_alb, tex_mr, mask_mr = bake_multiview_textures(
            state["paint_pipeline"],
            state["albedo_views"],
            state["mr_views"],
            cfg,
            save_dir=save_dir,
        )

    # Compute coverage %
    coverage_pct = "—"
    try:
        import numpy as np
        arr = mask_alb.squeeze(-1) if hasattr(mask_alb, "squeeze") else mask_alb
        if hasattr(arr, "detach"):
            arr = arr.detach().float().cpu().numpy()
        else:
            arr = np.asarray(arr)
        pct = 100.0 * float((arr > 1e-8).sum()) / arr.size
        coverage_pct = f"{pct:.1f}%"
        log.metric("texel_coverage", coverage_pct)
    except Exception:
        pass

    state["texture_albedo"] = tex_alb
    state["mask_albedo"]    = mask_alb
    state["texture_mr"]     = tex_mr
    state["mask_mr"]        = mask_mr

    atlas_b64 = _path_to_b64(save_dir / "baked_albedo.png")
    return {
        "stage":   STAGE_LABELS["bake_texture"],
        "type":    "atlas",
        "atlas":   atlas_b64,
        "metrics": {"coverage": coverage_pct},
    }, None


def _stage_inpaint_texture(
    state: dict,
    save_dir: Path,
    cfg,
    log,
) -> tuple[dict, Optional[Path]]:
    required = ("paint_pipeline", "texture_albedo", "mask_albedo", "texture_mr", "mask_mr")
    missing = [k for k in required if k not in state]
    if missing:
        raise RuntimeError(f"Bake Texture must run first. Missing: {missing}")

    from src.inpaint_texture import inpaint_textures

    with log.step("inpaint_textures"):
        ref_alb, ref_mr = inpaint_textures(
            state["paint_pipeline"],
            state["texture_albedo"],
            state["mask_albedo"],
            state["texture_mr"],
            state["mask_mr"],
            vertex_inpaint=cfg.vertex_inpaint,
            method=cfg.inpaint_method,
            save_dir=save_dir,
        )

    state["refined_albedo"] = ref_alb
    state["refined_mr"]     = ref_mr

    atlas_b64 = _path_to_b64(save_dir / "refined_albedo.png")
    return {
        "stage":   STAGE_LABELS["inpaint_texture"],
        "type":    "atlas",
        "atlas":   atlas_b64,
        "metrics": {},
    }, None


def _stage_export_glb(
    state: dict,
    save_dir: Path,
    cfg,
    log,
) -> tuple[dict, Optional[Path]]:
    required = ("paint_pipeline", "refined_albedo", "refined_mr", "uv_mesh")
    missing = [k for k in required if k not in state]
    if missing:
        raise RuntimeError(f"Inpaint Texture must run first. Missing: {missing}")

    from src.export_glb import export_textured_mesh

    fmt = state.get("output_format", "glb")
    out_path = save_dir / f"output.{fmt}"
    with log.step("export_textured_mesh"):
        result_path = export_textured_mesh(
            state["paint_pipeline"],
            state["refined_albedo"],
            state["refined_mr"],
            state["uv_mesh"],
            output_path=out_path,
            cleanup_obj=(fmt == "glb"),
            output_format=fmt,
        )

    size_mb = result_path.stat().st_size / 1024 ** 2 if result_path.exists() else 0
    log.metric("output_size_mb", f"{size_mb:.1f}")

    state["output_mesh"] = result_path

    return {
        "stage":    STAGE_LABELS["export_glb"],
        "type":     "mesh",
        "glb_path": result_path,
        "metrics":  {"format": fmt.upper(), "size": f"{size_mb:.1f} MB"},
    }, result_path


_RUNNERS = {
    "remove_background":   _stage_remove_background,
    "canonical_multiview": _stage_canonical_multiview,
    "preprocess":          _stage_preprocess,
    "mesh_generate":       _stage_mesh_generate,
    "mesh_postprocess":    _stage_mesh_postprocess,
    "render_multiview":    _stage_render_multiview,
    "paint_multiview":     _stage_paint_multiview,
    "bake_texture":        _stage_bake_texture,
    "inpaint_texture":     _stage_inpaint_texture,
    "export_glb":          _stage_export_glb,
}


# ─── Auto-scan callback ───────────────────────────────────────────────────────

def on_folder_change(folder_str: str) -> str:
    """Scan the input folder and return a short status markdown string."""
    if not folder_str.strip():
        return "_Enter an input folder path above._"
    folder = Path(folder_str.strip())
    if not folder.is_dir():
        return f"⚠ Folder not found: `{folder}`"
    mode, paths = scan_views(folder_str)
    if mode == "none":
        return f"⚠ No images found in `{folder.name}`"
    if mode == "multiview":
        names = ", ".join(f"`{p.name}`" for p in paths.values())
        return f"✓ **4 images (multiview)**: {names}"
    # single
    return f"✓ **1 image (single-view)**: `{next(iter(paths.values())).name}`"


# ─── Pipeline generator ───────────────────────────────────────────────────────

def run_pipeline(
    test_name: str,
    input_folder: str,
    selected_stages: list[str],
    output_format: str = "glb",
    # --- remove_background params ---
    remove_bg: bool = True,
    center_subject: bool = True,
    preprocess_target_size: Optional[int] = None,
    # --- canonical_multiview params ---
    canonical_steps: int = 50,
    canonical_guidance: float = 3.0,
    canonical_ref_scale: float = 1.0,
    canonical_use_depth: bool = True,
    canonical_depth_scale: float = 0.5,
    canonical_use_canny: bool = False,
    canonical_canny_scale: float = 0.2,
    canonical_prompt: str = "high quality",
):
    """
    Gradio generator: yields (html_table, mesh_path_or_None) after each stage.
    """
    # ---- Validation ---------------------------------------------------------
    test_name = (test_name or "test").strip().replace(" ", "_") or "test"
    if not input_folder.strip():
        yield "<p style='color:#f38ba8'>⚠ Please enter an input folder.</p>", None
        return

    mode, view_paths = scan_views(input_folder)
    if mode == "none":
        yield (
            "<p style='color:#f38ba8'>⚠ No images found in the specified folder.</p>",
            None,
        )
        return

    if not selected_stages:
        yield "<p style='color:#f38ba8'>⚠ Select at least one pipeline stage.</p>", None
        return

    # ---- Setup output directory + RunLogger ---------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = _PROJECT_ROOT / "outputs" / "test" / test_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts.test_utils import RunLogger
    log = RunLogger(out_dir, phase=f"UI:{test_name}")

    # ---- Device / config ----------------------------------------------------
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    from src.config import PipelineConfig
    cfg = PipelineConfig.for_runpod() if has_cuda else PipelineConfig.for_macos_dev()

    # ---- Build input row ----------------------------------------------------
    n_views   = len(view_paths)
    view_keys = list(view_paths.keys())

    input_b64s = [_path_to_b64(view_paths[k]) for k in view_keys]
    rows: list[dict] = [
        {
            "stage":   "Input",
            "type":    "input",
            "views":   input_b64s,
            "metrics": {
                "mode":   mode,
                "views":  str(n_views),
                "output": str(out_dir.relative_to(_PROJECT_ROOT)),
            },
        }
    ]

    latest_mesh: Optional[str] = None
    yield build_table_html(rows, n_views), latest_mesh

    # ---- Apply canonical multiview config to cfg ----------------------------
    cfg.canonical.steps                        = canonical_steps
    cfg.canonical.guidance_scale               = canonical_guidance
    cfg.canonical.reference_conditioning_scale = canonical_ref_scale
    cfg.canonical.use_depth                    = canonical_use_depth
    cfg.canonical.depth_scale                  = canonical_depth_scale
    cfg.canonical.use_canny                    = canonical_use_canny
    cfg.canonical.canny_scale                  = canonical_canny_scale
    cfg.canonical.prompt                       = canonical_prompt

    # ---- Pipeline state accumulated across stages ---------------------------
    from src.mesh_io import MeshSaver
    state: dict = {
        "view_paths":              view_paths,
        "mode":                    mode,
        "n_views":                 n_views,
        "output_format":           output_format,
        "mesh_saver":              MeshSaver(output_format),
        "remove_bg":               remove_bg,
        "center_subject":          center_subject,
        "preprocess_target_size":  preprocess_target_size,
    }

    # ---- Run selected stages in order ---------------------------------------
    _pipeline_t0 = time.perf_counter()
    for stage_id in STAGE_IDS:
        if stage_id not in selected_stages:
            continue

        stage_num = STAGE_IDS.index(stage_id) + 1
        stage_dir = out_dir / f"{stage_num:02d}_{stage_id}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Show "running" placeholder while stage executes
        rows.append({
            "stage":   STAGE_LABELS[stage_id],
            "type":    "running",
            "running": True,
            "metrics": {},
        })
        yield build_table_html(rows, n_views), latest_mesh

        t0 = time.perf_counter()
        try:
            runner = _RUNNERS[stage_id]
            row_data, new_mesh = runner(state, stage_dir, cfg, log)
            elapsed = time.perf_counter() - t0
            if isinstance(row_data, list):
                # Runner returned multiple sub-step rows (e.g. preprocess).
                # Stamp elapsed time on the last row and replace the placeholder.
                if row_data:
                    row_data[-1].setdefault("metrics", {})["elapsed"] = f"{elapsed:.1f} s"
                rows[-1:] = row_data
            else:
                row_data.setdefault("metrics", {})["elapsed"] = f"{elapsed:.1f} s"
                rows[-1] = row_data
            if new_mesh and Path(new_mesh).exists():
                latest_mesh = str(new_mesh)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.error("Stage %s failed: %s", stage_id, err_msg)
            traceback.print_exc()
            rows[-1] = {
                "stage":   STAGE_LABELS[stage_id],
                "type":    "error",
                "error":   err_msg,
                "metrics": {"elapsed": f"{elapsed:.1f} s"},
            }

        yield build_table_html(rows, n_views), latest_mesh

    log.save()

    # ---- RunPod cost summary row ----------------------------------------
    try:
        from src.cost import cost_summary_from_steps

        pipeline_elapsed_s = time.perf_counter() - _pipeline_t0
        cost = cost_summary_from_steps(
            steps=log._steps,
            total_s=pipeline_elapsed_s,
        )
        cost_display = {
            "GPU":                    cost["gpu_label"],
            "Run cost (flex)":        f"${cost['run_cost_flex']:.4f}",
            "Run cost (active)":      f"${cost['run_cost_active']:.4f}",
            "Monthly always-live":    f"${cost['monthly_always_live']:.2f}/mo",
            f"Monthly on-demand ({cost['monthly_on_demand_runs']} runs)":
                                      f"${cost['monthly_on_demand']:.2f}/mo",
            "Cold start assumed":     f"{cost['cold_start_s']:.0f} s",
            "Model load (measured)":  f"{cost['model_load_s']:.1f} s",
            "Execution (measured)":   f"{cost['execution_s']:.1f} s",
        }
        rows.append({
            "stage": "Cost Summary",
            "type":  "cost",
            "cost":  cost_display,
        })
        yield build_table_html(rows, n_views), latest_mesh
    except Exception:
        pass  # never block the UI on cost errors


# ─── Load-run helpers ─────────────────────────────────────────────────────────

def _list_runs() -> list[tuple[str, str]]:
    """Scan outputs/test/ and return sorted (label, abs_path) pairs, newest first."""
    root = _PROJECT_ROOT / "outputs" / "test"
    if not root.exists():
        return []
    choices: list[tuple[str, str]] = []
    for test_dir in sorted(root.iterdir()):
        if not test_dir.is_dir():
            continue
        for ts_dir in sorted(test_dir.iterdir(), reverse=True):
            if ts_dir.is_dir():
                label = f"{test_dir.name}  /  {ts_dir.name}"
                choices.append((label, str(ts_dir)))
    return choices


def _detect_n_views(stage_dirs: list[Path]) -> int:
    """Infer view count from orientation-named PNG files in stage dirs."""
    found: set[str] = set()
    for d in stage_dirs:
        for f in d.glob("*.png"):
            stem = f.stem.lower()
            for v in ("front", "left", "right", "back"):
                if stem.endswith(f"_{v}"):
                    found.add(v)
    return len(found) if found else 1


def _load_view_images(
    stage_dir: Path, view_keys: list[str], prefixes: list[str]
) -> list[Optional[str]]:
    """
    Load per-view images from stage_dir, trying each prefix in order.
    Falls back to any file ending in ``_{view}.png``.
    Returns a list of base64 data-URIs (or None) aligned to view_keys.
    """
    result: list[Optional[str]] = []
    for view in view_keys:
        b64 = None
        for prefix in prefixes:
            candidate = stage_dir / f"{prefix}_{view}.png"
            if candidate.exists():
                b64 = _path_to_b64(candidate)
                break
        if b64 is None:
            candidates = sorted(stage_dir.glob(f"*_{view}.png"))
            if candidates:
                b64 = _path_to_b64(candidates[0])
        result.append(b64)
    return result


def load_run(run_path_str: str) -> tuple[str, Optional[str]]:
    """
    Reconstruct the results table from a saved run directory.
    Returns (html_table, glb_path_or_None) — same signature as run_pipeline yields.
    """
    if not run_path_str:
        return "<p style='color:#f38ba8'>⚠ No run selected.</p>", None

    run_dir = Path(run_path_str)
    if not run_dir.is_dir():
        return f"<p style='color:#f38ba8'>⚠ Directory not found: {run_path_str}</p>", None

    # Read metrics.json for run-level info
    import json as _json
    metrics_data: dict = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics_data = _json.loads(metrics_path.read_text())
        except Exception:
            pass

    # Discover numbered stage dirs  (e.g. 01_preprocess, 02_mesh_generate)
    stage_dirs: list[Path] = sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and len(d.name) > 2 and d.name[:2].isdigit()
    )

    _VIEW_KEYS = ["front", "left", "right", "back"]
    n_views = _detect_n_views(stage_dirs)
    view_keys = _VIEW_KEYS[:n_views] if n_views <= 4 else _VIEW_KEYS

    # ── Input row: pull raw images from first stage dir ───────────────────────
    input_b64s: list[Optional[str]] = []
    if stage_dirs:
        input_b64s = _load_view_images(stage_dirs[0], view_keys, ["raw"])
    if not any(input_b64s):
        input_b64s = [None] * n_views

    run_meta: dict[str, str] = {}
    if metrics_data:
        run_meta["phase"] = metrics_data.get("phase", "")
        ts = metrics_data.get("timestamp", "")
        if ts:
            run_meta["timestamp"] = ts[:19].replace("T", " ")
        total_s = metrics_data.get("total_elapsed_s")
        if total_s is not None:
            run_meta["total"] = f"{total_s:.1f} s"
    run_meta["output"] = str(run_dir.relative_to(_PROJECT_ROOT))

    rows: list[dict] = [{
        "stage":   "Input",
        "type":    "input",
        "views":   input_b64s,
        "metrics": run_meta,
    }]

    latest_glb: Optional[str] = None

    for stage_dir in stage_dirs:
        parts = stage_dir.name.split("_", 1)
        if len(parts) < 2:
            continue
        stage_id = parts[1]
        label = STAGE_LABELS.get(stage_id, stage_id.replace("_", " ").title())

        glb_files = sorted(stage_dir.glob("*.glb"))
        row: dict = {"stage": label, "metrics": {}}

        if glb_files:
            # Prefer output.glb > postprocessed > uv > first found
            glb = glb_files[0]
            for pref in ("output", "postprocessed", "uv"):
                preferred = [g for g in glb_files if pref in g.stem.lower()]
                if preferred:
                    glb = preferred[0]
                    break
            row.update({"type": "mesh", "glb_path": glb})
            latest_glb = str(glb)

        elif stage_id == "bake_texture":
            atlas_path = stage_dir / "baked_albedo.png"
            row.update({
                "type":  "atlas",
                "atlas": _path_to_b64(atlas_path) if atlas_path.exists() else None,
            })

        elif stage_id == "inpaint_texture":
            atlas_path = stage_dir / "refined_albedo.png"
            row.update({
                "type":  "atlas",
                "atlas": _path_to_b64(atlas_path) if atlas_path.exists() else None,
            })

        elif stage_id == "paint_multiview":
            candidates = sorted(stage_dir.glob("albedo_upscaled_*.png"))
            atlas_path = candidates[0] if candidates else None
            row.update({
                "type":  "atlas",
                "atlas": _path_to_b64(atlas_path) if atlas_path else None,
            })

        elif stage_id == "render_multiview":
            normals = sorted(stage_dir.glob("normal_*.png"))[:n_views]
            b64s: list[Optional[str]] = [_path_to_b64(p) for p in normals]
            b64s += [None] * (n_views - len(b64s))
            row.update({"type": "images", "views": b64s})

        else:
            # Image stages: prefer most-processed variant
            b64s = _load_view_images(
                stage_dir, view_keys, ["resized", "rgba", "centered", "raw"]
            )
            row.update({"type": "images", "views": b64s})

        rows.append(row)

    # Prefer the final export GLB for the 3-D viewer
    for stage_dir in stage_dirs:
        if "export_glb" in stage_dir.name:
            candidate = stage_dir / "output.glb"
            if candidate.exists():
                latest_glb = str(candidate)
            break

    return build_table_html(rows, n_views), latest_glb


# ─── Gradio layout ────────────────────────────────────────────────────────────

def _build_ui() -> gr.Blocks:
    with gr.Blocks(title="3D Pipeline Tester", theme=gr.themes.Base()) as demo:
        gr.Markdown("# 3D Pipeline Tester")
        gr.Markdown(
            "Run any subset of the Hunyuan3D multiview pipeline stages "
            "and inspect the outputs after each step."
        )

        # ── Section 1: Test name ─────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### Test Name")
            test_name_box = gr.Textbox(
                label="Test name",
                placeholder="e.g. bbc_plata_run1",
                info="Combined with a timestamp to create the output folder.",
            )

        # ── Section 2: Dataset & Stages ──────────────────────────────────────
        with gr.Group():
            gr.Markdown("### Dataset & Stages")

            input_folder_box = gr.Textbox(
                label="Input folder",
                placeholder="e.g. inputs/bbc_plata",
                info="Folder containing 1 image (single-view) or 4 images "
                     "named *-front, *-left, *-right, *-back (multiview).",
            )
            scan_info = gr.Markdown("_Enter an input folder path above._")

            stage_checkboxes = gr.CheckboxGroup(
                choices=[(STAGE_LABELS[s], s) for s in STAGE_IDS],
                value=_DEFAULT_STAGE_IDS,
                label="Pipeline stages",
                info="Stages run in dependency order. "
                     "GPU-required stages will be skipped automatically on CPU. "
                     "Canonical Multiview is opt-in (requires ~14 GB VRAM).",
            )

            with gr.Accordion("Remove Background Options", open=False):
                gr.Markdown(
                    "Controls applied during the **Remove Background** stage. "
                    "Each enabled step is shown as a separate row in the results table."
                )
                with gr.Row():
                    remove_bg_chk = gr.Checkbox(
                        label="Remove background (rembg)",
                        value=True,
                        info="Strip the background using rembg (hy3dshape or standard library fallback).",
                    )
                    center_subject_chk = gr.Checkbox(
                        label="Center and pad subject",
                        value=True,
                        info="Crop to subject bounding box, extend to square, add 10% margin.",
                    )
                with gr.Row():
                    resize_chk = gr.Checkbox(
                        label="Resize to canonical size",
                        value=False,
                        info="Resize to the target size after centering (518 px = DINOv2-Giant encoder).",
                    )
                    resize_size_num = gr.Number(
                        label="Target size (px)",
                        value=518,
                        minimum=64,
                        maximum=2048,
                        step=1,
                        info="Only used when 'Resize to canonical size' is checked.",
                    )

            with gr.Accordion("Canonical Multiview (MV-Adapter)", open=False):
                gr.Markdown(
                    "**Optional stage** — uses MV-Adapter i2mv SDXL to generate consistent "
                    "`front / right / back / left` canonical views from a single anchor photo. "
                    "Requires ~14 GB VRAM. Enable by checking the stage above. "
                    "The canonical views replace HunyuanPaint albedo for higher texture fidelity."
                )
                with gr.Row():
                    canon_steps_num = gr.Number(
                        label="Steps",
                        value=50,
                        minimum=1,
                        maximum=100,
                        step=1,
                        info="Diffusion steps (50 = quality; 8–12 = preview).",
                    )
                    canon_guidance_num = gr.Number(
                        label="Guidance scale",
                        value=3.0,
                        minimum=1.0,
                        maximum=10.0,
                        step=0.1,
                        info="CFG guidance scale (default: 3.0).",
                    )
                    canon_ref_scale_num = gr.Number(
                        label="Reference conditioning scale",
                        value=1.0,
                        minimum=0.1,
                        maximum=2.0,
                        step=0.05,
                        info="How strongly the anchor photo guides generation (default: 1.0).",
                    )
                with gr.Row():
                    canon_use_depth_chk = gr.Checkbox(
                        label="Depth control (DPT-midas)",
                        value=True,
                        info="Adds depth ControlNet for structural stability.",
                    )
                    canon_depth_scale_num = gr.Number(
                        label="Depth scale",
                        value=0.5,
                        minimum=0.0,
                        maximum=1.5,
                        step=0.05,
                        info="ControlNet conditioning scale for depth (0.4–0.6 recommended).",
                    )
                with gr.Row():
                    canon_use_canny_chk = gr.Checkbox(
                        label="Canny edge control",
                        value=False,
                        info="Adds weak canny ControlNet for silhouette edges.",
                    )
                    canon_canny_scale_num = gr.Number(
                        label="Canny scale",
                        value=0.2,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        info="ControlNet conditioning scale for canny (0.15–0.3 recommended).",
                    )
                canon_prompt_box = gr.Textbox(
                    label="Generation prompt",
                    value="high quality",
                    placeholder="e.g. high quality industrial marine ship",
                    info="Text prompt passed to SDXL during canonical view generation.",
                )

            with gr.Row():
                output_format_dropdown = gr.Dropdown(
                    choices=[("GLB — binary glTF with PBR materials", "glb"),
                             ("OBJ — Wavefront mesh + JPEG textures", "obj")],
                    value="glb",
                    label="Output format",
                    info="Applies to the Export GLB stage.",
                    scale=2,
                )
                run_btn = gr.Button("▶ Run Pipeline", variant="primary", scale=1)

        # ── Section 3: Load Previous Run ─────────────────────────────────────
        with gr.Group():
            gr.Markdown("### Load Previous Run")
            with gr.Row():
                run_selector = gr.Dropdown(
                    choices=_list_runs(),
                    label="Select run",
                    info="Runs from outputs/test/, newest first. Hit Refresh after new runs complete.",
                    scale=4,
                )
                refresh_runs_btn = gr.Button("🔄 Refresh", scale=1)
                load_run_btn = gr.Button("📂 Load Run", variant="secondary", scale=1)

        # ── Section 4: Results ───────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### Results")
            with gr.Row():
                model3d_viewer = gr.Model3D(
                    label="3D Viewer — updates after each mesh stage",
                    height=480,
                    display_mode="solid",
                    clear_color=(0.12, 0.12, 0.16, 1.0),
                    camera_position=(None, None, None),
                    zoom_speed=1.2,
                    interactive=False,
                )
            results_html = gr.HTML(
                value="<p style='color:#585b70'>Results will appear here after running.</p>"
            )

        # ── Events ───────────────────────────────────────────────────────────
        input_folder_box.change(
            fn=on_folder_change,
            inputs=[input_folder_box],
            outputs=[scan_info],
        )

        def _run_pipeline_ui(
            test_name,
            input_folder,
            selected_stages,
            output_format,
            remove_bg,
            center_subject,
            resize_enabled,
            resize_size,
            canon_steps,
            canon_guidance,
            canon_ref_scale,
            canon_use_depth,
            canon_depth_scale,
            canon_use_canny,
            canon_canny_scale,
            canon_prompt,
        ):
            target_size = int(resize_size) if resize_enabled else None
            yield from run_pipeline(
                test_name=test_name,
                input_folder=input_folder,
                selected_stages=selected_stages,
                output_format=output_format,
                remove_bg=remove_bg,
                center_subject=center_subject,
                preprocess_target_size=target_size,
                canonical_steps=int(canon_steps),
                canonical_guidance=float(canon_guidance),
                canonical_ref_scale=float(canon_ref_scale),
                canonical_use_depth=canon_use_depth,
                canonical_depth_scale=float(canon_depth_scale),
                canonical_use_canny=canon_use_canny,
                canonical_canny_scale=float(canon_canny_scale),
                canonical_prompt=canon_prompt,
            )

        run_btn.click(
            fn=_run_pipeline_ui,
            inputs=[
                test_name_box,
                input_folder_box,
                stage_checkboxes,
                output_format_dropdown,
                remove_bg_chk,
                center_subject_chk,
                resize_chk,
                resize_size_num,
                canon_steps_num,
                canon_guidance_num,
                canon_ref_scale_num,
                canon_use_depth_chk,
                canon_depth_scale_num,
                canon_use_canny_chk,
                canon_canny_scale_num,
                canon_prompt_box,
            ],
            outputs=[results_html, model3d_viewer],
        )

        refresh_runs_btn.click(
            fn=lambda: gr.update(choices=_list_runs()),
            outputs=[run_selector],
        )

        load_run_btn.click(
            fn=load_run,
            inputs=[run_selector],
            outputs=[results_html, model3d_viewer],
        )

    return demo


if __name__ == "__main__":
    demo = _build_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
