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
    "preprocess",
    "mesh_generate",
    "mesh_postprocess",
    "render_multiview",
    "paint_multiview",
    "bake_texture",
    "inpaint_texture",
    "export_glb",
]

STAGE_LABELS = {
    "preprocess":       "Preprocess",
    "mesh_generate":    "Mesh Generate",
    "mesh_postprocess": "Mesh Postprocess",
    "render_multiview": "Render Multiview",
    "paint_multiview":  "Paint Multiview",
    "bake_texture":     "Bake Texture",
    "inpaint_texture":  "Inpaint Texture",
    "export_glb":       "Export GLB",
}

GPU_REQUIRED = {
    "mesh_generate", "mesh_postprocess",
    "render_multiview", "paint_multiview",
    "bake_texture", "export_glb",
}

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

def _stage_preprocess(
    state: dict,
    save_dir: Path,
    cfg,
    log,
) -> tuple[list[dict], Optional[Path]]:
    from src.preprocess import collect_views, preprocess_all_views, compose_over_gray

    view_paths: dict[str, Path] = state["view_paths"]
    view_keys  = list(view_paths.keys())
    n_cols     = state["n_views"]

    remove_bg      = state.get("remove_bg", True)
    center_subject = state.get("center_subject", True)
    target_size    = state.get("preprocess_target_size")

    with log.step("collect_views"):
        raw_views = collect_views(**{k: str(v) for k, v in view_paths.items()})

    with log.step("preprocess_all_views"):
        processed = preprocess_all_views(
            raw_views,
            remove_bg=remove_bg,
            center_subject=center_subject,
            target_size=target_size,
        )

    # Persist into pipeline state for downstream stages.
    state["processed"]   = processed
    state["raw_views"]   = raw_views
    state["shape_views"] = {k: v["shape"] for k, v in processed.items()}
    state["paint_views"] = {k: v["paint"] for k, v in processed.items()}
    log.metric("views_preprocessed", len(processed))

    def _row_b64s(key: str) -> list[Optional[str]]:
        """Extract base64 thumbnails for a given intermediate key across all views."""
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

    def _row_b64s_rgba_on_gray(key: str) -> list[Optional[str]]:
        """Like _row_b64s but composites the RGBA intermediate over gray for display."""
        b64s: list[Optional[str]] = []
        for orient in view_keys:
            entry = processed.get(orient)
            if entry and key in entry:
                img = entry[key]
                display_img = compose_over_gray(img) if img.mode == "RGBA" else img
                b64s.append(_pil_to_b64(display_img))
            else:
                b64s.append(None)
        while len(b64s) < n_cols:
            b64s.append(None)
        return b64s[:n_cols]

    # Save all intermediates to disk.
    for orient in view_keys:
        entry = processed.get(orient, {})
        for key, img in entry.items():
            fname = save_dir / f"{key}_{orient}.png"
            img.save(str(fname))

    rows: list[dict] = []

    # Row: raw input (already in state as input row, but useful in context).
    rows.append({
        "stage":   "Pre: Raw",
        "type":    "images",
        "views":   _row_b64s_rgba_on_gray("raw"),
        "metrics": {"step": "loaded RGBA"},
    })

    # Row: after background removal (only if remove_bg was enabled).
    if remove_bg:
        rows.append({
            "stage":   "Pre: Rembg",
            "type":    "images",
            "views":   _row_b64s_rgba_on_gray("rgba"),
            "metrics": {"step": "background removed"},
        })

    # Row: after center+pad (only if enabled).
    if center_subject:
        rows.append({
            "stage":   "Pre: Center+Pad",
            "type":    "images",
            "views":   _row_b64s_rgba_on_gray("centered"),
            "metrics": {"step": "centered & padded"},
        })

    # Row: after resize (only if a target size was set).
    if target_size is not None:
        rows.append({
            "stage":   f"Pre: Resize→{target_size}",
            "type":    "images",
            "views":   _row_b64s_rgba_on_gray("resized"),
            "metrics": {"step": f"→ {target_size}×{target_size} px"},
        })

    # Row: paint reference (gray composite — what the paint model sees).
    rows.append({
        "stage":   "Pre: Paint ref",
        "type":    "images",
        "views":   _row_b64s("paint"),
        "metrics": {"step": "gray composite (paint)"},
    })

    # Row: shape reference (white composite — what the shape DiT sees).
    rows.append({
        "stage":   "Pre: Shape ref",
        "type":    "images",
        "views":   _row_b64s("shape"),
        "metrics": {"step": "white composite (shape)"},
    })

    return rows, None


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

    glb_path = save_dir / "mesh_raw.glb"
    raw_mesh.export(str(glb_path))
    state["raw_mesh"] = raw_mesh

    return {
        "stage":    STAGE_LABELS["mesh_generate"],
        "type":     "mesh",
        "glb_path": glb_path,
        "metrics": {
            "verts": f"{len(raw_mesh.vertices):,}",
            "faces": f"{len(raw_mesh.faces):,}",
        },
    }, glb_path


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

    glb_path = save_dir / "mesh_postprocessed.glb"
    post_mesh.export(str(glb_path))
    state["post_mesh"] = post_mesh

    return {
        "stage":    STAGE_LABELS["mesh_postprocess"],
        "type":     "mesh",
        "glb_path": glb_path,
        "metrics": {
            "verts": f"{len(post_mesh.vertices):,}",
            "faces": f"{len(post_mesh.faces):,}",
        },
    }, glb_path


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

    uv_mesh.export(str(save_dir / "mesh_uv.glb"))

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

    uv_glb_path = save_dir / "mesh_uv.glb"

    return {
        "stage":   STAGE_LABELS["render_multiview"],
        "type":    "images",
        "views":   view_b64s,
        "metrics": {"normal_maps": str(len(normal_maps))},
    }, uv_glb_path


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

    paint_pipeline = state["paint_pipeline"]
    front_path     = state["view_paths"]["front"]

    with log.step("delight_reference"):
        front_rgba = load_image_rgba(front_path)
        reference  = delight_reference(front_rgba, cfg)
        reference.save(str(save_dir / "reference_delighted.png"))

    with log.step("MultiviewDiffusionNet init"):
        mvd = MultiviewDiffusionNet(cfg)

    with log.step(f"paint diffusion (steps={cfg.paint_steps})"):
        paint_out   = mvd(reference, state["normal_maps"], state["position_maps"], cfg)
        albedo_views = paint_out["albedo"]
        mr_views     = paint_out["mr"]
        log.metric("albedo_views", len(albedo_views))
        log.metric("mr_views",     len(mr_views))

    with log.step("upscale_views"):
        albedo_up = upscale_views(albedo_views, target_size=cfg.render_size)
        mr_up     = upscale_views(mr_views,     target_size=cfg.render_size)

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
    "preprocess":       _stage_preprocess,
    "mesh_generate":    _stage_mesh_generate,
    "mesh_postprocess": _stage_mesh_postprocess,
    "render_multiview": _stage_render_multiview,
    "paint_multiview":  _stage_paint_multiview,
    "bake_texture":     _stage_bake_texture,
    "inpaint_texture":  _stage_inpaint_texture,
    "export_glb":       _stage_export_glb,
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
    remove_bg: bool = True,
    center_subject: bool = True,
    preprocess_target_size: Optional[int] = None,
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

    # ---- Pipeline state accumulated across stages ---------------------------
    state: dict = {
        "view_paths":              view_paths,
        "mode":                    mode,
        "n_views":                 n_views,
        "output_format":           output_format,
        "remove_bg":               remove_bg,
        "center_subject":          center_subject,
        "preprocess_target_size":  preprocess_target_size,
    }

    # ---- Run selected stages in order ---------------------------------------
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
                value=STAGE_IDS,
                label="Pipeline stages",
                info="Stages run in dependency order. "
                     "GPU-required stages will be skipped automatically on CPU.",
            )

            with gr.Accordion("Preprocessing Options", open=False):
                gr.Markdown(
                    "Controls applied during the **Preprocess** stage. "
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

        # ── Section 3: Results ───────────────────────────────────────────────
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
            ],
            outputs=[results_html, model3d_viewer],
        )

    return demo


if __name__ == "__main__":
    demo = _build_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
