"""
src — Hunyuan3D Multiview Pipeline package.

On import this module inserts the required third-party directories into
sys.path so that hy3dshape, hy3dpaint, hy3dgen, DifferentiableRenderer,
hunyuanpaintpbr, and other subpackages are importable without installation.

Path layout (official Hunyuan3D-2.1 repo uses nested package structure):

  third_party/Hunyuan3D-2.1/              ← repo root
  third_party/Hunyuan3D-2.1/hy3dshape/   ← outer dir; inner hy3dshape/ is the Python pkg
  third_party/Hunyuan3D-2.1/hy3dpaint/   ← paint modules live here directly
  third_party/Hunyuan3D-2/               ← v2.0 repo (Delight utility via hy3dgen)

All four are prepended to sys.path only once even if src is imported multiple times.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THIRD_PARTY = _PROJECT_ROOT / "third_party"

_RUNTIME_PATHS: list[str] = [
    # Repo root (top-level scripts / any modules at root level)
    str(_THIRD_PARTY / "Hunyuan3D-2.1"),
    # hy3dshape uses a nested package layout: hy3dshape/hy3dshape/__init__.py
    # Adding the outer hy3dshape/ dir makes `import hy3dshape` resolve to the inner package.
    str(_THIRD_PARTY / "Hunyuan3D-2.1" / "hy3dshape"),
    # Paint modules (DifferentiableRenderer, hunyuanpaintpbr, utils, convert_utils, …)
    # all live directly inside hy3dpaint/ without an extra nesting level.
    str(_THIRD_PARTY / "Hunyuan3D-2.1" / "hy3dpaint"),
    # Hunyuan3D-2.0 repo — provides hy3dgen (Delight utility)
    str(_THIRD_PARTY / "Hunyuan3D-2"),
]

for _p in reversed(_RUNTIME_PATHS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

del _p  # clean up loop variable from module namespace
