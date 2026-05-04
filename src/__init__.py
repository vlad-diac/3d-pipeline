"""
src — Hunyuan3D Multiview Pipeline package.

On import this module inserts the two third-party runtime directories into
sys.path so that  hy3dshape,  hy3dpaint,  hy3dgen,  DifferentiableRenderer,
and other subpackages are importable without any global installation.

Import order matters:
  1. Hunyuan3D-2.1  (primary runtime — shape v2.1 + paint PBR v2.1)
  2. Hunyuan3D-2    (provides Delight utility from hy3dgen.texgen.utils)

Both entries are prepended only once even if src is imported multiple times.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THIRD_PARTY = _PROJECT_ROOT / "third_party"

_RUNTIME_PATHS: list[str] = [
    str(_THIRD_PARTY / "Hunyuan3D-2.1"),
    str(_THIRD_PARTY / "Hunyuan3D-2"),
]

for _p in reversed(_RUNTIME_PATHS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

del _p  # clean up loop variable from module namespace
