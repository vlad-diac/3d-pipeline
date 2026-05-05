"""
Shared utilities for phase test scripts.

RunLogger — records step durations, VRAM deltas, and arbitrary metrics.
Writes two files to the run's output directory at the end:

  metrics.log  — human-readable table (also printed live to stdout)
  metrics.json — machine-readable dict for programmatic comparison

Usage:
    from scripts.test_utils import RunLogger

    log = RunLogger(OUTPUT_DIR, phase="Phase 3")

    with log.step("load_shape_pipeline"):
        pipeline = load_shape_pipeline_auto(cfg)

    log.metric("raw_vertices", len(raw_mesh.vertices))
    log.metric("raw_faces",    len(raw_mesh.faces))
    log.metric("peak_vram_gb", torch.cuda.max_memory_allocated() / 1024**3, unit="GB")

    log.save()   # also called automatically via atexit
"""

from __future__ import annotations

import atexit
import contextlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional


# ---------------------------------------------------------------------------
# VRAM helper
# ---------------------------------------------------------------------------

def _vram_allocated_gb() -> Optional[float]:
    """Return currently allocated CUDA memory in GB, or None if CUDA unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 ** 3
    except ImportError:
        pass
    return None


def _vram_peak_gb() -> Optional[float]:
    """Return peak allocated CUDA memory since last reset, in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 3
    except ImportError:
        pass
    return None


def _reset_vram_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def resolve_seed(seed: int | None) -> int:
    """
    Return a seed to use for this run.

    If ``seed`` is None, a random 32-bit seed is generated so the run is
    still fully reproducible (the actual value is recorded in metrics).

    Args:
        seed: Value from --seed CLI arg, or None if the flag was omitted.

    Returns:
        The seed that should be used for this run.
    """
    import random
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    return seed


class RunLogger:
    """
    Records timing and metrics for a single test run.

    Args:
        output_dir: Directory where metrics.log and metrics.json are written.
        phase:      Human-readable phase name shown in the log header.
    """

    def __init__(self, output_dir: Path, phase: str = "Test") -> None:
        self.output_dir = Path(output_dir)
        self.phase = phase
        self.run_start = time.perf_counter()
        self.timestamp = datetime.now()

        self._steps: list[dict] = []          # ordered step records
        self._metrics: dict[str, Any] = {}    # named scalar metrics
        self._saved = False

        atexit.register(self._atexit_save)

    # ------------------------------------------------------------------
    # Step timer context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def step(self, name: str, tier: str = "") -> Iterator[None]:
        """
        Context manager that times a block and records the result.

        Args:
            name: Human-readable step name (shown in the log).
            tier: Optional tag shown in the log (e.g. "CPU", "GPU").

        Example:
            with log.step("load_shape_pipeline", tier="GPU"):
                pipeline = load_shape_pipeline_auto(cfg)
        """
        label = f"[{tier}] {name}" if tier else name
        vram_before = _vram_allocated_gb()
        _reset_vram_peak()

        t0 = time.perf_counter()
        print(f"  ↳ {label} ...", end="", flush=True)

        exc_info = None
        try:
            yield
        except Exception as e:
            exc_info = e
            raise
        finally:
            elapsed = time.perf_counter() - t0
            vram_after = _vram_allocated_gb()
            peak = _vram_peak_gb()

            record: dict = {
                "name": name,
                "tier": tier,
                "elapsed_s": round(elapsed, 3),
            }

            vram_str = ""
            if vram_before is not None and vram_after is not None:
                delta = vram_after - vram_before
                record["vram_before_gb"] = round(vram_before, 2)
                record["vram_after_gb"] = round(vram_after, 2)
                record["vram_delta_gb"] = round(delta, 2)
                if peak is not None:
                    record["vram_peak_gb"] = round(peak, 2)
                    vram_str = f"   VRAM {vram_before:.2f}→{vram_after:.2f} GB  peak {peak:.2f} GB"
                else:
                    vram_str = f"   VRAM {vram_before:.2f}→{vram_after:.2f} GB"

            if exc_info is not None:
                record["status"] = "FAILED"
                record["error"] = str(exc_info)
                status_str = "  FAILED"
            else:
                record["status"] = "OK"
                status_str = ""

            self._steps.append(record)
            print(f"\r  ✓ {label:<45}  {elapsed:7.3f} s{vram_str}{status_str}")

    # ------------------------------------------------------------------
    # Metric recording
    # ------------------------------------------------------------------

    def metric(self, key: str, value: Any, *, unit: str = "", note: str = "") -> None:
        """
        Record a named scalar metric.

        Args:
            key:   Metric name (e.g. "raw_vertices", "peak_vram_gb").
            value: Numeric or string value.
            unit:  Optional unit suffix for display (e.g. "GB", "faces").
            note:  Optional annotation shown in the log.
        """
        self._metrics[key] = {"value": value, "unit": unit, "note": note}
        unit_str = f" {unit}" if unit else ""
        note_str = f"  ({note})" if note else ""
        print(f"      {key}: {value}{unit_str}{note_str}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write metrics.log and metrics.json to output_dir."""
        if self._saved:
            return
        self._saved = True

        total_s = time.perf_counter() - self.run_start
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---- build structured data ----------------------------------------
        data: dict[str, Any] = {
            "phase": self.phase,
            "timestamp": self.timestamp.isoformat(),
            "total_elapsed_s": round(total_s, 3),
            "steps": self._steps,
            "metrics": {k: v["value"] for k, v in self._metrics.items()},
            "metrics_full": self._metrics,
        }

        # ---- RunPod cost estimate (only when running on RunPod) -----------
        try:
            from src.cost import cost_summary_from_steps, is_on_runpod
            if is_on_runpod():
                data["cost"] = cost_summary_from_steps(
                    steps=self._steps,
                    total_s=total_s,
                )
        except Exception as _cost_err:
            pass  # never block metrics saving on cost errors

        # ---- JSON ----------------------------------------------------------
        json_path = self.output_dir / "metrics.json"
        json_path.write_text(json.dumps(data, indent=2))

        # ---- human-readable log --------------------------------------------
        lines: list[str] = []
        sep = "─" * 72

        lines.append(sep)
        lines.append(f"  {self.phase} Run — {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Output: {self.output_dir}")
        lines.append(sep)
        lines.append("")

        # steps table
        if self._steps:
            lines.append(f"  {'Step':<45}  {'Time':>8}  {'Status':<8}  VRAM (before→after, peak)")
            lines.append(f"  {'─'*45}  {'─'*8}  {'─'*8}  {'─'*30}")
            for s in self._steps:
                label = f"[{s['tier']}] {s['name']}" if s.get("tier") else s["name"]
                elapsed_str = f"{s['elapsed_s']:>7.3f} s"
                status = s.get("status", "")
                if "vram_before_gb" in s:
                    vram_str = (
                        f"{s['vram_before_gb']:.2f}→{s['vram_after_gb']:.2f} GB"
                        f"  peak {s.get('vram_peak_gb', 0):.2f} GB"
                    )
                else:
                    vram_str = "—"
                lines.append(f"  {label:<45}  {elapsed_str}  {status:<8}  {vram_str}")

        lines.append("")

        # metrics table
        if self._metrics:
            lines.append("  Metrics:")
            for key, entry in self._metrics.items():
                unit_str = f" {entry['unit']}" if entry.get("unit") else ""
                note_str = f"  # {entry['note']}" if entry.get("note") else ""
                lines.append(f"    {key:<40}  {entry['value']}{unit_str}{note_str}")
            lines.append("")

        lines.append(f"  Total time: {total_s:.3f} s")
        lines.append(sep)

        log_text = "\n".join(lines) + "\n"
        log_path = self.output_dir / "metrics.log"
        log_path.write_text(log_text)

        print(f"\n{'─'*60}")
        print(log_text, end="")
        print(f"  Metrics written to:")
        print(f"    {log_path}")
        print(f"    {json_path}")

    def _atexit_save(self) -> None:
        if not self._saved:
            self.save()
