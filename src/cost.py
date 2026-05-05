"""
RunPod cost estimation for the 3D pipeline.

Calculates per-run cost, estimated monthly cost for on-demand (Flex) workers,
and estimated monthly cost for always-live (Active) workers, taking into account
pod cold-start time, model load time, and the idle timeout period.

Pricing source: https://docs.runpod.io/serverless/pricing (as of May 2026)

Usage:
    from src.cost import RunPodCostEstimator

    estimator = RunPodCostEstimator(model_load_s=45.0)
    summary = estimator.cost_summary(execution_s=120.0, runs_per_month=500)
    print(summary)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------
# Keyed by the RUNPOD_GPU_SIZE environment variable value.
# flex_per_s : Flex (on-demand) worker rate in USD per second
# active_per_s: Active (always-live) worker rate in USD per second

@dataclass(frozen=True)
class _GpuTierPricing:
    label: str          # Human-readable GPU description
    vram_gb: int        # Approximate VRAM in GB (used for fallback detection)
    flex_per_s: float   # Flex worker rate (USD/s)
    active_per_s: float # Active worker rate (USD/s)


_PRICING: dict[str, _GpuTierPricing] = {
    "AMPERE_16": _GpuTierPricing("A4000 / A4500 / RTX 4000 (16 GB)",  16, 0.00016, 0.00011),
    "AMPERE_24": _GpuTierPricing("A5000 / 3090 / L4 (24 GB)",         24, 0.00019, 0.00013),
    "ADA_24":    _GpuTierPricing("4090 PRO (24 GB)",                   24, 0.00031, 0.00021),
    "AMPERE_48": _GpuTierPricing("A6000 / A40 (48 GB)",                48, 0.00034, 0.00024),
    "ADA_48":    _GpuTierPricing("L40 / L40S / 6000 Ada PRO (48 GB)", 48, 0.00053, 0.00037),
    "AMPERE_80": _GpuTierPricing("A100 (80 GB)",                       80, 0.00076, 0.00060),
    "HOPPER_80": _GpuTierPricing("H100 PRO (80 GB)",                   80, 0.00116, 0.00093),
    "HOPPER_141":_GpuTierPricing("H200 PRO (141 GB)",                 141, 0.00155, 0.00124),
    "BLACKWELL_180": _GpuTierPricing("B200 (180 GB)",                 180, 0.00240, 0.00190),
}

_SECONDS_PER_MONTH = 30 * 24 * 3600  # 2,592,000 s


# ---------------------------------------------------------------------------
# GPU tier auto-detection helpers
# ---------------------------------------------------------------------------

def _detect_tier_from_env() -> Optional[str]:
    """Read RUNPOD_GPU_SIZE and return the matching tier key, or None."""
    gpu_size = os.environ.get("RUNPOD_GPU_SIZE", "").upper().strip()
    if gpu_size and gpu_size in _PRICING:
        return gpu_size
    return None


def _detect_tier_from_vram() -> Optional[str]:
    """
    Try to read PyTorch VRAM and pick the cheapest tier whose vram_gb
    is >= the detected VRAM, or None if PyTorch/CUDA is unavailable.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        return None

    # Pick the smallest tier whose vram_gb covers the detected size
    candidates = sorted(
        ((k, v) for k, v in _PRICING.items() if v.vram_gb >= int(vram_gb)),
        key=lambda kv: kv[1].vram_gb,
    )
    if candidates:
        return candidates[0][0]
    # Detected VRAM is larger than all known tiers — pick the largest
    return max(_PRICING.keys(), key=lambda k: _PRICING[k].vram_gb)


def _resolve_tier(gpu_tier: Optional[str]) -> str:
    """Return the best available tier key, falling back to AMPERE_48."""
    if gpu_tier is not None:
        if gpu_tier in _PRICING:
            return gpu_tier
        logger.warning("Unknown gpu_tier %r — falling back to auto-detection.", gpu_tier)

    tier = _detect_tier_from_env()
    if tier:
        return tier

    tier = _detect_tier_from_vram()
    if tier:
        return tier

    logger.warning(
        "Could not detect RunPod GPU tier. Defaulting to AMPERE_48 (A6000/A40 48 GB). "
        "Set RUNPOD_GPU_SIZE env var or pass gpu_tier explicitly for accurate estimates."
    )
    return "AMPERE_48"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RunPodCostEstimator:
    """
    Estimates RunPod serverless compute costs for a 3D pipeline run.

    Three billed phases per job (Flex worker):
      1. Cold start  — container init, from worker start to job receipt
      2. Model load  — loading weights into VRAM (part of execution from RunPod's
                       perspective but separable here for analysis)
      3. Execution   — active pipeline processing
      4. Idle timeout — worker stays live after a job before scaling down

    Active (always-live) workers skip cold start and run at a discounted rate.

    Args:
        gpu_tier:       RunPod GPU size key (e.g. "AMPERE_48"). None → auto-detect.
        cold_start_s:   Container cold-start duration in seconds. Overridable via
                        RUNPOD_COLD_START_S env var. Default: 60 s.
        model_load_s:   Model-load duration in seconds (summed from RunLogger steps
                        whose names contain "load"). Default: 0.0.
        idle_timeout_s: Worker idle timeout after a job. Overridable via
                        RUNPOD_IDLE_TIMEOUT_S env var. Default: 5 s.
    """

    def __init__(
        self,
        gpu_tier: Optional[str] = None,
        cold_start_s: float = 60.0,
        model_load_s: float = 0.0,
        idle_timeout_s: float = 5.0,
    ) -> None:
        # Allow env-var overrides for timing constants
        self.cold_start_s = float(
            os.environ.get("RUNPOD_COLD_START_S", cold_start_s)
        )
        self.idle_timeout_s = float(
            os.environ.get("RUNPOD_IDLE_TIMEOUT_S", idle_timeout_s)
        )
        self.model_load_s = model_load_s

        self.gpu_tier = _resolve_tier(gpu_tier)
        self._pricing = _PRICING[self.gpu_tier]

    # ------------------------------------------------------------------
    # Per-run costs
    # ------------------------------------------------------------------

    def run_cost_flex(self, execution_s: float) -> float:
        """
        Cost for one job on a Flex (on-demand) worker.

        Billed duration = cold_start + model_load + execution + idle_timeout.
        All billed at the flex rate.
        """
        billed_s = self.cold_start_s + self.model_load_s + execution_s + self.idle_timeout_s
        return billed_s * self._pricing.flex_per_s

    def run_cost_active(self, execution_s: float) -> float:
        """
        Marginal cost for one job on an Active (always-live) worker.

        No cold start. Billed duration = model_load + execution + idle_timeout
        at the active (discounted) rate.

        Note: The fixed 24/7 base cost is captured in monthly_always_live().
        """
        billed_s = self.model_load_s + execution_s + self.idle_timeout_s
        return billed_s * self._pricing.active_per_s

    # ------------------------------------------------------------------
    # Monthly costs
    # ------------------------------------------------------------------

    def monthly_always_live(self) -> float:
        """
        Monthly cost of a single Active worker running 24/7 for 30 days.

        This is the fixed floor cost regardless of request volume.
        """
        return _SECONDS_PER_MONTH * self._pricing.active_per_s

    def monthly_on_demand(self, execution_s: float, runs_per_month: int) -> float:
        """
        Estimated monthly cost for Flex (on-demand) workers.

        Each run incurs the full overhead: cold start + model load + execution
        + idle timeout. Total = runs_per_month × run_cost_flex(execution_s).
        """
        return runs_per_month * self.run_cost_flex(execution_s)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def cost_summary(
        self,
        execution_s: float,
        runs_per_month: int = 500,
    ) -> dict:
        """
        Return a dict with all cost figures and the assumptions used.

        Args:
            execution_s:     Pure pipeline execution time (excluding model load).
            runs_per_month:  Assumed request volume for the on-demand monthly estimate.

        Returns:
            {
              "gpu_tier": str,
              "gpu_label": str,
              "flex_rate_per_s": float,
              "active_rate_per_s": float,
              "cold_start_s": float,
              "model_load_s": float,
              "execution_s": float,
              "idle_timeout_s": float,
              "run_cost_flex": float,
              "run_cost_active": float,
              "monthly_always_live": float,
              "monthly_on_demand": float,
              "monthly_on_demand_runs": int,
            }
        """
        return {
            "gpu_tier":               self.gpu_tier,
            "gpu_label":              self._pricing.label,
            "flex_rate_per_s":        self._pricing.flex_per_s,
            "active_rate_per_s":      self._pricing.active_per_s,
            "cold_start_s":           self.cold_start_s,
            "model_load_s":           round(self.model_load_s, 3),
            "execution_s":            round(execution_s, 3),
            "idle_timeout_s":         self.idle_timeout_s,
            "run_cost_flex":          round(self.run_cost_flex(execution_s), 6),
            "run_cost_active":        round(self.run_cost_active(execution_s), 6),
            "monthly_always_live":    round(self.monthly_always_live(), 4),
            "monthly_on_demand":      round(self.monthly_on_demand(execution_s, runs_per_month), 4),
            "monthly_on_demand_runs": runs_per_month,
        }


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def is_on_runpod() -> bool:
    """Return True when running inside a RunPod worker."""
    return bool(
        os.environ.get("RUNPOD_POD_ID")
        or os.environ.get("RUNPOD_GPU_SIZE")
        or os.environ.get("RUNPOD_ENDPOINT_ID")
    )


def cost_summary_from_steps(
    steps: list[dict],
    total_s: float,
    runs_per_month: int = 500,
    gpu_tier: Optional[str] = None,
) -> dict:
    """
    Convenience function that extracts model_load_s from RunLogger step records
    and returns a full cost_summary dict.

    Args:
        steps:          RunLogger._steps list (each has "name" and "elapsed_s").
        total_s:        Total pipeline wall time in seconds.
        runs_per_month: Assumed monthly request volume.
        gpu_tier:       Optional explicit GPU tier override.
    """
    model_load_s = sum(
        s["elapsed_s"] for s in steps if "load" in s.get("name", "").lower()
    )
    execution_s = max(total_s - model_load_s, 0.0)
    estimator = RunPodCostEstimator(
        gpu_tier=gpu_tier,
        model_load_s=model_load_s,
    )
    return estimator.cost_summary(execution_s=execution_s, runs_per_month=runs_per_month)
