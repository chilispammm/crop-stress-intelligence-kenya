"""Optical data quality assessment and compositing diagnostics.

Computes observation accounting metrics, SCL cloud rejection rates, clear-sky spatial
percentages, and composite data density across the target analysis window.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd
import xarray as xr


def generate_quality_report(
    raw_ds: xr.Dataset,
    masked_ds: xr.Dataset,
    composite_da: xr.DataArray,
    valid_obs_count: xr.DataArray,
) -> Dict[str, Any]:
    """Generate quality and observation diagnostics matrix for optical preprocessing.

    Parameters
    ----------
    raw_ds : xr.Dataset
        Raw Sentinel-2 dataset prior to SCL masking.
    masked_ds : xr.Dataset
        Sentinel-2 dataset after applying SCL quality mask.
    composite_da : xr.DataArray
        14-day aggregated biophysical composite array.
    valid_obs_count : xr.DataArray
        Array tracking count of clear observations contributing to each composite cell.

    Returns
    -------
    dict
        Comprehensive diagnostics report containing:
        - acquisition_dates: List of satellite acquisition timestamps (ISO strings).
        - clear_pixel_pct_per_date: Dict mapping date string to spatial clear-pixel percentage.
        - rejection_rate_scl: Total percentage of pixels removed by SCL masking.
        - mean_valid_obs_per_bin: Average count of clear observations per composite cell.
        - unfilled_bins_pct: Percentage of composite cells that remain NaN.
    """
    # 1. Acquisition dates
    if "time" in raw_ds.coords:
        timestamps = pd.to_datetime(raw_ds["time"].values)
        acq_dates = [ts.strftime("%Y-%m-%d") for ts in timestamps]
    else:
        acq_dates = []

    # Pick a representative optical band to compute pixel stats
    rep_band = "B04" if "B04" in raw_ds.data_vars else list(raw_ds.data_vars)[0]
    raw_band = raw_ds[rep_band]
    masked_band = masked_ds[rep_band]

    # 2. Clear pixel percentage per scene
    clear_pct_by_date: Dict[str, float] = {}
    if "time" in raw_band.dims:
        for t_idx, date_str in enumerate(acq_dates):
            scene_raw = raw_band.isel(time=t_idx)
            scene_masked = masked_band.isel(time=t_idx)

            total_valid_input = np.sum(np.isfinite(scene_raw.values))
            total_retained = np.sum(np.isfinite(scene_masked.values))

            pct = (total_retained / total_valid_input * 100.0) if total_valid_input > 0 else 0.0
            clear_pct_by_date[date_str] = round(float(pct), 2)

    # 3. Overall SCL rejection rate
    total_raw_valid = np.sum(np.isfinite(raw_band.values))
    total_masked_valid = np.sum(np.isfinite(masked_band.values))
    rejection_rate_scl = (
        ((total_raw_valid - total_masked_valid) / total_raw_valid * 100.0)
        if total_raw_valid > 0
        else 0.0
    )

    # 4. Mean valid observations per composite bin
    obs_vals = valid_obs_count.values.flatten()
    mean_obs = float(np.mean(obs_vals)) if len(obs_vals) > 0 else 0.0

    # 5. Percentage of unfilled (NaN) composite bins
    total_comp_pixels = int(composite_da.size)
    nan_comp_pixels = int(np.sum(np.isnan(composite_da.values)))
    unfilled_pct = (nan_comp_pixels / total_comp_pixels * 100.0) if total_comp_pixels > 0 else 0.0

    return {
        "acquisition_dates": acq_dates,
        "scene_count": len(acq_dates),
        "clear_pixel_pct_per_date": clear_pct_by_date,
        "rejection_rate_scl": round(float(rejection_rate_scl), 2),
        "mean_valid_obs_per_bin": round(mean_obs, 2),
        "unfilled_bins_pct": round(float(unfilled_pct), 2),
    }
