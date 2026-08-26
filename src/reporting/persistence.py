"""Seasonal Hotspot and Spatial Persistence Calculator for Crop Stress Intelligence Kenya.

Quantifies the spatial recurrence frequency of actionable diagnostic screening signals
across all sequential 14-day production bins without introducing arbitrary weighting models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from src.diagnostics.screening import ACTIONABLE_CASES, DiagnosticCase


def calculate_spatial_persistence(
    diagnostic_grids: Sequence[xr.DataArray],
    actionable_cases: Sequence[int] = (
        DiagnosticCase.CASE_A.value,
        DiagnosticCase.CASE_B.value,
        DiagnosticCase.CASE_D.value,
        DiagnosticCase.MULTI_SIGNAL.value,
    ),
) -> xr.Dataset:
    """Calculate the recurrence count and frequency of actionable signals across sequential bins.

    Parameters
    ----------
    diagnostic_grids : sequence of xr.DataArray
        List of 2D diagnostic classification DataArrays across sequential canonical bins.
    actionable_cases : sequence of int, optional
        Integer enum values treated as actionable signals. Defaults to Cases A, B, D, and MULTI_SIGNAL.

    Returns
    -------
    xr.Dataset
        xarray Dataset containing:
        - `actionable_recurrence_count`: Total number of bins each pixel exhibited an actionable signal.
        - `actionable_recurrence_freq`: Recurrence fraction in [0.0, 1.0] relative to valid observed bins.
        - `case_b_recurrence_count`: Recurrence count specifically for Case B.
        - `case_d_recurrence_count`: Recurrence count specifically for Case D.
        - `multi_recurrence_count`: Recurrence count specifically for MULTI_SIGNAL.
        - `valid_observation_count`: Total number of bins with evaluable (non-insufficient) state.
    """
    if len(diagnostic_grids) == 0:
        raise ValueError("Cannot calculate persistence from an empty sequence of diagnostic grids.")

    # Validate matching dimensions and coordinates
    ref = diagnostic_grids[0]
    shape = ref.shape

    for i, grid in enumerate(diagnostic_grids[1:], 1):
        if grid.shape != shape:
            raise ValueError(
                f"Grid dimension mismatch at index {i}: expected {shape}, got {grid.shape}."
            )

    n_bins = len(diagnostic_grids)

    # Initialize count arrays
    total_actionable_count = np.zeros(shape, dtype=np.int32)
    case_a_count = np.zeros(shape, dtype=np.int32)
    case_b_count = np.zeros(shape, dtype=np.int32)
    case_d_count = np.zeros(shape, dtype=np.int32)
    multi_count = np.zeros(shape, dtype=np.int32)
    valid_count = np.zeros(shape, dtype=np.int32)

    for grid in diagnostic_grids:
        vals = grid.values

        # Valid mask (not insufficient evidence and not NaN)
        is_valid = np.isfinite(vals) & (vals != DiagnosticCase.INSUFFICIENT_EVIDENCE.value)
        valid_count += is_valid.astype(np.int32)

        # Actionable matches
        is_actionable = np.isin(vals, actionable_cases)
        total_actionable_count += is_actionable.astype(np.int32)

        # Case-specific counts
        case_a_count += (vals == DiagnosticCase.CASE_A.value).astype(np.int32)
        case_b_count += (vals == DiagnosticCase.CASE_B.value).astype(np.int32)
        case_d_count += (vals == DiagnosticCase.CASE_D.value).astype(np.int32)
        multi_count += (vals == DiagnosticCase.MULTI_SIGNAL.value).astype(np.int32)

    # Recurrence frequency relative to valid observed bins
    with np.errstate(divide="ignore", invalid="ignore"):
        recurrence_freq = np.where(valid_count > 0, total_actionable_count / valid_count, 0.0).astype(np.float32)

    coords = dict(ref.coords)
    dims = ref.dims

    ds_persist = xr.Dataset(
        data_vars={
            "actionable_recurrence_count": (dims, total_actionable_count, {
                "long_name": "Actionable Screening Signal Recurrence Count",
                "units": "bins",
                "valid_min": 0,
                "valid_max": n_bins,
                "description": f"Number of bins (out of {n_bins}) exhibiting actionable screening flags",
            }),
            "actionable_recurrence_freq": (dims, recurrence_freq, {
                "long_name": "Actionable Screening Recurrence Frequency",
                "units": "fraction",
                "valid_min": 0.0,
                "valid_max": 1.0,
                "description": "Fraction of valid observed bins exhibiting actionable screening flags",
            }),
            "case_a_recurrence_count": (dims, case_a_count, {
                "long_name": "Case A Drought Stress Recurrence Count",
                "units": "bins",
            }),
            "case_b_recurrence_count": (dims, case_b_count, {
                "long_name": "Case B Non-Dry Anomaly Recurrence Count",
                "units": "bins",
            }),
            "case_d_recurrence_count": (dims, case_d_count, {
                "long_name": "Case D Hydrological Disconnect Recurrence Count",
                "units": "bins",
            }),
            "multi_recurrence_count": (dims, multi_count, {
                "long_name": "MULTI_SIGNAL Recurrence Count",
                "units": "bins",
            }),
            "valid_observation_count": (dims, valid_count, {
                "long_name": "Valid Observation Count",
                "units": "bins",
                "description": f"Number of bins (out of {n_bins}) with valid, non-insufficient data",
            }),
        },
        coords=coords,
        attrs={
            "title": "Seasonal Spatial Persistence & Recurrence Summary",
            "n_canonical_bins": n_bins,
            "crs": ref.rio.crs.to_string() if hasattr(ref, "rio") and ref.rio.crs else "EPSG:6933",
        },
    )

    if hasattr(ref, "rio") and ref.rio.crs:
        ds_persist = ds_persist.rio.write_crs(ref.rio.crs)

    return ds_persist
