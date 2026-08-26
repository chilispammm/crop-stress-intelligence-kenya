"""Temporal compositing and missingness-preserving rolling smoothing.

Compositing Strategy:
    14-day median aggregation reduces cloud contamination and gaps while preserving
    vegetation phenology during the Long Rains season.
    The exact count of valid (non-null) pixel-time observations is tracked per cell.

Smoothing Policy:
    Rolling smoothing must NEVER invent data over empty bins.
    Missing bins (all observations masked by SCL/clouds) remain NaN after smoothing.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import xarray as xr


def composite_14d(
    da: xr.DataArray,
    freq: str = "14D",
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Aggregate masked observations into 14-day median composite bins and track observation counts.

    Parameters
    ----------
    da : xr.DataArray
        Time-series DataArray (e.g. cloud-masked NDVI, EVI, or NDMI).
    freq : str, default '14D'
        Resampling frequency interval.

    Returns
    -------
    tuple of (xr.DataArray, xr.DataArray)
        - composite_da: 14-day median composite DataArray.
        - valid_obs_count: Exact count of non-null clear pixel observations contributing to each cell.
    """
    if "time" not in da.coords and "time" not in da.dims:
        raise ValueError("Input DataArray must contain a 'time' coordinate for temporal compositing.")

    # 1. Median composite
    composite_da = da.resample(time=freq).median(dim="time")
    composite_da.name = f"{da.name}_composite" if da.name else "composite"
    composite_da.attrs.update(da.attrs)
    composite_da.attrs.update({
        "aggregation_method": "median",
        "compositing_frequency": freq,
    })

    # 2. Exact observation counting: sum of non-null pixels in the bin
    valid_obs_count = da.notnull().astype(int).resample(time=freq).sum(dim="time")
    valid_obs_count.name = "valid_obs_count"
    valid_obs_count.attrs.update({
        "standard_name": "valid_observation_count",
        "long_name": "Count of Valid Clear-Sky Observations",
        "units": "count",
        "compositing_frequency": freq,
    })

    return composite_da, valid_obs_count


def smooth_temporal_series(
    da_composite: xr.DataArray,
    window: int = 3,
    min_periods: int = 1,
) -> xr.DataArray:
    """Apply centered rolling window smoothing while strictly preserving original missingness.

    Formula:
        smoothed = da_composite.rolling(time=window, min_periods=min_periods, center=True).mean()
        smoothed = smoothed.where(da_composite.notnull(), np.nan)

    Parameters
    ----------
    da_composite : xr.DataArray
        Input composite time-series DataArray.
    window : int, default 3
        Size of the moving window (number of time steps).
    min_periods : int, default 1
        Minimum number of observations in window required to have a value.

    Returns
    -------
    xr.DataArray
        Smoothed DataArray where originally empty bins strictly remain NaN.
    """
    if "time" not in da_composite.coords and "time" not in da_composite.dims:
        raise ValueError("Input DataArray must contain a 'time' coordinate for rolling smoothing.")

    # Apply rolling centered mean
    rolling_smoothed = da_composite.rolling(
        time=window,
        min_periods=min_periods,
        center=True,
    ).mean()

    # Enforce strict missingness preservation: empty bins must remain NaN
    smoothed_clean = rolling_smoothed.where(da_composite.notnull(), np.nan)
    smoothed_clean.name = f"{da_composite.name}_smoothed" if da_composite.name else "smoothed"
    smoothed_clean.attrs.update(da_composite.attrs)
    smoothed_clean.attrs.update({
        "smoothing_applied": True,
        "smoothing_window": window,
        "smoothing_method": "rolling_mean_centered",
        "missingness_preservation": "strict_nan_retention",
    })

    return smoothed_clean
