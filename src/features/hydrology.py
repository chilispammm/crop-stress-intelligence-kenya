"""Soil Hydrology Dynamics Engine.

Resamples daily GRAFS topsoil (s0) and root-zone (s1) soil water index to match the
canonical 14-day optical compositing calendar, and computes 14-day root-zone dynamics (Delta SWI_{14d}).

Key Architectural Principles:
1. `target_time_bins` from the optical pipeline is the single authoritative temporal input.
2. Temporal bins are strictly half-open intervals: [t_i, t_{i+1}) with 14-day steps.
3. Differencing is evaluated exclusively on the resampled s1 (Root-Zone SWI) series:
   Delta SWI_{14d}(t) = s1(t) - s1(t-1), with t=0 assigned NaN.
4. Topsoil moisture (s0) is retained in the resampled dataset for profile analysis.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
import xarray as xr


def resample_soil_moisture_to_calendar(
    ds_grafs: xr.Dataset,
    target_time_bins: Union[xr.DataArray, Sequence[Union[str, pd.Timestamp]]],
) -> xr.Dataset:
    """Resample daily GRAFS soil moisture to match optical 14-day time bins.

    Aggregates daily observations into half-open intervals [t_i, t_i + 14 days)
    matching `target_time_bins`.

    Parameters
    ----------
    ds_grafs : xr.Dataset
        Daily GRAFS dataset containing 's0' (topsoil) and 's1' (root-zone).
    target_time_bins : xr.DataArray or sequence of pd.Timestamp / str
        Authoritative bin start timestamps from the optical compositing pipeline.

    Returns
    -------
    xr.Dataset
        Resampled GRAFS dataset with exact temporal alignment to `target_time_bins`.

    Raises
    ------
    ValueError
        If target_time_bins is empty, non-unique, non-monotonic, or contains
        intervals that deviate from the 14-day step.
    """
    if isinstance(target_time_bins, xr.DataArray):
        bin_timestamps = pd.to_datetime(target_time_bins.values)
    else:
        bin_timestamps = pd.to_datetime(list(target_time_bins))

    if len(bin_timestamps) == 0:
        raise ValueError("target_time_bins cannot be empty.")

    # Validate uniqueness and monotonic order
    dt_index = pd.DatetimeIndex(bin_timestamps)
    if not dt_index.is_monotonic_increasing:
        raise ValueError("target_time_bins must be strictly monotonically increasing.")
    if len(dt_index) != len(set(dt_index)):
        raise ValueError("target_time_bins must contain unique timestamps.")

    # Validate 14-day interval spacing between adjacent bins
    if len(dt_index) > 1:
        diffs = dt_index[1:] - dt_index[:-1]
        for idx, d in enumerate(diffs):
            if d != pd.Timedelta(days=14):
                raise ValueError(
                    f"Adjacent target_time_bins at index {idx} ({dt_index[idx]} -> {dt_index[idx+1]}) "
                    f"has interval {d}, expected exactly 14 days."
                )

    bin_slices = []
    # Evaluate half-open intervals [t_start, t_end)
    for t_start in dt_index:
        t_start_ts = pd.Timestamp(t_start)
        t_end_ts = t_start_ts + pd.Timedelta(days=14)

        # Boolean mask for half-open interval [t_start, t_end)
        time_vals = pd.to_datetime(ds_grafs["time"].values)
        mask = (time_vals >= t_start_ts) & (time_vals < t_end_ts)

        if np.any(mask):
            ds_window = ds_grafs.isel(time=np.where(mask)[0])
            slice_mean = ds_window.mean(dim="time", skipna=True)
        else:
            # If no observations fall within window, fill with NaN
            slice_mean = ds_grafs.isel(time=0).drop_vars("time")
            for var in slice_mean.data_vars:
                slice_mean[var] = xr.full_like(slice_mean[var], np.nan)

        bin_slices.append(slice_mean)

    # Concat along time dimension
    time_idx = pd.Index(dt_index, name="time")
    res_ds = xr.concat(bin_slices, dim=time_idx)
    res_ds.attrs.update(ds_grafs.attrs)
    res_ds.attrs["temporal_resampling"] = "14-day half-open mean aggregation [t, t+14d)"

    # Retain variable metadata
    if "s0" in res_ds:
        res_ds["s0"].attrs.update({
            "standard_name": "topsoil_relative_wetness_14d",
            "long_name": "14-day Mean Topsoil Relative Wetness (0-5 cm)",
            "units": "dimensionless",
        })
    if "s1" in res_ds:
        res_ds["s1"].attrs.update({
            "standard_name": "root_zone_swi_14d",
            "long_name": "14-day Mean Root-Zone Soil Water Index (0-100 cm)",
            "units": "dimensionless",
        })

    return res_ds


def calculate_swi_14d_change(
    ds_grafs_14d: xr.Dataset,
) -> xr.Dataset:
    """Calculate 14-day Root-Zone Soil Water Index change metric (Delta SWI_{14d}).

    Delta SWI_{14d}(t) = s1(t) - s1(t-1)

    Evaluates signed change strictly on root-zone moisture (s1). Topsoil moisture (s0)
    is preserved if present in the dataset, but is never used for differencing.
    The first time step (t=0) evaluates strictly to NaN.

    Parameters
    ----------
    ds_grafs_14d : xr.Dataset
        14-day resampled GRAFS dataset containing 's1'.

    Returns
    -------
    xr.Dataset
        Dataset containing 'delta_swi_14d' (and original 's0', 's1' profiles) on the
        native 0.1 degree GRAFS grid (EPSG:4326).
    """
    if "s1" not in ds_grafs_14d.data_vars:
        raise ValueError("Input dataset must contain 's1' (root-zone soil water index).")

    s1 = ds_grafs_14d["s1"]
    time_len = ds_grafs_14d.sizes.get("time", 0)

    if time_len <= 1:
        delta_swi = xr.full_like(s1, np.nan)
    else:
        diff_da = s1.diff(dim="time")
        initial_nan = xr.full_like(s1.isel(time=slice(0, 1)), np.nan)
        delta_swi = xr.concat([initial_nan, diff_da], dim="time")

    delta_swi.name = "delta_swi_14d"
    delta_swi.attrs.update({
        "standard_name": "root_zone_swi_14d_change",
        "long_name": "14-day Root-Zone Soil Water Index Change (Delta SWI_{14d})",
        "units": "dimensionless_fraction",
        "formula": "s1(t) - s1(t-1)",
    })

    res_ds = ds_grafs_14d.copy()
    res_ds["delta_swi_14d"] = delta_swi
    res_ds.attrs["hydrology_metric"] = "delta_swi_14d"
    return res_ds
