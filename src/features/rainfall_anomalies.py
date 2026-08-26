"""Precipitation Climatology & Standardized Anomaly Engine.

Computes 30-year CHIRPS monthly baseline statistics (mean, std) and evaluates
continuous standardized precipitation anomalies (Z-scores) on native EPSG:4326 grids.

Zero-Variance Protection:
    Invariant baseline pixels (sigma <= std_epsilon) are masked to NaN rather than
    inflating anomalies with an arbitrary epsilon term in the denominator.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr

from src.utils.exceptions import DataCompletenessError


def calculate_rainfall_climatology(
    ds_chirps_monthly: xr.Dataset,
    baseline_years: Tuple[int, int] = (1991, 2020),
    std_epsilon: float = 1e-4,
) -> Tuple[xr.Dataset, xr.Dataset, xr.DataArray]:
    """Compute 30-year monthly precipitation climatology baseline.

    Operates on a pre-loaded monthly CHIRPS dataset, verifies 360-month continuity
    over the baseline period, and computes monthly mean and standard deviation.

    Parameters
    ----------
    ds_chirps_monthly : xr.Dataset
        Monthly CHIRPS precipitation dataset containing 'rainfall' or 'precip'.
    baseline_years : tuple of int, default (1991, 2020)
        Start and end year of the canonical 30-year climatological baseline.
    std_epsilon : float, default 1e-4
        Minimum standard deviation threshold for valid variance.

    Returns
    -------
    clim_mean : xr.Dataset
        Monthly mean precipitation climatology (12 calendar months).
    clim_std : xr.Dataset
        Monthly standard deviation climatology (12 calendar months).
    valid_std_mask : xr.DataArray
        Boolean mask indicating valid non-zero baseline variance (sigma > std_epsilon).

    Raises
    ------
    DataCompletenessError
        If the baseline time series is missing months, contains duplicates, or lacks
        continuous monthly coverage across all expected baseline dates.
    ValueError
        If required precipitation variables are not found in `ds_chirps_monthly`.
    """
    start_year, end_year = baseline_years
    expected_dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")
    expected_count = len(expected_dates)

    if "time" not in ds_chirps_monthly.coords and "time" not in ds_chirps_monthly.dims:
        raise DataCompletenessError("Input CHIRPS dataset lacks 'time' dimension.")

    # Select requested baseline slice
    start_ts = pd.Timestamp(f"{start_year}-01-01")
    end_ts = pd.Timestamp(f"{end_year}-12-31")

    all_times = pd.to_datetime(ds_chirps_monthly["time"].values)
    mask = (all_times >= start_ts) & (all_times <= end_ts)
    if not np.any(mask):
        raise DataCompletenessError(
            f"CHIRPS baseline {start_year}-{end_year} is incomplete: no observations found in baseline window."
        )

    ds_baseline = ds_chirps_monthly.isel(time=np.where(mask)[0])

    # Validate monotonicity and continuity on the baseline slice
    time_index = pd.DatetimeIndex(ds_baseline.time.values)
    if not time_index.is_monotonic_increasing:
        raise DataCompletenessError(
            f"CHIRPS baseline {start_year}-{end_year} is incomplete or non-continuous: "
            "baseline time index is not monotonically increasing."
        )

    # Normalize to start of month for matching
    time_month_starts = pd.DatetimeIndex([pd.Timestamp(t.year, t.month, 1) for t in time_index])

    if len(time_month_starts) != expected_count or not time_month_starts.equals(expected_dates):
        missing = set(expected_dates) - set(time_month_starts)
        duplicates = len(time_month_starts) - len(set(time_month_starts))
        raise DataCompletenessError(
            f"CHIRPS baseline {start_year}-{end_year} is incomplete or non-continuous: expected {expected_count} continuous months, "
            f"found {len(time_month_starts)} (missing: {len(missing)}, duplicates: {duplicates})."
        )

    # Determine canonical variable
    var_name = "rainfall" if "rainfall" in ds_baseline.data_vars else (
        "precip" if "precip" in ds_baseline.data_vars else list(ds_baseline.data_vars)[0]
    )

    # Compute monthly mean and std
    clim_mean = ds_baseline.groupby("time.month").mean(dim="time")
    clim_std = ds_baseline.groupby("time.month").std(dim="time")

    # Zero-variance protection mask
    valid_std_mask = clim_std[var_name] > std_epsilon
    valid_std_mask.name = "valid_std_mask"
    valid_std_mask.attrs.update({
        "standard_name": "valid_variance_mask",
        "long_name": "Valid Baseline Standard Deviation Mask",
        "std_epsilon": std_epsilon,
        "invariant_pixel_count": int((~valid_std_mask).sum().values),
    })

    clim_mean.attrs["climatology_baseline"] = f"{start_year}-{end_year}"
    clim_std.attrs["climatology_baseline"] = f"{start_year}-{end_year}"

    return clim_mean, clim_std, valid_std_mask


def calculate_rainfall_zscore(
    ds_target: xr.Dataset,
    clim_mean: xr.Dataset,
    clim_std: xr.Dataset,
    std_epsilon: float = 1e-4,
) -> xr.Dataset:
    """Calculate continuous standardized precipitation anomaly (Z-score).

    Z_R = (R_{x, m} - mu_{R, m}) / sigma_{R, m}

    Parameters
    ----------
    ds_target : xr.Dataset
        Target period CHIRPS monthly precipitation dataset.
    clim_mean : xr.Dataset
        Monthly mean precipitation climatology (grouped by month).
    clim_std : xr.Dataset
        Monthly standard deviation precipitation climatology (grouped by month).
    std_epsilon : float, default 1e-4
        Minimum standard deviation threshold for valid variance.

    Returns
    -------
    xr.Dataset
        Continuous Z-score anomaly dataset on native EPSG:4326 grid with variable 'z_rainfall'.
    """
    var_name = "rainfall" if "rainfall" in ds_target.data_vars else (
        "precip" if "precip" in ds_target.data_vars else list(ds_target.data_vars)[0]
    )

    target_da = ds_target[var_name]
    mean_da = clim_mean[var_name]
    std_da = clim_std[var_name]

    # Align baseline statistics to target time steps by calendar month
    target_months = target_da["time.month"]
    mean_matched = mean_da.sel(month=target_months).drop_vars("month")
    mean_matched["time"] = target_da["time"]
    std_matched = std_da.sel(month=target_months).drop_vars("month")
    std_matched["time"] = target_da["time"]

    # Zero-variance protection: sigma > std_epsilon
    valid_variance = std_matched > std_epsilon

    # Standardized Z-score calculation
    diff = target_da - mean_matched
    z_da = (diff / std_matched).where(valid_variance, np.nan)
    z_da.name = "z_rainfall"
    z_da.attrs.update({
        "standard_name": "precipitation_z_score",
        "long_name": "Standardized Precipitation Anomaly (Z-score)",
        "units": "dimensionless",
        "std_epsilon": std_epsilon,
    })

    res_ds = xr.Dataset(data_vars={"z_rainfall": z_da}, coords=target_da.coords, attrs=ds_target.attrs)
    res_ds.attrs["anomaly_type"] = "standardized_z_score"
    return res_ds
