"""Resolution-Honest Vegetation Anomaly Engine.

Aligns high-resolution Sentinel-2 NDVI composites to the authoritative reference
climatology grid (EPSG:6933) using area-preserving valid-pixel averaging, and
computes continuous standardized NDVI anomalies (Z-scores).

Key Architectural Rules:
1. The reference climatology grid (reference_clim, native EPSG:6933) is the sole
   authoritative target grid definition.
2. Area-weighted aggregation:
   NDVI_out = (sum_i A_i * NDVI_i * I_i) / (sum_i A_i * I_i)
   where A_i is valid source-pixel overlap area within the destination cell,
   I_i = 1 for valid pixels, and I_i = 0 for invalid/NaN pixels.
   If sum_i A_i * I_i == 0, then NDVI_out = NaN.
3. Quality-filtered baseline: monthly pixels where count < min_valid_obs evaluate to NaN.
4. Invariant climatology pixels (sigma <= std_epsilon) evaluate to NaN.
"""

from __future__ import annotations

from typing import Optional, Union
import numpy as np
import pandas as pd
import rasterio
import rasterio.crs
import rasterio.warp
from rasterio.enums import Resampling
from rasterio.transform import from_origin
import rioxarray
import xarray as xr


def filter_ndvi_climatology_qa(
    ds_ndvi_clim: xr.Dataset,
    min_valid_obs: int = 20,
) -> xr.Dataset:
    """Filter monthly NDVI climatology statistics by minimum observation count.

    Masks monthly baseline mean and std pixels where observation count < min_valid_obs to NaN.

    Parameters
    ----------
    ds_ndvi_clim : xr.Dataset
        NDVI climatology dataset containing mean, stddev, and count variables or dimensions.
    min_valid_obs : int, default 20
        Minimum number of valid historical observations required to retain a baseline pixel.

    Returns
    -------
    xr.Dataset
        Quality-filtered NDVI climatology dataset with count < min_valid_obs masked to NaN.
    """
    ds_filtered = ds_ndvi_clim.copy(deep=True)

    # Schema A: monthly variables (e.g. mean_apr, stddev_apr, count_apr)
    month_codes = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    for m in month_codes:
        cnt_var = f"count_{m}"
        mean_var = f"mean_{m}"
        std_var = f"stddev_{m}" if f"stddev_{m}" in ds_filtered.data_vars else f"std_{m}"

        if cnt_var in ds_filtered.data_vars:
            cnt = ds_filtered[cnt_var]
            valid_mask = cnt >= min_valid_obs
            if mean_var in ds_filtered.data_vars:
                ds_filtered[mean_var] = ds_filtered[mean_var].where(valid_mask, np.nan)
            if std_var in ds_filtered.data_vars:
                ds_filtered[std_var] = ds_filtered[std_var].where(valid_mask, np.nan)

    # Schema B: dimension 'month' with variables 'mean', 'std'/'stddev', 'count'
    if "count" in ds_filtered.data_vars:
        cnt = ds_filtered["count"]
        valid_mask = cnt >= min_valid_obs
        for var in ("mean", "stddev", "std"):
            if var in ds_filtered.data_vars:
                ds_filtered[var] = ds_filtered[var].where(valid_mask, np.nan)

    ds_filtered.attrs["min_valid_obs_filter"] = min_valid_obs
    return ds_filtered


def aggregate_target_to_climatology_grid(
    target_ndvi: xr.DataArray,
    reference_clim: xr.DataArray,
) -> xr.DataArray:
    """Reproject and aggregate Sentinel-2 NDVI onto the authoritative climatology grid.

    Derives target affine transform, CRS (EPSG:6933), spatial coordinates, dimensions,
    and bounding extent directly from `reference_clim`. Uses true area-weighted average
    downsampling via `rasterio.warp.reproject(resampling=Resampling.average)`.
    Invalid/NaN source pixels contribute zero weight to both numerator and denominator;
    target cells containing zero valid contributing pixels evaluate to NaN.

    Parameters
    ----------
    target_ndvi : xr.DataArray
        Sentinel-2 NDVI composite in working projection (EPSG:32736).
    reference_clim : xr.DataArray
        Authoritative reference climatology DataArray in native EPSG:6933.

    Returns
    -------
    xr.DataArray
        Area-weighted aggregated NDVI composite matching the grid, CRS, coordinates,
        affine transform, and dimensions of `reference_clim`.
    """
    # 1. Derive destination grid metadata directly from reference_clim
    if hasattr(reference_clim, "rio") and reference_clim.rio.crs is not None:
        dst_crs = reference_clim.rio.crs
    elif "crs" in reference_clim.attrs:
        dst_crs = rasterio.crs.CRS.from_user_input(reference_clim.attrs["crs"])
    elif "spatial_ref" in reference_clim.coords or "spatial_ref" in reference_clim.attrs:
        val = reference_clim.coords.get("spatial_ref", reference_clim.attrs.get("spatial_ref"))
        dst_crs = rasterio.crs.CRS.from_user_input(val if val is not None else "EPSG:6933")
    else:
        raise ValueError("reference_clim must contain valid CRS metadata.")

    # Destination transform
    if hasattr(reference_clim, "rio") and reference_clim.rio.transform(recalc=False) is not None:
        dst_transform = reference_clim.rio.transform()
    else:
        x_coords = reference_clim["x"].values if "x" in reference_clim.coords else reference_clim["lon"].values
        y_coords = reference_clim["y"].values if "y" in reference_clim.coords else reference_clim["lat"].values
        dx = float(np.abs(x_coords[1] - x_coords[0])) if len(x_coords) > 1 else 30.0
        dy = float(np.abs(y_coords[1] - y_coords[0])) if len(y_coords) > 1 else 30.0
        min_x = float(np.min(x_coords)) - dx / 2.0
        max_y = float(np.max(y_coords)) + dy / 2.0
        dst_transform = from_origin(min_x, max_y, dx, dy)

    # 2. Derive source grid metadata from target_ndvi
    if hasattr(target_ndvi, "rio") and target_ndvi.rio.crs is not None:
        src_crs = target_ndvi.rio.crs
    elif "crs" in target_ndvi.attrs:
        src_crs = rasterio.crs.CRS.from_user_input(target_ndvi.attrs["crs"])
    else:
        src_crs = rasterio.crs.CRS.from_string("EPSG:32736")

    if hasattr(target_ndvi, "rio") and target_ndvi.rio.transform(recalc=False) is not None:
        src_transform = target_ndvi.rio.transform()
    else:
        x_coords = target_ndvi["x"].values if "x" in target_ndvi.coords else target_ndvi["lon"].values
        y_coords = target_ndvi["y"].values if "y" in target_ndvi.coords else target_ndvi["lat"].values
        dx = float(np.abs(x_coords[1] - x_coords[0])) if len(x_coords) > 1 else 10.0
        dy = float(np.abs(y_coords[1] - y_coords[0])) if len(y_coords) > 1 else 10.0
        min_x = float(np.min(x_coords)) - dx / 2.0
        max_y = float(np.max(y_coords)) + dy / 2.0
        src_transform = from_origin(min_x, max_y, dx, dy)

    # Spatial dimensions of destination
    y_dim = "y" if "y" in reference_clim.coords else ("lat" if "lat" in reference_clim.coords else reference_clim.dims[-2])
    x_dim = "x" if "x" in reference_clim.coords else ("lon" if "lon" in reference_clim.coords else reference_clim.dims[-1])
    dst_h = reference_clim.sizes[y_dim]
    dst_w = reference_clim.sizes[x_dim]

    # Handle 2D vs 3D (multi-temporal) arrays
    if target_ndvi.ndim == 2:
        src_arr = np.ascontiguousarray(target_ndvi.values, dtype=np.float32)
        dst_arr = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
        rasterio.warp.reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.average,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        res_coords = {y_dim: reference_clim[y_dim], x_dim: reference_clim[x_dim]}
        res_da = xr.DataArray(dst_arr, coords=res_coords, dims=[y_dim, x_dim], name=target_ndvi.name or "ndvi_aligned")
    else:
        time_dim = [d for d in target_ndvi.dims if d not in (y_dim, x_dim, "lat", "lon", "y", "x")][0]
        res_slices = []
        for i in range(target_ndvi.sizes[time_dim]):
            slice_src = np.ascontiguousarray(target_ndvi.isel({time_dim: i}).values, dtype=np.float32)
            dst_slice = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
            rasterio.warp.reproject(
                source=slice_src,
                destination=dst_slice,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.average,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            res_slices.append(dst_slice)
        res_arr = np.stack(res_slices, axis=0)
        res_coords = {time_dim: target_ndvi[time_dim], y_dim: reference_clim[y_dim], x_dim: reference_clim[x_dim]}
        res_da = xr.DataArray(res_arr, coords=res_coords, dims=[time_dim, y_dim, x_dim], name=target_ndvi.name or "ndvi_aligned")

    res_da = res_da.rio.write_crs(dst_crs)
    res_da = res_da.rio.write_transform(dst_transform)
    res_da = res_da.rio.write_nodata(np.nan)
    res_da.attrs.update(target_ndvi.attrs)
    res_da.attrs.update({
        "resampling_method": "area_weighted_average",
        "target_grid_crs": str(dst_crs),
        "target_grid_shape": (dst_h, dst_w),
    })
    return res_da


def calculate_ndvi_zscore(
    target_ndvi_aligned: xr.DataArray,
    clim_mean: xr.DataArray,
    clim_std: xr.DataArray,
    std_epsilon: float = 1e-4,
) -> xr.DataArray:
    """Calculate continuous standardized NDVI anomaly (Z-score).

    Z_{NDVI} = (NDVI_{aligned, m} - mu_{NDVI, m}) / sigma_{NDVI, m}

    Explicitly maps each 14-day optical composite to its corresponding calendar-month
    baseline statistics. Invariant baseline pixels (sigma <= std_epsilon) and QA-masked
    pixels (count < min_valid_obs) evaluate strictly to NaN.

    Parameters
    ----------
    target_ndvi_aligned : xr.DataArray
        NDVI composite aggregated onto the reference climatology grid (EPSG:6933).
    clim_mean : xr.DataArray
        Monthly mean NDVI climatology baseline.
    clim_std : xr.DataArray
        Monthly standard deviation NDVI climatology baseline.
    std_epsilon : float, default 1e-4
        Minimum standard deviation threshold for valid variance.

    Returns
    -------
    xr.DataArray
        Continuous Z_{NDVI} anomaly DataArray strictly on the climatology grid (EPSG:6933).
    """
    # Deterministic calendar-month mapping
    if "month" in clim_mean.dims and "time" in target_ndvi_aligned.dims:
        target_months = target_ndvi_aligned["time.month"].values
        mean_slices = [clim_mean.sel(month=m).drop_vars("month") for m in target_months]
        std_slices = [clim_std.sel(month=m).drop_vars("month") for m in target_months]
        mean_matched = xr.concat(mean_slices, dim=target_ndvi_aligned["time"])
        std_matched = xr.concat(std_slices, dim=target_ndvi_aligned["time"])
    else:
        mean_matched = clim_mean
        std_matched = clim_std

    # Zero-variance protection: sigma > std_epsilon
    valid_variance = std_matched > std_epsilon

    # Standardized Z-score calculation
    diff = target_ndvi_aligned - mean_matched
    z_ndvi = (diff / std_matched).where(valid_variance, np.nan)
    z_ndvi.name = "z_ndvi"
    z_ndvi.attrs.update(target_ndvi_aligned.attrs)
    z_ndvi.attrs.update({
        "standard_name": "vegetation_z_score",
        "long_name": "Standardized NDVI Anomaly (Z-score)",
        "units": "dimensionless",
        "std_epsilon": std_epsilon,
    })
    return z_ndvi
