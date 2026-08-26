"""Cropland mask application and agricultural filtering.

Enforces spatial boundaries for agricultural analysis by masking out non-cropland pixels.

Scientific & Resolution Note:
    The canonical DE Africa Cropland Extent layer (`crop_mask_eastern`) is produced at native 20 m
    resolution in EPSG:32736. Target-grid alignment to 10 m optical features (or 20 m SWIR features)
    does NOT alter the underlying physical 20 m native resolution of this mask.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import xarray as xr


def apply_cropland_mask(
    feature_da: xr.DataArray,
    crop_mask: xr.DataArray,
    cultivated_value: int = 1,
) -> xr.DataArray:
    """Apply agricultural / cropland mask to a biophysical feature array.

    Aligns the cropland mask (native 20 m in EPSG:32736) to the feature array's target
    spatial grid (e.g. 10 m NDVI/EVI or 20 m NDMI) and sets all non-cultivated pixels
    (crop_mask != cultivated_value) to NaN.

    Parameters
    ----------
    feature_da : xr.DataArray
        Input biophysical feature array (e.g. NDVI, EVI, NDMI, or band reflectance).
    crop_mask : xr.DataArray
        Binary or categorical cropland mask (e.g. DE Africa Cropland Extent 20 m).
    cultivated_value : int, default 1
        Value within `crop_mask` representing active/cultivated cropland.

    Returns
    -------
    xr.DataArray
        A copy of `feature_da` with non-cropland pixels set to NaN.
    """
    # Identify spatial dimension names
    y_name = "y" if "y" in feature_da.coords else ("lat" if "lat" in feature_da.coords else "latitude")
    x_name = "x" if "x" in feature_da.coords else ("lon" if "lon" in feature_da.coords else "longitude")

    mask_y = "y" if "y" in crop_mask.coords else ("lat" if "lat" in crop_mask.coords else "latitude")
    mask_x = "x" if "x" in crop_mask.coords else ("lon" if "lon" in crop_mask.coords else "longitude")

    # Rename mask coordinates if names differ
    mask_aligned = crop_mask
    rename_dict = {}
    if mask_y != y_name and mask_y in mask_aligned.coords:
        rename_dict[mask_y] = y_name
    if mask_x != x_name and mask_x in mask_aligned.coords:
        rename_dict[mask_x] = x_name
    if rename_dict:
        mask_aligned = mask_aligned.rename(rename_dict)

    # Reindex / align mask to feature_da grid using nearest-neighbor sampling
    if mask_aligned.sizes.get(y_name) != feature_da.sizes.get(y_name) or mask_aligned.sizes.get(x_name) != feature_da.sizes.get(x_name):
        try:
            mask_aligned = mask_aligned.reindex_like(feature_da, method="nearest")
        except Exception:
            mask_aligned = mask_aligned.interp(
                {y_name: feature_da[y_name], x_name: feature_da[x_name]},
                method="nearest",
            )

    # Boolean condition: pixel is cultivated cropland
    is_cropland = mask_aligned == cultivated_value

    # Apply mask
    masked_da = feature_da.where(is_cropland, np.nan)
    masked_da.attrs.update(feature_da.attrs)
    masked_da.attrs["cropland_masked"] = True
    masked_da.attrs["cultivated_value"] = cultivated_value

    return masked_da
