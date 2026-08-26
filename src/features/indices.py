"""Resolution-aware biophysical feature engineering and spectral index extraction.

Supported Indices:
    - NDVI: Normalized Difference Vegetation Index (10 m native using B04 & B08)
    - EVI: Enhanced Vegetation Index (10 m native using B02, B04 & B08)
    - NDMI: Normalized Difference Moisture Index (20 m derived using 10 m NIR downsampled to 20 m B11)

Mathematical Formulations & Lazy Scaling:
    Reflectance inputs are stored natively as scaled integers (uint16, scale factor 10^-4).
    All calculation functions perform lazy scaling during computation:
        Reflectance = raw_band * 0.0001 (if integer scaled)
    Raw floating-point index values are preserved without blind clamping to [-1.0, 1.0].
    Distribution audits are performed via `validate_index_distribution()`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import xarray as xr


class GridAlignmentError(ValueError):
    """Raised when spatial grids, extents, CRS, or resolutions cannot be safely aligned."""
    pass


def _to_float_reflectance(da: xr.DataArray) -> xr.DataArray:
    """Convert native scaled integer reflectance (uint16) to float [0.0, 1.0] lazily."""
    if np.issubdtype(da.dtype, np.integer):
        return da.astype(np.float32) * 0.0001
    elif da.attrs.get("scale_factor") == 0.0001 and float(da.max()) > 1.5:
        return da * 0.0001
    return da


def calculate_ndvi(
    red: xr.DataArray,
    nir: xr.DataArray,
    epsilon: float = 1e-6,
) -> xr.DataArray:
    """Calculate Normalized Difference Vegetation Index (NDVI) at native 10 m resolution.

    Formula:
        NDVI = (NIR - Red) / (NIR + Red + 1e-6)

    Parameters
    ----------
    red : xr.DataArray
        Sentinel-2 B04 (Red) band array at 10 m resolution.
    nir : xr.DataArray
        Sentinel-2 B08 (NIR Broad) band array at 10 m resolution.
    epsilon : float, default 1e-6
        Small positive constant to prevent division by zero in dark / masked pixels.

    Returns
    -------
    xr.DataArray
        NDVI array at 10 m resolution with metadata attached.
    """
    red_f = _to_float_reflectance(red)
    nir_f = _to_float_reflectance(nir)

    ndvi = (nir_f - red_f) / (nir_f + red_f + epsilon)
    ndvi.name = "ndvi"
    ndvi.attrs.update({
        "standard_name": "normalized_difference_vegetation_index",
        "long_name": "Normalized Difference Vegetation Index (NDVI)",
        "units": "dimensionless",
        "spatial_resolution": "10 m native",
        "formula": "(NIR - Red) / (NIR + Red + 1e-6)",
        "bands_used": "B08 (NIR), B04 (Red)",
    })
    return ndvi


def calculate_evi(
    blue: xr.DataArray,
    red: xr.DataArray,
    nir: xr.DataArray,
    gain: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    canopy_l: float = 1.0,
    epsilon: float = 1e-6,
) -> xr.DataArray:
    """Calculate standard 3-band Enhanced Vegetation Index (EVI) at native 10 m resolution.

    Formula:
        EVI = 2.5 * (NIR - Red) / (NIR + 6.0 * Red - 7.5 * Blue + 1.0)

    Parameters
    ----------
    blue : xr.DataArray
        Sentinel-2 B02 (Blue) band array at 10 m resolution.
    red : xr.DataArray
        Sentinel-2 B04 (Red) band array at 10 m resolution.
    nir : xr.DataArray
        Sentinel-2 B08 (NIR Broad) band array at 10 m resolution.
    gain : float, default 2.5
        EVI overall gain factor.
    c1 : float, default 6.0
        Atmospheric aerosol resistance coefficient for Red.
    c2 : float, default 7.5
        Atmospheric aerosol resistance coefficient for Blue.
    canopy_l : float, default 1.0
        Canopy background adjustment factor.
    epsilon : float, default 1e-6
        Small positive constant to prevent denominator zero-crossing.

    Returns
    -------
    xr.DataArray
        EVI array at 10 m resolution with metadata attached.
    """
    blue_f = _to_float_reflectance(blue)
    red_f = _to_float_reflectance(red)
    nir_f = _to_float_reflectance(nir)

    denominator = nir_f + (c1 * red_f) - (c2 * blue_f) + canopy_l
    denominator = denominator.where(denominator != 0.0, epsilon)

    evi = gain * (nir_f - red_f) / denominator
    evi.name = "evi"
    evi.attrs.update({
        "standard_name": "enhanced_vegetation_index",
        "long_name": "Enhanced Vegetation Index (EVI)",
        "units": "dimensionless",
        "spatial_resolution": "10 m native",
        "formula": "2.5 * (NIR - Red) / (NIR + 6.0 * Red - 7.5 * Blue + 1.0)",
        "bands_used": "B08 (NIR), B04 (Red), B02 (Blue)",
    })
    return evi


def _get_spatial_coord_names(da: xr.DataArray) -> Tuple[str, str]:
    """Identify horizontal and vertical spatial coordinate names."""
    y_name = "y" if "y" in da.coords else ("lat" if "lat" in da.coords else "latitude")
    x_name = "x" if "x" in da.coords else ("lon" if "lon" in da.coords else "longitude")
    if y_name not in da.coords or x_name not in da.coords:
        raise GridAlignmentError(
            f"Array is missing spatial coordinates (found coords: {list(da.coords)})"
        )
    return y_name, x_name


def _extract_crs(da: xr.DataArray) -> Optional[Any]:
    """Safely extract CRS from DataArray attributes or rioxarray accessor."""
    if "crs" in da.attrs:
        return da.attrs["crs"]
    try:
        if hasattr(da, "rio") and da.rio.crs is not None:
            return str(da.rio.crs)
    except Exception:
        pass
    return None


def _verify_grid_alignment(nir: xr.DataArray, swir: xr.DataArray) -> None:
    """Verify spatial CRS match, extent overlap, orientation, and resolution ratio."""
    # 1. Coordinate presence
    nir_y, nir_x = _get_spatial_coord_names(nir)
    swir_y, swir_x = _get_spatial_coord_names(swir)

    # 2. CRS match (if CRS metadata present on both)
    nir_crs = _extract_crs(nir)
    swir_crs = _extract_crs(swir)
    if nir_crs is not None and swir_crs is not None and str(nir_crs) != str(swir_crs):
        raise GridAlignmentError(
            f"CRS mismatch: NIR grid has CRS '{nir_crs}', SWIR grid has CRS '{swir_crs}'."
        )

    # 3. Spatial extent overlap check
    nir_min_y, nir_max_y = float(nir[nir_y].min()), float(nir[nir_y].max())
    nir_min_x, nir_max_x = float(nir[nir_x].min()), float(nir[nir_x].max())
    swir_min_y, swir_max_y = float(swir[swir_y].min()), float(swir[swir_y].max())
    swir_min_x, swir_max_x = float(swir[swir_x].min()), float(swir[swir_x].max())

    overlap_y = min(nir_max_y, swir_max_y) - max(nir_min_y, swir_min_y)
    overlap_x = min(nir_max_x, swir_max_x) - max(nir_min_x, swir_min_x)

    if overlap_y <= 0 or overlap_x <= 0:
        raise GridAlignmentError(
            f"Spatial extents do not overlap: NIR [{nir_min_x:.4f}, {nir_min_y:.4f}, {nir_max_x:.4f}, {nir_max_y:.4f}] "
            f"vs SWIR [{swir_min_x:.4f}, {swir_min_y:.4f}, {swir_max_x:.4f}, {swir_max_y:.4f}]."
        )

    # 4. Coordinate orientation match
    nir_y_asc = bool(nir[nir_y].values[0] < nir[nir_y].values[-1]) if len(nir[nir_y]) > 1 else True
    swir_y_asc = bool(swir[swir_y].values[0] < swir[swir_y].values[-1]) if len(swir[swir_y]) > 1 else True
    if nir_y_asc != swir_y_asc:
        raise GridAlignmentError(
            f"Coordinate orientation mismatch: NIR Y ascending={nir_y_asc}, SWIR Y ascending={swir_y_asc}."
        )

    # 5. Approximate 2:1 resolution relationship
    nir_ny, nir_nx = nir.sizes[nir_y], nir.sizes[nir_x]
    swir_ny, swir_nx = swir.sizes[swir_y], swir.sizes[swir_x]

    if swir_ny > 0 and swir_nx > 0:
        ratio_y = nir_ny / swir_ny
        ratio_x = nir_nx / swir_nx
        if not (1.4 <= ratio_y <= 2.6 and 1.4 <= ratio_x <= 2.6):
            raise GridAlignmentError(
                f"Resolution ratio violation: NIR ({nir_ny}x{nir_nx}) to SWIR ({swir_ny}x{swir_nx}) "
                f"ratio is ({ratio_y:.2f}, {ratio_x:.2f}). Expected ~2:1 relationship."
            )


def calculate_ndmi(
    nir: xr.DataArray,
    swir: xr.DataArray,
    resampling_method: str = "average",
    epsilon: float = 1e-6,
) -> xr.DataArray:
    """Calculate Normalized Difference Moisture Index (NDMI) with explicit 20 m downsampling.

    Performs resolution-aware downsampling of 10 m NIR (B08) to match 20 m SWIR (B11)
    using area-weighted averaging.

    Formula:
        NDMI = (NIR_20m - SWIR_20m) / (NIR_20m + SWIR_20m + 1e-6)

    Parameters
    ----------
    nir : xr.DataArray
        Sentinel-2 B08 (NIR Broad) band array at 10 m native resolution.
    swir : xr.DataArray
        Sentinel-2 B11 (SWIR 1) band array at 20 m native resolution.
    resampling_method : str, default 'average'
        Method used to downsample 10 m NIR to 20 m grid ('average', 'bilinear', 'nearest').
    epsilon : float, default 1e-6
        Small positive constant to prevent division by zero.

    Returns
    -------
    xr.DataArray
        NDMI array at 20 m resolution with metadata attached.

    Raises
    ------
    GridAlignmentError
        If NIR and SWIR grids fail CRS, extent overlap, orientation, or resolution ratio safeguards.
    """
    _verify_grid_alignment(nir, swir)

    nir_f = _to_float_reflectance(nir)
    swir_f = _to_float_reflectance(swir)

    nir_y, nir_x = _get_spatial_coord_names(nir_f)
    swir_y, swir_x = _get_spatial_coord_names(swir_f)

    # Downsample 10 m NIR to match 20 m SWIR grid
    try:
        if nir_f.sizes[nir_y] == 2 * swir_f.sizes[swir_y] and nir_f.sizes[nir_x] == 2 * swir_f.sizes[swir_x]:
            nir_20m = nir_f.coarsen({nir_y: 2, nir_x: 2}, boundary="trim").mean()
            nir_20m = nir_20m.assign_coords({
                nir_y: swir_f[swir_y].values,
                nir_x: swir_f[swir_x].values,
            })
        else:
            nir_20m = nir_f.interp(
                {nir_y: swir_f[swir_y], nir_x: swir_f[swir_x]},
                method="linear" if resampling_method == "average" else resampling_method,
            )
    except Exception:
        nir_20m = nir_f.interp(
            {nir_y: swir_f[swir_y], nir_x: swir_f[swir_x]},
            method="nearest",
        )

    ndmi = (nir_20m - swir_f) / (nir_20m + swir_f + epsilon)
    ndmi.name = "ndmi"
    ndmi.attrs.update({
        "standard_name": "normalized_difference_moisture_index",
        "long_name": "Normalized Difference Moisture Index (NDMI)",
        "units": "dimensionless",
        "spatial_resolution": "20 m derived (10 m NIR downsampled to 20 m SWIR)",
        "formula": "(NIR_20m - SWIR_20m) / (NIR_20m + SWIR_20m + 1e-6)",
        "bands_used": "B08 (NIR 10m -> 20m), B11 (SWIR 20m)",
        "resampling_method": resampling_method,
    })
    return ndmi


def validate_index_distribution(
    da: xr.DataArray,
    index_name: str,
    expected_bounds: Tuple[float, float] = (-1.0, 1.0),
) -> Dict[str, Any]:
    """Audit the statistical distribution of a spectral index without mutating raw values.

    Parameters
    ----------
    da : xr.DataArray
        Index DataArray to audit.
    index_name : str
        Name of the index (e.g. 'NDVI', 'EVI', 'NDMI').
    expected_bounds : tuple of float, default (-1.0, 1.0)
        Theoretical or nominal physical bounds for reporting out-of-bounds occurrences.

    Returns
    -------
    dict
        Audit report with pixel counts, NaN percentage, finite statistics, and out-of-bound statistics.
    """
    total_pixels = int(da.size)
    values = da.values.flatten()

    nan_mask = np.isnan(values)
    nan_count = int(np.sum(nan_mask))
    valid_count = total_pixels - nan_count
    nan_pct = float((nan_count / total_pixels) * 100.0) if total_pixels > 0 else 0.0

    finite_vals = values[~nan_mask]

    min_b, max_b = expected_bounds

    if len(finite_vals) > 0:
        min_v = float(np.min(finite_vals))
        max_v = float(np.max(finite_vals))
        mean_v = float(np.mean(finite_vals))
        std_v = float(np.std(finite_vals))

        oob_mask = (finite_vals < min_b) | (finite_vals > max_b)
        oob_count = int(np.sum(oob_mask))
        oob_pct = float((oob_count / valid_count) * 100.0)
    else:
        min_v = max_v = mean_v = std_v = None
        oob_count = 0
        oob_pct = 0.0

    return {
        "index_name": index_name,
        "total_pixels": total_pixels,
        "valid_pixels": valid_count,
        "nan_pixels": nan_count,
        "nan_pct": round(nan_pct, 2),
        "min": round(min_v, 4) if min_v is not None else None,
        "max": round(max_v, 4) if max_v is not None else None,
        "mean": round(mean_v, 4) if mean_v is not None else None,
        "std": round(std_v, 4) if std_v is not None else None,
        "expected_bounds": expected_bounds,
        "out_of_bounds_count": oob_count,
        "out_of_bounds_pct": round(oob_pct, 2),
    }
