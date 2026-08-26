"""Agricultural / Cropland mask ingestion loader.

Canonical Product:
    Digital Earth Africa Cropland Extent Map (20 m)
    - Collection: `crop_mask_eastern`
    - Band: `mask` (1 = crop, 0 = non-crop)
    - Native Resolution: 20 m
    - Native CRS: `EPSG:32736`
    - Provider: Digital Earth Africa

Fallback Product (Defined separately):
    ESA WorldCover 10 m
    - Collection: `esa_worldcover_2021` / `esa_worldcover`
    - Band: `map` (Class 40 = Cropland / Herbaceous wetland / Cultivated vegetation)

Critical Scientific Limitations & Qualifications:
    1. GENERAL CROPLAND VS MAIZE: This mask identifies general cultivated/arable land. It does NOT
       provide a direct mono-crop maize classification. Maize is the dominant staple in Uasin Gishu
       during the Long Rains season (>75% of cultivated area), but parcels may also contain wheat,
       barley, beans, horticulture, or seasonal fallow.
    2. STATIC BASELINE: A historical or static baseline mask does NOT represent current-year ground truth.
       It must be qualified as an agricultural spatial filter and combined with phenological time-series
       screening to detect active crop growth.
    3. NATIVE RESOLUTION INTEGRITY: The canonical DE Africa Cropland Extent layer is produced at native 20 m
       resolution in EPSG:32736. Target-grid alignment to 10 m optical features in downstream masking does
       NOT change its native 20 m resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import rasterio
import xarray as xr

from src.utils.provenance import attach_provenance, create_provenance_metadata


DEFAULT_DEAFRICA_STAC_URL = "https://explorer.digitalearth.africa/stac"
CANONICAL_MASK_COLLECTION = "crop_mask_eastern"
FALLBACK_MASK_COLLECTION = "esa_worldcover_2021"


def validate_crop_mask(da: xr.DataArray) -> None:
    """Validate binary values of the crop mask.

    Parameters
    ----------
    da : xr.DataArray
        Binary mask DataArray.

    Raises
    ------
    ValueError
        If values contain integers or floats outside {0, 1, NaN}.
    """
    vals = da.values[np.isfinite(da.values)]
    if len(vals) > 0:
        unique_vals = set(np.unique(vals))
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"Crop mask must be binary with values in {{0, 1}}, found values: {unique_vals}"
            )


def load_crop_mask(
    bbox: Tuple[float, float, float, float],
    stac_url: str = DEFAULT_DEAFRICA_STAC_URL,
    collection: str = CANONICAL_MASK_COLLECTION,
    resolution: int = 20,
    crs: str = "EPSG:32736",
    source: Optional[Union[str, Path, xr.DataArray, xr.Dataset]] = None,
) -> xr.DataArray:
    """Load and subset the agricultural / cropland mask for the target AOI at native 20 m resolution.

    Target-grid alignment to 10 m features does not alter the native 20 m resolution of this product.

    Parameters
    ----------
    bbox : tuple of float
        Spatial bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    stac_url : str, default 'https://explorer.digitalearth.africa/stac'
        STAC catalog endpoint URL.
    collection : str, default 'crop_mask_eastern'
        Canonical STAC collection identifier.
    resolution : int, default 20
        Spatial resolution in meters (native: 20 m).
    crs : str, default 'EPSG:32736'
        Spatial coordinate reference system.
    source : str, Path, xr.DataArray, or xr.Dataset, optional
        Source mask data file or preloaded xarray object (for offline testing).

    Returns
    -------
    xr.DataArray
        Binary 20 m agricultural mask (1 = cultivated land, 0 = non-agricultural, NaN = nodata)
        with explicit qualifications and provenance metadata.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    if isinstance(source, xr.DataArray):
        da_raw = source
    elif isinstance(source, xr.Dataset):
        var_name = "mask" if "mask" in source.data_vars else ("crop_mask" if "crop_mask" in source.data_vars else list(source.data_vars)[0])
        da_raw = source[var_name]
    elif isinstance(source, (str, Path)) and Path(str(source)).exists():
        ds = xr.open_dataset(source)
        var_name = "mask" if "mask" in ds.data_vars else ("crop_mask" if "crop_mask" in ds.data_vars else list(ds.data_vars)[0])
        da_raw = ds[var_name]
    else:
        # Live STAC discovery and loading via Digital Earth Africa
        import pystac_client
        import odc.stac

        try:
            client = pystac_client.Client.open(stac_url)
            search = client.search(
                collections=[collection],
                bbox=[min_lon, min_lat, max_lon, max_lat],
                limit=1,
            )
            items = list(search.items())
            if not items:
                raise ValueError(
                    f"No crop mask items found in collection '{collection}' for bbox {bbox} on endpoint '{stac_url}'."
                )

            with rasterio.Env(
                AWS_NO_SIGN_REQUEST="YES",
                AWS_DEFAULT_REGION="af-south-1",
                AWS_REGION="af-south-1",
            ):
                band_to_load = ["mask"] if "mask" in items[0].assets else (["map"] if "map" in items[0].assets else None)
                ds_loaded = odc.stac.load(
                    items,
                    bands=band_to_load,
                    bbox=[min_lon, min_lat, max_lon, max_lat],
                    resolution=resolution,
                    crs=crs,
                )
                var_name = list(ds_loaded.data_vars)[0]
                da_raw = ds_loaded[var_name]

                # If loaded with a time dimension of 1, drop or squeeze time
                if "time" in da_raw.dims and da_raw.sizes["time"] == 1:
                    da_raw = da_raw.squeeze("time", drop=True)

        except Exception as exc:
            raise ConnectionError(
                f"Failed to query or load live crop mask from STAC endpoint '{stac_url}': {exc}"
            ) from exc

    # Standardize coordinates
    coord_map = {}
    for c in da_raw.coords:
        if c.lower() in ("latitude", "lats", "y"):
            coord_map[c] = "y" if "y" in da_raw.coords else "lat"
        elif c.lower() in ("longitude", "lons", "x"):
            coord_map[c] = "x" if "x" in da_raw.coords else "lon"
    if coord_map:
        da_raw = da_raw.rename(coord_map)

    # Spatial slicing if source was pre-loaded
    if "lat" in da_raw.coords and "lon" in da_raw.coords:
        lat_vals = da_raw["lat"].values
        lat_ascending = bool(lat_vals[0] < lat_vals[-1]) if len(lat_vals) > 1 else True
        lat_slice = slice(min_lat, max_lat) if lat_ascending else slice(max_lat, min_lat)
        lon_slice = slice(min_lon, max_lon)
        da_subset = da_raw.sel(lat=lat_slice, lon=lon_slice)
    else:
        da_subset = da_raw

    # Convert to standard binary [0, 1]
    da_subset = da_subset.where(np.isfinite(da_subset), np.nan)
    # If ESA WorldCover class 40 is used
    if da_subset.max() > 1.5:
        da_subset = (da_subset == 40).astype(float).where(np.isfinite(da_subset))
    else:
        da_subset = (da_subset > 0).astype(float).where(np.isfinite(da_subset))

    da_subset.name = "crop_mask"
    da_subset.attrs.update({
        "standard_name": "crop_mask",
        "long_name": "Digital Earth Africa Cropland Extent Mask",
        "units": "binary_flag",
        "valid_values": "0: non-crop, 1: agricultural cropland",
        "spatial_resolution": "20 m native",
        "native_crs": crs,
        "maize_qualification": "General cropland filter; not a verified mono-crop maize classification.",
    })

    # Attach provenance
    prov = create_provenance_metadata(
        source_name="Digital Earth Africa",
        product_name="Cropland Extent Map (20 m)",
        product_version="v1.0",
        spatial_resolution="20 m native",
        temporal_resolution="Annual / Static Baseline",
        native_crs=crs,
        known_limitations=[
            "Identifies general arable/cultivated fields, NOT mono-crop maize specifically",
            "Static historical baseline does not reflect real-time seasonal crop rotations",
            "Must be combined with seasonal phenology time-series to verify active maize cultivation",
            "Native resolution is 20 m in EPSG:32736; target-grid alignment to 10 m does not alter native scale",
        ],
        source_url=stac_url,
        collection_id=collection,
        transformations_applied=["spatial_bounding_box_subset", "binary_thresholding"],
    )
    attach_provenance(da_subset, prov)
    validate_crop_mask(da_subset)
    return da_subset
