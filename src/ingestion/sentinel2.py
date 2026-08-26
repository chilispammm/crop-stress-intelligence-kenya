"""Sentinel-2 Level-2A (L2A) surface reflectance ingestion loader.

Connects to Digital Earth Africa STAC catalog to discover and load Analysis Ready
Data (ARD) multispectral reflectance and Scene Classification Layer (SCL) at native
10-20 m spatial resolution in EPSG:32736.

Memory Integrity & Integer Preservation:
    Native Sentinel-2 L2A surface reflectance is stored as scaled integers (uint16,
    scale factor 0.0001: 10,000 = 1.0 reflectance, 0 = nodata). SCL is stored as uint8.
    This loader PRESERVES native integer representations (uint16 / uint8) and supports
    lazy out-of-core Dask loading to protect memory. Scaling to float is deferred to
    downstream feature extraction.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from src.utils.provenance import attach_provenance, create_provenance_metadata


DEFAULT_DEAFRICA_STAC_URL = "https://explorer.digitalearth.africa/stac"
DEFAULT_S2_COLLECTION = "s2_l2a"
DEFAULT_S2_BANDS = ("B02", "B03", "B04", "B08", "B11", "SCL")


def validate_sentinel2_dataset(ds: xr.Dataset) -> None:
    """Validate Sentinel-2 dataset coordinates, dtypes, and pixel ranges.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset containing Sentinel-2 band variables.

    Raises
    ------
    ValueError
        If coordinates are missing or if pixel values violate physical bounds.
    """
    if "time" not in ds.coords and "time" not in ds.dims:
        raise ValueError("Sentinel-2 dataset is missing 'time' coordinate.")

    for band in ds.data_vars:
        arr = ds[band]
        if band.upper() == "SCL":
            vals = arr.values[np.isfinite(arr.values)]
            if len(vals) > 0:
                min_v, max_v = int(vals.min()), int(vals.max())
                if min_v < 0 or max_v > 11:
                    raise ValueError(f"SCL values must be in [0, 11], found [{min_v}, {max_v}].")
        else:
            vals = arr.values[np.isfinite(arr.values)]
            if len(vals) > 0:
                min_v, max_v = float(vals.min()), float(vals.max())
                if np.issubdtype(arr.dtype, np.integer):
                    if min_v < 0 or max_v > 20000:
                        raise ValueError(
                            f"Scaled integer reflectance for '{band}' out of range [0, 20000]: [{min_v}, {max_v}]."
                        )
                else:
                    # Float scaled [0.0, 1.0] (with small margin for high cloud reflectance up to 1.5)
                    if min_v < -0.05 or max_v > 1.5:
                        raise ValueError(
                            f"Reflectance for band '{band}' out of physical range [0.0, 1.0]: [{min_v}, {max_v}]."
                        )


def load_sentinel2_data(
    bbox: Tuple[float, float, float, float],
    start_date: Union[str, datetime.date, datetime.datetime, pd.Timestamp],
    end_date: Union[str, datetime.date, datetime.datetime, pd.Timestamp],
    bands: Sequence[str] = DEFAULT_S2_BANDS,
    stac_url: str = DEFAULT_DEAFRICA_STAC_URL,
    collection: str = DEFAULT_S2_COLLECTION,
    resolution: int = 10,
    crs: str = "EPSG:32736",
    chunks: Optional[Dict[str, int]] = None,
    source: Optional[Union[str, Path, xr.Dataset]] = None,
) -> xr.Dataset:
    """Load and subset Sentinel-2 L2A optical data at native 10-20 m resolution.

    Maintains memory integrity by preserving native uint16 reflectance and uint8 SCL
    representations. Supports lazy loading via Dask.

    Parameters
    ----------
    bbox : tuple of float
        Spatial bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    start_date : str or datetime
        Start date (inclusive).
    end_date : str or datetime
        End date (inclusive).
    bands : sequence of str, default ('B02', 'B03', 'B04', 'B08', 'B11', 'SCL')
        Band identifiers to load.
    stac_url : str, default 'https://explorer.digitalearth.africa/stac'
        STAC catalog endpoint URL.
    collection : str, default 's2_l2a'
        Sentinel-2 STAC collection ID.
    resolution : int, default 10
        Spatial resolution in meters (e.g. 10 or 20).
    crs : str, default 'EPSG:32736'
        Target optical projection (UTM Zone 36S).
    chunks : dict, optional
        Dask chunk structure (e.g. {'x': 512, 'y': 512}).
    source : str, Path, or xr.Dataset, optional
        Local dataset or preloaded fixture (for testing/offline execution).

    Returns
    -------
    xr.Dataset
        Dataset containing optical bands at native resolution with full provenance metadata.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if start_ts > end_ts:
        raise ValueError(f"start_date ({start_ts}) must not be after end_date ({end_ts}).")

    if isinstance(source, xr.Dataset):
        ds_raw = source
    elif isinstance(source, (str, Path)) and Path(str(source)).exists():
        ds_raw = xr.open_dataset(source)
    else:
        # Live STAC discovery and loading via Digital Earth Africa STAC
        import pystac_client
        import odc.stac

        try:
            client = pystac_client.Client.open(stac_url)
            date_range = f"{start_ts.strftime('%Y-%m-%d')}/{end_ts.strftime('%Y-%m-%d')}"
            search = client.search(
                collections=[collection],
                bbox=[min_lon, min_lat, max_lon, max_lat],
                datetime=date_range,
            )
            items = list(search.items())
            if not items:
                raise ValueError(
                    f"No Sentinel-2 items found in collection '{collection}' for bbox {bbox} "
                    f"and datetime range {date_range} on endpoint '{stac_url}'."
                )

            # Digital Earth Africa COGs reside in AWS region af-south-1 (public unsigned access)
            with rasterio.Env(
                AWS_NO_SIGN_REQUEST="YES",
                AWS_DEFAULT_REGION="af-south-1",
                AWS_REGION="af-south-1",
            ):
                load_kwargs: Dict[str, Any] = {
                    "bands": list(bands),
                    "bbox": [min_lon, min_lat, max_lon, max_lat],
                    "resolution": resolution,
                    "crs": crs,
                }
                if chunks is not None:
                    load_kwargs["chunks"] = chunks

                ds_raw = odc.stac.load(items, **load_kwargs)

        except Exception as exc:
            raise ConnectionError(
                f"Failed to query or load live Sentinel-2 data from STAC endpoint '{stac_url}': {exc}"
            ) from exc

    # Standardize coordinate names
    coord_map = {}
    for c in ds_raw.coords:
        if c.lower() in ("latitude", "lats", "y"):
            coord_map[c] = "y" if "y" in ds_raw.coords else "lat"
        elif c.lower() in ("longitude", "lons", "x"):
            coord_map[c] = "x" if "x" in ds_raw.coords else "lon"
    if coord_map:
        ds_raw = ds_raw.rename(coord_map)

    # Filter bands
    available_bands = [b for b in bands if b in ds_raw.data_vars]
    if not available_bands:
        raise ValueError(f"None of the requested bands {bands} found in dataset: {list(ds_raw.data_vars)}")
    ds_bands = ds_raw[available_bands]

    # Temporal subset if source was pre-loaded
    if "time" in ds_bands.coords:
        ds_subset = ds_bands.sel(time=slice(start_ts, end_ts))
    else:
        ds_subset = ds_bands

    # Attach canonical band metadata (maintaining native integer representations)
    for band_name in ds_subset.data_vars:
        da = ds_subset[band_name]
        if band_name.upper() == "SCL":
            da.attrs.update({
                "standard_name": "scene_classification",
                "long_name": "Scene Classification Layer (SCL)",
                "units": "class_enum",
                "valid_range": (0, 11),
                "dtype_native": str(da.dtype),
            })
        else:
            da.attrs.update({
                "standard_name": f"surface_reflectance_{band_name.lower()}",
                "long_name": f"Sentinel-2 L2A BOA Reflectance {band_name}",
                "scale_factor": 0.0001,
                "units": "scaled_integer_reflectance" if np.issubdtype(da.dtype, np.integer) else "surface_reflectance",
                "dtype_native": str(da.dtype),
            })

    # Attach provenance
    prov = create_provenance_metadata(
        source_name="Digital Earth Africa / Copernicus ESA",
        product_name="Sentinel-2 Level-2A Analysis Ready Data",
        product_version="L2A",
        spatial_resolution="10-20 m native",
        temporal_resolution="5 days (Sentinel-2A + 2B constellation)",
        native_crs=crs,
        known_limitations=[
            "Cloud cover and atmospheric haze during peak Long Rains (April-July) cause optical gaps",
            "Reflectance values are stored as native scaled integers (uint16) to conserve memory",
        ],
        source_url=stac_url,
        collection_id=collection,
        transformations_applied=["spatial_bounding_box_subset", "band_selection"],
    )
    attach_provenance(ds_subset, prov)
    validate_sentinel2_dataset(ds_subset)
    return ds_subset
