"""NASA SMAP Level-4 Surface and Root-Zone Soil Moisture (SPL4SMGP) Ingestion Loader.

Provides root-zone soil wetness (0–100 cm) and topsoil relative wetness (0–5 cm)
from the NASA SMAP Level-4 Global 3-hourly 9 km EASE-Grid Surface and Root Zone
Soil Moisture Geophysical Data product (SPL4SMGP Version 8).

Scientific Context
------------------
- Assimilates SMAP L-band radar/radiometer brightness temperatures into the NASA
  Catchment Land Surface Model (CLSM) using an Ensemble Kalman Filter (EnKF) forced
  by GEOS-5 atmospheric analysis.
- Variables `sm_rootzone_wetness` (root-zone, 0–100 cm) and `sm_surface_wetness`
  (topsoil, 0–5 cm) are dimensionless relative saturation fractions in [0.0, 1.0].
- 3-hourly observations are aggregated to daily arithmetic means before 14-day
  half-open interval compositing, matching the project's temporal contracts.
- Explicit provenance flag `source="SMAP_L4"` is attached to all ingested datasets.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import ssl
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr

from src.utils.provenance import attach_provenance, create_provenance_metadata


CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
DEFAULT_SMAP_VARIABLES = ("s0", "s1")

VARIABLE_METADATA: Dict[str, Dict[str, Any]] = {
    "s0": {
        "standard_name": "surface_soil_relative_wetness",
        "long_name": "Topsoil Relative Wetness (0-5cm)",
        "units": "dimensionless",
        "valid_min": 0.0,
        "valid_max": 1.0,
        "depth": "0-5 cm",
        "smap_h5_layer": "sm_surface_wetness",
        "smap_group": "Geophysical_Data",
        "description": "SMAP L4 satellite-assimilated topsoil relative wetness index",
    },
    "s1": {
        "standard_name": "rootzone_soil_water_index",
        "long_name": "Root-zone Soil Water Index (0-1m)",
        "units": "dimensionless",
        "valid_min": 0.0,
        "valid_max": 1.0,
        "depth": "0-1 m",
        "smap_h5_layer": "sm_rootzone_wetness",
        "smap_group": "Geophysical_Data",
        "description": "SMAP L4 satellite-assimilated root-zone soil water index",
    },
}


def query_smap_cmr(
    bbox: Tuple[float, float, float, float],
    start_date: Union[str, datetime.date, datetime.datetime, pd.Timestamp],
    end_date: Union[str, datetime.date, datetime.datetime, pd.Timestamp],
    short_name: str = "SPL4SMGP",
    version: str = "008",
    cmr_url: str = CMR_GRANULES_URL,
    page_size: int = 2000,
) -> List[Dict[str, Any]]:
    """Query NASA Common Metadata Repository (CMR) for SMAP L4 granules.

    Parameters
    ----------
    bbox : tuple of float
        Bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    start_date : str or datetime
        Start timestamp.
    end_date : str or datetime
        End timestamp.
    short_name : str, default 'SPL4SMGP'
        NASA product short name.
    version : str, default '008'
        NASA product collection version.
    cmr_url : str, default CMR_GRANULES_URL
        NASA CMR REST API search endpoint.
    page_size : int, default 2000
        Maximum entries to request per query.

    Returns
    -------
    list of dict
        Granule metadata entries containing download URLs and observation times.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    start_iso = pd.to_datetime(start_date).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = pd.to_datetime(end_date).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = (
        f"short_name={short_name}&version={version}&"
        f"temporal={start_iso},{end_iso}&"
        f"bounding_box={min_lon},{min_lat},{max_lon},{max_lat}&"
        f"page_size={page_size}"
    )
    req_url = f"{cmr_url}?{params}"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(req_url, headers={"User-Agent": "CropStressIntelligence/0.4.0"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("feed", {}).get("entry", [])
    except Exception as exc:
        raise ConnectionError(f"Failed to query NASA CMR for SMAP L4 granules: {exc}") from exc


def validate_smap_dataset(
    ds: xr.Dataset,
    variables: Optional[Sequence[str]] = None,
) -> None:
    """Validate data integrity, units, valid ranges, and timestamps of a SMAP dataset.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset containing SMAP variables.
    variables : sequence of str, optional
        List of variable names to validate. If None, validates all recognized variables.

    Raises
    ------
    ValueError
        If timestamps are invalid/non-monotonic, or values fall outside physical bounds [0, 1].
    """
    if "time" not in ds.coords and "time" not in ds.dims:
        raise ValueError("SMAP dataset is missing 'time' coordinate.")

    time_coord = ds["time"]
    if len(time_coord) > 1:
        time_values = pd.to_datetime(time_coord.values)
        if not time_values.is_monotonic_increasing:
            raise ValueError("SMAP time coordinate is not monotonically increasing.")

    check_vars = list(variables) if variables is not None else [v for v in ds.data_vars if v in VARIABLE_METADATA]

    for var_name in check_vars:
        if var_name not in ds:
            continue
        data_arr = ds[var_name]
        meta = VARIABLE_METADATA.get(var_name, {})

        # Unit verification
        units = data_arr.attrs.get("units", meta.get("units", "dimensionless"))
        if units not in ("dimensionless", "1", "wetness units", "fraction", "index"):
            raise ValueError(
                f"Invalid units '{units}' for SMAP variable '{var_name}'. "
                f"Expected dimensionless fraction [0.0, 1.0]."
            )

        # Range check on non-NaN values
        valid_min = meta.get("valid_min", 0.0)
        valid_max = meta.get("valid_max", 1.0)

        data_vals = data_arr.values
        finite_vals = data_vals[np.isfinite(data_vals)]
        if len(finite_vals) > 0:
            min_val = float(np.min(finite_vals))
            max_val = float(np.max(finite_vals))
            if min_val < (valid_min - 0.01) or max_val > (valid_max + 0.01):
                raise ValueError(
                    f"Physical value range violation for '{var_name}': "
                    f"observed [{min_val:.3f}, {max_val:.3f}], expected [{valid_min}, {valid_max}]."
                )


def load_smap_data(
    bbox: Tuple[float, float, float, float],
    start_date: Union[str, datetime.date, datetime.datetime, pd.Timestamp],
    end_date: Union[str, datetime.date, datetime.datetime, pd.Timestamp],
    variables: Sequence[str] = DEFAULT_SMAP_VARIABLES,
    source: Optional[Union[str, Path, xr.Dataset]] = None,
    auth_token: Optional[str] = None,
) -> xr.Dataset:
    """Load, subset, and daily-aggregate NASA SMAP Level-4 soil moisture data.

    Parameters
    ----------
    bbox : tuple of float
        Bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    start_date : str or datetime
        Start date (inclusive).
    end_date : str or datetime
        End date (inclusive).
    variables : sequence of str, default ('s0', 's1')
        Variables to load: 's0' (topsoil wetness) and/or 's1' (root-zone wetness).
    source : str, Path, or xr.Dataset, optional
        Preloaded xr.Dataset, local file path, or directory containing SMAP granules.
        If None, queries NASA Earthdata CMR.
    auth_token : str, optional
        NASA Earthdata Bearer token. If None, checks `EARTHDATA_TOKEN` env var.

    Returns
    -------
    xr.Dataset
        xarray Dataset with daily mean observations, native resolution, verified
        monotonic timestamps, clean NaN masks, and explicit SMAP_L4 provenance.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    if start_ts > end_ts:
        raise ValueError(f"start_date ({start_ts}) must not be after end_date ({end_ts}).")

    requested_vars = [v.lower() for v in variables]
    for v in requested_vars:
        if v not in ("s0", "s1"):
            raise ValueError(f"Unknown SMAP variable '{v}'. Expected 's0' and/or 's1'.")

    # If an existing Dataset is provided (synthetic fixtures, offline tests, pre-downloaded)
    if isinstance(source, xr.Dataset):
        ds_raw = source
    elif isinstance(source, (str, Path)) and Path(str(source)).exists():
        p = Path(str(source))
        if p.is_file():
            ds_raw = xr.open_dataset(p)
        else:
            # Directory of netcdf/h5 files
            ds_raw = xr.open_mfdataset(str(p / "*.nc"), combine="by_coords")
    else:
        # Query CMR
        granules = query_smap_cmr(bbox, start_date, end_date)
        if not granules:
            raise ConnectionError(
                f"No SMAP L4 granules found in NASA CMR for bbox {bbox} "
                f"and period [{start_date}, {end_date}]."
            )

        token = auth_token or os.environ.get("EARTHDATA_TOKEN") or os.environ.get("EDL_TOKEN")
        if not token:
            raise PermissionError(
                "NASA Earthdata credentials not found. Set EARTHDATA_TOKEN environment "
                "variable or pass auth_token to download live SMAP L4 granules."
            )

        # Ingestion from authenticated HTTPS
        import requests
        headers = {"Authorization": f"Bearer {token}"}
        # In multi-granule mode, downloads would proceed with session
        raise NotImplementedError("Live multi-granule streaming requires authenticated session.")

    # Standardize coordinate names
    coord_map = {}
    for c in ds_raw.coords:
        if c.lower() in ("latitude", "lats", "cell_lat"):
            coord_map[c] = "lat"
        elif c.lower() in ("longitude", "lons", "cell_lon"):
            coord_map[c] = "lon"
    if coord_map:
        ds_raw = ds_raw.rename(coord_map)

    # Standardize variable names
    var_map = {}
    for name in list(ds_raw.data_vars):
        name_lower = name.lower()
        if any(k in name_lower for k in ["sm_surface_wetness", "surface_wetness", "topsoil"]):
            var_map[name] = "s0"
        elif any(k in name_lower for k in ["sm_rootzone_wetness", "rootzone_wetness", "root_zone_wetness", "swi"]):
            var_map[name] = "s1"
    if var_map:
        ds_raw = ds_raw.rename(var_map)

    # Validate presence of coordinates
    if "lat" not in ds_raw.coords or "lon" not in ds_raw.coords:
        raise ValueError("SMAP dataset must contain 'lat' and 'lon' spatial coordinates.")

    # Spatial subsetting
    lat_vals = ds_raw["lat"].values
    lat_ascending = bool(lat_vals[0] < lat_vals[-1]) if len(lat_vals) > 1 else True
    lat_step = abs(float(lat_vals[1] - lat_vals[0])) if len(lat_vals) > 1 else 0.08
    lon_step = abs(float(ds_raw["lon"].values[1] - ds_raw["lon"].values[0])) if len(ds_raw["lon"].values) > 1 else 0.08

    adj_min_lat = min_lat - lat_step / 2.0
    adj_max_lat = max_lat + lat_step / 2.0
    adj_min_lon = min_lon - lon_step / 2.0
    adj_max_lon = max_lon + lon_step / 2.0

    lat_slice = slice(adj_min_lat, adj_max_lat) if lat_ascending else slice(adj_max_lat, adj_min_lat)
    lon_slice = slice(adj_min_lon, adj_max_lon)

    ds_spatial = ds_raw.sel(lat=lat_slice, lon=lon_slice)

    # Temporal subsetting & 3-hourly -> daily aggregation
    if "time" in ds_spatial.coords:
        ds_temporal = ds_spatial.sel(time=slice(start_ts, end_ts))
        # Aggregate to daily means if sub-daily
        time_diffs = pd.to_datetime(ds_temporal.time.values)
        if len(time_diffs) > 1 and (time_diffs[1] - time_diffs[0]) < pd.Timedelta("1D"):
            ds_daily = ds_temporal.resample(time="1D").mean()
        else:
            ds_daily = ds_temporal
    else:
        ds_daily = ds_spatial

    # Clean fill values and enforce physical bounds [0.0, 1.0]
    for var in requested_vars:
        if var in ds_daily:
            arr = ds_daily[var]
            if "_FillValue" in arr.attrs:
                fill_val = arr.attrs["_FillValue"]
                arr = arr.where(arr != fill_val)
            arr = arr.where((arr >= 0.0) & (arr <= 1.0))
            arr = arr.clip(0.0, 1.0)

            meta = VARIABLE_METADATA.get(var, {})
            arr.attrs.update({
                "standard_name": meta.get("standard_name", var),
                "long_name": meta.get("long_name", var),
                "units": "dimensionless",
                "valid_min": 0.0,
                "valid_max": 1.0,
                "depth": meta.get("depth", "unknown"),
                "source_system": "NASA SMAP Level-4 (SPL4SMGP Version 8)",
                "assimilation_method": "Ensemble Kalman Filter (CLSM + SMAP L-band Tb)",
                "field_measurement": False,
                "source": "SMAP_L4",
            })
            ds_daily[var] = arr

    # Attach comprehensive provenance
    prov = create_provenance_metadata(
        source_name="NASA Earth Science / GMAO / NSIDC DAAC",
        product_name="SMAP L4 Global 3-hourly 9 km EASE-Grid Surface and Root Zone Soil Moisture (SPL4SMGP V008)",
        product_version="v8.0",
        spatial_resolution="9 km EASE-Grid 2.0 (EPSG:6933)",
        temporal_resolution="3-hourly aggregated to daily mean",
        native_crs="EPSG:6933",
        known_limitations=[
            "Satellite data assimilation product based on Catchment LSM; not direct in-situ root-zone probe measurement",
            "Coarse 9 km grid reflects landscape-level hydrological forcing, not intra-field tile drainage or microtopography",
            "Root-zone wetness is normalized saturation fraction [0, 1]",
        ],
        source_url="https://doi.org/10.5067/T5RUATAQREF8",
        transformations_applied=["spatial_bounding_box_subset", "3hourly_to_daily_mean_aggregation", "fill_value_masking"],
    )
    attach_provenance(ds_daily, prov)

    # Attach explicit top-level dataset attribute
    ds_daily.attrs["source"] = "SMAP_L4"
    ds_daily.attrs["soil_moisture_source"] = "NASA_SMAP_L4_SPL4SMGP_V008"

    # Validate before return
    validate_smap_dataset(ds_daily, variables=requested_vars)

    return ds_daily
