"""Global Root-zone moisture Analysis & Forecasting System (GRAFS) ingestion loader.

GRAFS is a near-real-time global soil moisture data assimilation system produced
by the Australian National University (ANU) Centre for Water and Landscape Dynamics
and distributed via the National Computational Infrastructure (NCI) THREDDS OPeNDAP server.

Scientific Note:
    GRAFS is a satellite-guided hydrological model assimilation product—combining
    Soil Moisture Active/Passive (SMAP) observations into an Antecedent Precipitation
    Index (API) model driven by Global Precipitation Measurement (GPM) precipitation
    via 4DVAR. It is NOT direct in-situ field probe measurements of root-zone moisture.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr

from src.utils.provenance import attach_provenance, create_provenance_metadata


# Canonical NCI THREDDS GRAFS endpoints
NCI_THREDDS_GRAFS_BASE_URL = "https://thredds.nci.org.au/thredds/dodsC/ub8/global/GRAFS"
DEFAULT_GRAFS_VARIABLES = ("s0", "s1")

VARIABLE_METADATA: Dict[str, Dict[str, Any]] = {
    "s0": {
        "standard_name": "surface_soil_relative_wetness",
        "long_name": "Topsoil Relative Wetness (0-5cm)",
        "units": "dimensionless",
        "valid_min": 0.0,
        "valid_max": 1.0,
        "depth": "0-5 cm",
        "remote_filename_pattern": "GRAFS_TopSoilRelativeWetness_{year}.nc",
        "description": "Satellite-guided topsoil relative wetness index",
    },
    "s1": {
        "standard_name": "rootzone_soil_water_index",
        "long_name": "Root-zone Soil Water Index (0-1m)",
        "units": "dimensionless",
        "valid_min": 0.0,
        "valid_max": 1.0,
        "depth": "0-1 m",
        "remote_filename_pattern": "GRAFS_RootzoneSoilWaterIndex_{year}.nc",
        "description": "Satellite-guided root-zone soil water index",
    },
}


def build_grafs_url(
    variable: str,
    year: int,
    base_url: str = NCI_THREDDS_GRAFS_BASE_URL,
) -> str:
    """Build the remote OPeNDAP URL for a specific GRAFS variable and year.

    Parameters
    ----------
    variable : str
        Variable identifier ('s0' for topsoil or 's1' for root-zone).
    year : int
        Calendar year for the dataset file.
    base_url : str, optional
        Base URL for the NCI THREDDS OPeNDAP catalog.

    Returns
    -------
    str
        Full OPeNDAP dataset URL.
    """
    var_lower = variable.lower()
    if var_lower in ("s0", "topsoil", "topsoilrelativewetness"):
        filename = f"GRAFS_TopSoilRelativeWetness_{year}.nc"
    elif var_lower in ("s1", "rootzone", "rootzonesoilwaterindex"):
        filename = f"GRAFS_RootzoneSoilWaterIndex_{year}.nc"
    else:
        raise ValueError(
            f"Unsupported GRAFS variable: '{variable}'. Expected 's0' or 's1'."
        )
    return f"{base_url.rstrip('/')}/{filename}"


def _open_dataset_with_retry(
    url: str,
    max_retries: int = 3,
    backoff_seconds: float = 1.5,
) -> xr.Dataset:
    """Open an OPeNDAP or local NetCDF dataset with bounded retries and descriptive error reporting.

    Parameters
    ----------
    url : str
        Remote OPeNDAP URL or local file path.
    max_retries : int, default 3
        Maximum retry attempts.
    backoff_seconds : float, default 1.5
        Base delay for exponential backoff between attempts.

    Returns
    -------
    xr.Dataset
        Opened xarray Dataset.

    Raises
    ------
    ConnectionError
        If all retry attempts fail, including the endpoint URL and cause.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            # decode_times=False to handle potential non-standard NetCDF time fill values safely
            ds = xr.open_dataset(url, decode_times=False)
            return ds
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))

    raise ConnectionError(
        f"Failed to access GRAFS OPeNDAP endpoint after {max_retries} attempts at URL: '{url}'. "
        f"Underlying error: {last_exc}"
    ) from last_exc


def _decode_grafs_time(ds: xr.Dataset) -> xr.Dataset:
    """Safely decode GRAFS time coordinate and filter out sentinel/fill values."""
    if "time" not in ds.coords:
        return ds

    time_raw = ds.time.values
    # Filter valid finite non-sentinel day counts (< 1e6 days)
    valid_mask = np.isfinite(time_raw) & (time_raw > 0) & (time_raw < 1e6)
    if not np.all(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        ds = ds.isel(time=valid_indices)
        time_raw = ds.time.values

    # Convert days since 1970-01-01 to pandas Timestamps
    time_origin = pd.Timestamp("1970-01-01")
    decoded_times = [time_origin + pd.Timedelta(days=float(d)) for d in time_raw]
    ds["time"] = decoded_times
    return ds


def validate_grafs_dataset(
    ds: xr.Dataset,
    variables: Optional[Sequence[str]] = None,
) -> None:
    """Validate data integrity, units, valid ranges, and temporal coordinates of a GRAFS dataset.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset containing GRAFS variables.
    variables : sequence of str, optional
        List of variable names to validate. If None, validates all present recognized variables.

    Raises
    ------
    ValueError
        If timestamps are invalid/non-monotonic, or if values fall outside physical bounds.
    """
    if "time" not in ds.coords and "time" not in ds.dims:
        raise ValueError("GRAFS dataset is missing 'time' coordinate.")

    time_coord = ds["time"]
    if len(time_coord) > 1:
        time_values = pd.to_datetime(time_coord.values)
        if not time_values.is_monotonic_increasing:
            raise ValueError("GRAFS time coordinate is not monotonically increasing.")

    check_vars = list(variables) if variables is not None else [v for v in ds.data_vars if v in VARIABLE_METADATA]

    for var_name in check_vars:
        if var_name not in ds:
            continue
        data_arr = ds[var_name]
        meta = VARIABLE_METADATA.get(var_name, {})

        # Unit verification
        units = data_arr.attrs.get("units", meta.get("units", "dimensionless"))
        if units not in ("dimensionless", "1", "%", "percent", "index", "fraction"):
            raise ValueError(
                f"Invalid units '{units}' for GRAFS variable '{var_name}'. "
                f"Expected dimensionless fraction [0.0, 1.0] or percentage [0, 100]."
            )

        # Range check on non-NaN values
        valid_min = meta.get("valid_min", 0.0)
        valid_max = 100.0 if units in ("%", "percent") else meta.get("valid_max", 1.0)

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


def load_grafs_data(
    bbox: Tuple[float, float, float, float],
    start_date: Union[str, datetime.date, datetime.datetime, pd.Timestamp],
    end_date: Union[str, datetime.date, datetime.datetime, pd.Timestamp],
    variables: Sequence[str] = DEFAULT_GRAFS_VARIABLES,
    source: Optional[Union[str, Path, xr.Dataset]] = None,
    base_url: str = NCI_THREDDS_GRAFS_BASE_URL,
    max_retries: int = 3,
) -> xr.Dataset:
    """Load and subset GRAFS root-zone and surface soil moisture data.

    Preserves native resolution (~0.1° / 10 km) and variable metadata.
    Handles missing values, decodes timestamps robustly, and validates ranges.

    Parameters
    ----------
    bbox : tuple of float
        Spatial bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    start_date : str or datetime
        Start date (inclusive).
    end_date : str or datetime
        End date (inclusive).
    variables : sequence of str, default ('s0', 's1')
        Variables to load: 's0' (Topsoil Relative Wetness) and/or 's1' (Rootzone Soil Water Index).
    source : str, Path, or xr.Dataset, optional
        Custom source path/URL or preloaded xr.Dataset (for testing and offline fixtures).
        If None, queries the official NCI THREDDS OPeNDAP service.
    base_url : str, default NCI_THREDDS_GRAFS_BASE_URL
        Base URL for remote THREDDS OPeNDAP access.
    max_retries : int, default 3
        Maximum connection retries for remote OPeNDAP access.

    Returns
    -------
    xr.Dataset
        Subsetted xarray Dataset containing requested variables at native resolution,
        with complete CF metadata, clean NaN masks, and verified timestamps.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if start_ts > end_ts:
        raise ValueError(f"start_date ({start_ts}) must not be after end_date ({end_ts}).")

    requested_vars = [v.lower() for v in variables]
    for v in requested_vars:
        if v not in ("s0", "s1"):
            raise ValueError(f"Unknown GRAFS variable '{v}'. Expected 's0' and/or 's1'.")

    # If an existing Dataset is passed (e.g. from local file / test fixture)
    if isinstance(source, xr.Dataset):
        ds_raw = source
    elif isinstance(source, (str, Path)) and Path(str(source)).exists():
        ds_raw = xr.open_dataset(source)
    else:
        # Remote OPeNDAP loading across required years with bounded retry resilience
        years = list(range(start_ts.year, end_ts.year + 1))
        yearly_datasets: List[xr.Dataset] = []

        for yr in years:
            var_datasets = []
            for var in requested_vars:
                url = build_grafs_url(var, yr, base_url=base_url)
                var_ds = _open_dataset_with_retry(url, max_retries=max_retries)
                var_ds = _decode_grafs_time(var_ds)

                # Standardize coordinate names
                coord_map = {}
                for c in var_ds.coords:
                    if c.lower() in ("latitude", "lats"):
                        coord_map[c] = "lat"
                    elif c.lower() in ("longitude", "lons"):
                        coord_map[c] = "lon"
                if coord_map:
                    var_ds = var_ds.rename(coord_map)

                # Normalize remote variable name (e.g. 'relative_wetness' -> 's0', 'soil_water_index' -> 's1')
                canonical_name = var
                renamed = False
                for name in list(var_ds.data_vars):
                    if name.lower() in ("relative_wetness", "topsoilrelativewetness", "rootzonesoilwaterindex", "wetness", "swi", "soil_water_index", "soilwaterindex", var.lower()):
                        var_ds = var_ds.rename({name: canonical_name})
                        renamed = True
                        break
                if not renamed and len(var_ds.data_vars) == 1:
                    orig_name = list(var_ds.data_vars)[0]
                    var_ds = var_ds.rename({orig_name: canonical_name})

                # Spatial subsetting at native resolution immediately at OPeNDAP level
                lat_vals = var_ds["lat"].values
                lat_ascending = bool(lat_vals[0] < lat_vals[-1]) if len(lat_vals) > 1 else True
                lat_step = abs(float(lat_vals[1] - lat_vals[0])) if len(lat_vals) > 1 else 0.1
                lon_step = abs(float(var_ds["lon"].values[1] - var_ds["lon"].values[0])) if len(var_ds["lon"].values) > 1 else 0.1

                adj_min_lat = min_lat - lat_step / 2.0
                adj_max_lat = max_lat + lat_step / 2.0
                adj_min_lon = min_lon - lon_step / 2.0
                adj_max_lon = max_lon + lon_step / 2.0

                lat_slice = slice(adj_min_lat, adj_max_lat) if lat_ascending else slice(adj_max_lat, adj_min_lat)
                lon_slice = slice(adj_min_lon, adj_max_lon)

                # Direct slice
                var_subset = var_ds[[canonical_name]].sel(lat=lat_slice, lon=lon_slice)
                if "time" in var_subset.coords:
                    var_subset = var_subset.sel(time=slice(start_ts, end_ts))
                var_subset = var_subset.load()
                var_datasets.append(var_subset)

            yr_merged = xr.merge(var_datasets, compat="override", join="override")
            yearly_datasets.append(yr_merged)

        ds_raw = xr.concat(yearly_datasets, dim="time") if len(yearly_datasets) > 1 else yearly_datasets[0]

    # Standardize coordinate names for preloaded/synthetic sources
    coord_map = {}
    for c in ds_raw.coords:
        if c.lower() in ("latitude", "lats"):
            coord_map[c] = "lat"
        elif c.lower() in ("longitude", "lons"):
            coord_map[c] = "lon"
    if coord_map:
        ds_raw = ds_raw.rename(coord_map)

    # Validate presence of lat/lon
    if "lat" not in ds_raw.coords or "lon" not in ds_raw.coords:
        raise ValueError("GRAFS dataset must contain 'lat' and 'lon' spatial coordinates.")

    # Spatial and temporal subsetting if source was preloaded
    if isinstance(source, (xr.Dataset, str, Path)):
        lat_vals = ds_raw["lat"].values
        lat_ascending = bool(lat_vals[0] < lat_vals[-1]) if len(lat_vals) > 1 else True
        lat_step = abs(float(lat_vals[1] - lat_vals[0])) if len(lat_vals) > 1 else 0.1
        lon_step = abs(float(ds_raw["lon"].values[1] - ds_raw["lon"].values[0])) if len(ds_raw["lon"].values) > 1 else 0.1

        adj_min_lat = min_lat - lat_step / 2.0
        adj_max_lat = max_lat + lat_step / 2.0
        adj_min_lon = min_lon - lon_step / 2.0
        adj_max_lon = max_lon + lon_step / 2.0

        lat_slice = slice(adj_min_lat, adj_max_lat) if lat_ascending else slice(adj_max_lat, adj_min_lat)
        lon_slice = slice(adj_min_lon, adj_max_lon)

        ds_spatial = ds_raw.sel(lat=lat_slice, lon=lon_slice)
        if "time" in ds_spatial.coords:
            ds_subset = ds_spatial.sel(time=slice(start_ts, end_ts))
        else:
            ds_subset = ds_spatial
    else:
        ds_subset = ds_raw

    # Mask fill values / nodata
    for var in requested_vars:
        if var in ds_subset:
            arr = ds_subset[var]
            if "_FillValue" in arr.attrs:
                fill_val = arr.attrs["_FillValue"]
                arr = arr.where(arr != fill_val)
            arr = arr.where((arr >= 0.0) & (arr <= 100.0))

            # Normalize 0-100 percentage to 0.0-1.0 fraction if needed
            if arr.size > 0:
                finite_arr = arr.values[np.isfinite(arr.values)]
                if len(finite_arr) > 0 and float(np.nanmax(finite_arr)) > 1.5:
                    arr = arr / 100.0
                    arr.attrs["units"] = "dimensionless"
            arr = arr.clip(0.0, 1.0)

            # Attach canonical metadata
            meta = VARIABLE_METADATA.get(var, {})
            arr.attrs.update({
                "standard_name": meta.get("standard_name", var),
                "long_name": meta.get("long_name", var),
                "units": "dimensionless",
                "valid_min": 0.0,
                "valid_max": 1.0,
                "depth": meta.get("depth", "unknown"),
                "source_system": "Global Root-zone moisture Analysis & Forecasting System (GRAFS)",
                "assimilation_method": "Satellite data assimilation (SMAP + GPM into API model via 4DVAR)",
                "field_measurement": False,
            })
            ds_subset[var] = arr

    # Attach provenance
    prov = create_provenance_metadata(
        source_name="ANU Centre for Water and Landscape Dynamics / NCI Australia",
        product_name="Global Root-zone moisture Analysis & Forecasting System (GRAFS)",
        product_version="v1.0",
        spatial_resolution="0.1 degree (~10 km native)",
        temporal_resolution="Daily",
        native_crs="EPSG:4326",
        known_limitations=[
            "Satellite-guided model assimilation product, not direct in-situ root-zone probe measurement",
            "Coarse 10 km spatial resolution cannot resolve within-field microtopography or tile drainage",
            "Relies on SMAP L-band microwave penetration depth (~5 cm) combined with API hydrologic model",
        ],
        source_url=base_url,
        transformations_applied=["spatial_bounding_box_subset", "temporal_subset", "fill_value_masking"],
    )
    attach_provenance(ds_subset, prov)

    # Validate dataset before return
    validate_grafs_dataset(ds_subset, variables=requested_vars)

    return ds_subset
