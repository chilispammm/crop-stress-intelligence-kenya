"""Climate Hazards Center InfraRed Precipitation with Station data (CHIRPS) ingestion loader.

Supports both monthly data (for long-term climatology / anomaly baselines) and
daily data (for high-frequency rainfall deficit tracking) via Digital Earth Africa
STAC catalog at native 0.05° (~5.5 km) resolution.

Scientific Note:
    CHIRPS blends satellite infrared cold cloud duration observations with in-situ
    station records. It is retained at its native 0.05° resolution and must NOT be
    resampled to 20 m.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from src.utils.provenance import attach_provenance, create_provenance_metadata


DEFAULT_DEAFRICA_STAC_URL = "https://explorer.digitalearth.africa/stac"
CHIRPS_COLLECTION_MONTHLY = "rainfall_chirps_monthly"
CHIRPS_COLLECTION_DAILY = "rainfall_chirps_daily"
DEFAULT_CHIRPS_VARIABLES = ("rainfall", "precip")


def validate_chirps_dataset(ds: xr.Dataset) -> None:
    """Validate CHIRPS rainfall dataset coordinates, units, and ranges.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset containing CHIRPS precipitation data.

    Raises
    ------
    ValueError
        If timestamps are invalid/non-monotonic or if precipitation values violate physical bounds.
    """
    if "time" not in ds.coords and "time" not in ds.dims:
        raise ValueError("CHIRPS dataset is missing 'time' coordinate.")

    time_coord = ds["time"]
    if len(time_coord) > 1:
        time_values = pd.to_datetime(time_coord.values)
        if not time_values.is_monotonic_increasing:
            raise ValueError("CHIRPS time coordinate is not monotonically increasing.")

    var_name = "rainfall" if "rainfall" in ds else ("precip" if "precip" in ds else None)
    if var_name:
        arr = ds[var_name]
        units = arr.attrs.get("units", "mm")
        if units not in ("mm", "mm/day", "mm/month", "mm/pentad", "kg m-2"):
            raise ValueError(f"Invalid CHIRPS precipitation units: '{units}'. Expected 'mm' or 'mm/month'.")

        data_vals = arr.values
        finite_vals = data_vals[np.isfinite(data_vals)]
        if len(finite_vals) > 0:
            min_val = float(np.min(finite_vals))
            max_val = float(np.max(finite_vals))
            if min_val < -0.01:
                raise ValueError(f"Precipitation cannot be negative, found min value: {min_val:.3f}")
            if max_val > 1500.0:  # 1500 mm monthly extreme physical limit
                raise ValueError(f"Precipitation exceeds extreme physical limit (1500 mm): {max_val:.3f}")


def load_chirps_data(
    bbox: Tuple[float, float, float, float],
    start_date: Union[str, datetime.date, datetime.datetime, pd.Timestamp],
    end_date: Union[str, datetime.date, datetime.datetime, pd.Timestamp],
    frequency: str = "monthly",
    stac_url: str = DEFAULT_DEAFRICA_STAC_URL,
    source: Optional[Union[str, Path, xr.Dataset]] = None,
) -> xr.Dataset:
    """Load and subset CHIRPS precipitation data at native ~5.5 km (0.05°) resolution.

    Parameters
    ----------
    bbox : tuple of float
        Spatial bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    start_date : str or datetime
        Start date (inclusive).
    end_date : str or datetime
        End date (inclusive).
    frequency : str, default 'monthly'
        Aggregation frequency: 'monthly' (for baseline climatology) or 'daily' (for daily events).
    stac_url : str, default 'https://explorer.digitalearth.africa/stac'
        STAC catalog endpoint URL.
    source : str, Path, or xr.Dataset, optional
        Local dataset or preloaded xarray Dataset (for offline operation/testing).

    Returns
    -------
    xr.Dataset
        Dataset containing precipitation at native 0.05° scale with full provenance metadata.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if start_ts > end_ts:
        raise ValueError(f"start_date ({start_ts}) must not be after end_date ({end_ts}).")

    freq_lower = frequency.lower()
    if freq_lower not in ("monthly", "daily"):
        raise ValueError(f"Unsupported frequency: '{frequency}'. Expected 'monthly' or 'daily'.")

    collection = CHIRPS_COLLECTION_MONTHLY if freq_lower == "monthly" else CHIRPS_COLLECTION_DAILY

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
                    f"No CHIRPS items found in collection '{collection}' for bbox {bbox} "
                    f"and datetime range {date_range} on endpoint '{stac_url}'."
                )

            with rasterio.Env(
                AWS_NO_SIGN_REQUEST="YES",
                AWS_DEFAULT_REGION="af-south-1",
                AWS_REGION="af-south-1",
            ):
                ds_raw = odc.stac.load(
                    items,
                    bbox=[min_lon, min_lat, max_lon, max_lat],
                )

        except Exception as exc:
            raise ConnectionError(
                f"Failed to query or load live CHIRPS data from STAC endpoint '{stac_url}': {exc}"
            ) from exc

    # Standardize coordinate names
    coord_map = {}
    for c in ds_raw.coords:
        if c.lower() in ("latitude", "lats"):
            coord_map[c] = "lat"
        elif c.lower() in ("longitude", "lons"):
            coord_map[c] = "lon"
    if coord_map:
        ds_raw = ds_raw.rename(coord_map)

    # Standardize variable name to 'rainfall'
    canonical_var = "rainfall"
    if canonical_var not in ds_raw.data_vars:
        for name in list(ds_raw.data_vars):
            if name.lower() in ("precip", "precipitation", "rain"):
                ds_raw = ds_raw.rename({name: canonical_var})
                break

    if canonical_var not in ds_raw.data_vars:
        raise ValueError(f"CHIRPS dataset does not contain '{canonical_var}', found {list(ds_raw.data_vars)}")

    # Spatial slicing if source was pre-loaded
    if "lat" in ds_raw.coords and "lon" in ds_raw.coords:
        lat_vals = ds_raw["lat"].values
        lat_ascending = bool(lat_vals[0] < lat_vals[-1]) if len(lat_vals) > 1 else True
        lat_slice = slice(min_lat, max_lat) if lat_ascending else slice(max_lat, min_lat)
        lon_slice = slice(min_lon, max_lon)
        ds_spatial = ds_raw.sel(lat=lat_slice, lon=lon_slice)
    else:
        ds_spatial = ds_raw

    # Temporal slicing if source was preloaded
    if "time" in ds_spatial.coords:
        ds_subset = ds_spatial.sel(time=slice(start_ts, end_ts))
    else:
        ds_subset = ds_spatial

    # Mask fill values / nodata
    arr = ds_subset[canonical_var]
    if "_FillValue" in arr.attrs:
        arr = arr.where(arr != arr.attrs["_FillValue"])
    arr = arr.where(arr >= 0.0)
    arr.attrs.update({
        "standard_name": "precipitation_amount",
        "long_name": f"CHIRPS {frequency.capitalize()} Precipitation",
        "units": "mm/month" if freq_lower == "monthly" else "mm/day",
        "spatial_resolution": "0.05 degree (~5.5 km native)",
    })
    ds_subset[canonical_var] = arr

    # Attach provenance
    prov = create_provenance_metadata(
        source_name="Climate Hazards Center, UC Santa Barbara / Digital Earth Africa",
        product_name=f"CHIRPS ({frequency.capitalize()} Precipitation)",
        product_version="v2.0",
        spatial_resolution="0.05 degree (~5.5 km native)",
        temporal_resolution=frequency.capitalize(),
        native_crs="EPSG:4326",
        known_limitations=[
            "Coarse ~5.5 km grid cannot capture localized convective rainfall variability",
            "Gauge station density in Western Kenya varies over historical baseline",
        ],
        source_url=stac_url,
        collection_id=collection,
        transformations_applied=["spatial_bounding_box_subset", "temporal_subset"],
    )
    attach_provenance(ds_subset, prov)

    validate_chirps_dataset(ds_subset)
    return ds_subset
