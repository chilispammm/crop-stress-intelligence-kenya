"""Unit tests for GRAFS ingestion loader."""

import datetime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.ingestion.grafs import (
    build_grafs_url,
    load_grafs_data,
    validate_grafs_dataset,
)


@pytest.fixture
def synthetic_grafs_dataset() -> xr.Dataset:
    """Create an offline synthetic GRAFS xarray Dataset at native 0.1 deg (~10 km) resolution."""
    times = pd.date_range(start="2024-04-01", end="2024-04-10", freq="D")
    lats = np.linspace(-0.5, 1.5, 21)  # ~0.1 deg step covering Uasin Gishu & Western Kenya
    lons = np.linspace(34.5, 36.0, 16)  # ~0.1 deg step

    # s0: topsoil relative wetness [0.1, 0.9]
    np.random.seed(42)
    s0_data = np.random.uniform(0.15, 0.85, size=(len(times), len(lats), len(lons)))
    # s1: rootzone soil water index [0.2, 0.8]
    s1_data = np.random.uniform(0.25, 0.75, size=(len(times), len(lats), len(lons)))

    ds = xr.Dataset(
        data_vars={
            "s0": (["time", "lat", "lon"], s0_data, {"units": "dimensionless", "_FillValue": -9999.0}),
            "s1": (["time", "lat", "lon"], s1_data, {"units": "dimensionless", "_FillValue": -9999.0}),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )
    return ds


def test_build_grafs_url() -> None:
    """Verify OPeNDAP URL construction for s0 and s1."""
    url_s0 = build_grafs_url("s0", 2024)
    assert "GRAFS_TopSoilRelativeWetness_2024.nc" in url_s0
    assert url_s0.startswith("https://thredds.nci.org.au/thredds/dodsC/ub8/global/GRAFS")

    url_s1 = build_grafs_url("s1", 2024)
    assert "GRAFS_RootzoneSoilWaterIndex_2024.nc" in url_s1

    with pytest.raises(ValueError, match="Unsupported GRAFS variable"):
        build_grafs_url("invalid_var", 2024)


def test_load_grafs_data_synthetic(synthetic_grafs_dataset: xr.Dataset) -> None:
    """Verify loader correctly subsets spatially and temporally while preserving metadata."""
    bbox = (35.0, 0.0, 35.8, 1.0)  # Uasin Gishu approximate bounds
    start_date = "2024-04-03"
    end_date = "2024-04-07"

    ds_subset = load_grafs_data(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        variables=["s0", "s1"],
        source=synthetic_grafs_dataset,
    )

    assert "s0" in ds_subset
    assert "s1" in ds_subset
    assert len(ds_subset.time) == 5  # April 3 to 7 inclusive
    assert ds_subset.lat.min() >= 0.0
    assert ds_subset.lat.max() <= 1.0
    assert ds_subset.lon.min() >= 35.0
    assert ds_subset.lon.max() <= 35.8

    # Metadata & scientific distinction checks
    assert ds_subset["s0"].attrs["field_measurement"] is False
    assert ds_subset["s1"].attrs["field_measurement"] is False
    assert "Topsoil" in ds_subset["s0"].attrs["long_name"]
    assert "Root-zone" in ds_subset["s1"].attrs["long_name"]
    assert ds_subset["s0"].attrs["units"] == "dimensionless"
    assert ds_subset["s1"].attrs["units"] == "dimensionless"


def test_validate_grafs_dataset_invalid_range() -> None:
    """Verify that values violating physical range trigger ValueError."""
    times = pd.date_range("2024-01-01", periods=2)
    lats = [0.0, 0.1]
    lons = [35.0, 35.1]
    # Out of physical bound: 2.5 (should be [0, 1])
    bad_data = np.array([[[2.5, 0.5], [0.4, 0.3]], [[0.5, 0.2], [0.3, 0.1]]])
    ds_bad = xr.Dataset(
        data_vars={"s0": (["time", "lat", "lon"], bad_data, {"units": "dimensionless"})},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    with pytest.raises(ValueError, match="Physical value range violation"):
        validate_grafs_dataset(ds_bad, variables=["s0"])


def test_validate_grafs_dataset_non_monotonic_time() -> None:
    """Verify that non-monotonic timestamps trigger ValueError."""
    bad_times = [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-02")]
    ds_bad = xr.Dataset(
        data_vars={"s0": (["time", "lat", "lon"], np.ones((2, 1, 1)) * 0.5, {"units": "dimensionless"})},
        coords={"time": bad_times, "lat": [0.0], "lon": [35.0]},
    )
    with pytest.raises(ValueError, match="not monotonically increasing"):
        validate_grafs_dataset(ds_bad, variables=["s0"])
