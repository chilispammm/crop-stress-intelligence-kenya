"""Unit tests for CHIRPS precipitation ingestion loader."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.ingestion.chirps import load_chirps_data, validate_chirps_dataset


@pytest.fixture
def synthetic_chirps_dataset() -> xr.Dataset:
    """Create synthetic native 0.05 deg (~5.5 km) CHIRPS precipitation dataset."""
    times = pd.date_range("2024-04-01", "2024-04-10", freq="D")
    lats = np.linspace(0.0, 1.0, 21)  # 0.05 deg step
    lons = np.linspace(35.0, 36.0, 21)  # 0.05 deg step

    np.random.seed(101)
    precip_data = np.random.uniform(0.0, 45.0, size=(len(times), len(lats), len(lons)))

    ds = xr.Dataset(
        data_vars={"rainfall": (["time", "lat", "lon"], precip_data, {"units": "mm/day", "_FillValue": -9999.0})},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    return ds


def test_load_chirps_data_synthetic(synthetic_chirps_dataset: xr.Dataset) -> None:
    """Verify CHIRPS loader extracts spatial/temporal bounds and attaches provenance."""
    bbox = (35.1, 0.2, 35.8, 0.8)
    ds = load_chirps_data(
        bbox=bbox,
        start_date="2024-04-03",
        end_date="2024-04-06",
        frequency="daily",
        source=synthetic_chirps_dataset,
    )
    assert "rainfall" in ds
    assert len(ds.time) == 4
    assert ds["rainfall"].attrs["units"] == "mm/day"
    assert "provenance_source_name" in ds.attrs
    assert ds.attrs["provenance_spatial_resolution"] == "0.05 degree (~5.5 km native)"


def test_validate_chirps_negative_precip() -> None:
    """Verify that negative precipitation raises ValueError."""
    bad_ds = xr.Dataset(
        data_vars={"rainfall": (["time", "lat", "lon"], [[[-5.0]]], {"units": "mm/day"})},
        coords={"time": [pd.Timestamp("2024-01-01")], "lat": [0.0], "lon": [35.0]},
    )
    with pytest.raises(ValueError, match="Precipitation cannot be negative"):
        validate_chirps_dataset(bad_ds)
