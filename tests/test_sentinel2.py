"""Unit tests for Sentinel-2 L2A optical ingestion loader."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.ingestion.sentinel2 import load_sentinel2_data, validate_sentinel2_dataset


@pytest.fixture
def synthetic_sentinel2_dataset() -> xr.Dataset:
    """Create synthetic Sentinel-2 optical bands dataset with uint16 reflectance and uint8 SCL."""
    times = pd.date_range("2024-04-01", "2024-04-20", freq="5D")
    lats = np.linspace(0.4, 0.6, 20)
    lons = np.linspace(35.2, 35.4, 20)

    np.random.seed(42)
    b02 = np.random.randint(200, 1500, size=(len(times), len(lats), len(lons)), dtype=np.uint16)
    b04 = np.random.randint(300, 2000, size=(len(times), len(lats), len(lons)), dtype=np.uint16)
    b08 = np.random.randint(2500, 6500, size=(len(times), len(lats), len(lons)), dtype=np.uint16)
    scl = np.random.choice([4, 5, 7], size=(len(times), len(lats), len(lons))).astype(np.uint8)

    ds = xr.Dataset(
        data_vars={
            "B02": (["time", "lat", "lon"], b02),
            "B04": (["time", "lat", "lon"], b04),
            "B08": (["time", "lat", "lon"], b08),
            "SCL": (["time", "lat", "lon"], scl),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    return ds


def test_load_sentinel2_data_synthetic(synthetic_sentinel2_dataset: xr.Dataset) -> None:
    """Verify Sentinel-2 loader subsets bands and attaches provenance."""
    bbox = (35.2, 0.4, 35.4, 0.6)
    ds = load_sentinel2_data(
        bbox=bbox,
        start_date="2024-04-01",
        end_date="2024-04-10",
        bands=["B04", "B08", "SCL"],
        source=synthetic_sentinel2_dataset,
    )
    assert "B04" in ds
    assert "B08" in ds
    assert "SCL" in ds
    assert "B02" not in ds
    assert len(ds.time) == 2  # April 1 and April 6
    assert str(ds.B04.dtype) == "uint16"
    assert str(ds.SCL.dtype) == "uint8"
    assert "provenance_source_name" in ds.attrs
    assert "10" in ds.attrs["provenance_spatial_resolution"]


def test_validate_sentinel2_out_of_bounds() -> None:
    """Verify that reflectance outside physical bounds raises ValueError."""
    bad_ds = xr.Dataset(
        data_vars={"B04": (["time", "lat", "lon"], np.array([[[1.9]]], dtype=np.float32))},
        coords={"time": [pd.Timestamp("2024-01-01")], "lat": [0.0], "lon": [35.0]},
    )
    with pytest.raises(ValueError, match="Reflectance for band .* out of physical range"):
        validate_sentinel2_dataset(bad_ds)
