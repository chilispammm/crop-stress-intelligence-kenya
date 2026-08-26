"""Unit tests for agricultural / cropland mask loader."""

import numpy as np
import pytest
import xarray as xr

from src.ingestion.mask import load_crop_mask, validate_crop_mask


@pytest.fixture
def synthetic_mask_array() -> xr.DataArray:
    """Create synthetic binary agricultural mask."""
    lats = np.linspace(0.4, 0.6, 20)
    lons = np.linspace(35.2, 35.4, 20)

    np.random.seed(55)
    mask_vals = np.random.choice([0.0, 1.0], size=(len(lats), len(lons)))
    return xr.DataArray(mask_vals, coords={"lat": lats, "lon": lons}, dims=["lat", "lon"], name="crop_mask")


def test_load_crop_mask_synthetic(synthetic_mask_array: xr.DataArray) -> None:
    """Verify crop mask loader correctly normalizes binary values and attaches metadata."""
    bbox = (35.2, 0.4, 35.4, 0.6)
    da = load_crop_mask(bbox=bbox, source=synthetic_mask_array)
    assert da.name == "crop_mask"
    assert "provenance_source_name" in da.attrs
    assert "maize_qualification" in da.attrs
    assert da.attrs["provenance_spatial_resolution"] == "20 m native"
    assert set(np.unique(da.values)).issubset({0.0, 1.0})


def test_validate_crop_mask_non_binary() -> None:
    """Verify non-binary values raise ValueError."""
    bad_da = xr.DataArray([[2.0, 0.0], [1.0, 5.0]], dims=["lat", "lon"])
    with pytest.raises(ValueError, match="Crop mask must be binary"):
        validate_crop_mask(bad_da)
