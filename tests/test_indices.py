"""Unit tests for resolution-aware biophysical feature engineering and index extraction."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from src.features.indices import (
    GridAlignmentError,
    calculate_evi,
    calculate_ndmi,
    calculate_ndvi,
    validate_index_distribution,
)


def test_ndvi_formula_exact() -> None:
    """Verify numerical correctness of NDVI formula (NIR - Red) / (NIR + Red + 1e-6)."""
    # Raw reflectance in float [0.0..1.0]
    red = xr.DataArray(np.array([[0.10, 0.20]]), coords={"y": [1.0], "x": [1.0, 2.0]}, dims=["y", "x"])
    nir = xr.DataArray(np.array([[0.50, 0.20]]), coords={"y": [1.0], "x": [1.0, 2.0]}, dims=["y", "x"])

    ndvi = calculate_ndvi(red, nir)

    # Pixel 0: (0.50 - 0.10) / (0.50 + 0.10 + 1e-6) = 0.40 / 0.600001 ≈ 0.666665
    assert pytest.approx(ndvi.values[0, 0], abs=1e-4) == 0.6667
    # Pixel 1: (0.20 - 0.20) / (0.20 + 0.20 + 1e-6) = 0.0
    assert pytest.approx(ndvi.values[0, 1], abs=1e-4) == 0.0

    # Test with uint16 integer scaled inputs (scale factor 10^-4)
    red_int = xr.DataArray(np.array([[1000, 2000]], dtype=np.uint16), coords={"y": [1.0], "x": [1.0, 2.0]}, dims=["y", "x"])
    nir_int = xr.DataArray(np.array([[5000, 2000]], dtype=np.uint16), coords={"y": [1.0], "x": [1.0, 2.0]}, dims=["y", "x"])
    ndvi_int = calculate_ndvi(red_int, nir_int)
    assert pytest.approx(ndvi_int.values[0, 0], abs=1e-4) == 0.6667


def test_evi_formula_exact() -> None:
    """Verify exact numerical formulation of 3-band EVI: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1.0)."""
    # NIR = 0.40, Red = 0.08, Blue = 0.04
    # Numerator = 2.5 * (0.40 - 0.08) = 2.5 * 0.32 = 0.80
    # Denominator = 0.40 + 6.0*0.08 - 7.5*0.04 + 1.0 = 0.40 + 0.48 - 0.30 + 1.0 = 1.58
    # Expected EVI = 0.80 / 1.58 ≈ 0.506329
    blue = xr.DataArray(np.array([[0.04]]), coords={"y": [1.0], "x": [1.0]}, dims=["y", "x"])
    red = xr.DataArray(np.array([[0.08]]), coords={"y": [1.0], "x": [1.0]}, dims=["y", "x"])
    nir = xr.DataArray(np.array([[0.40]]), coords={"y": [1.0], "x": [1.0]}, dims=["y", "x"])

    evi = calculate_evi(blue, red, nir)
    assert pytest.approx(evi.values[0, 0], abs=1e-4) == 0.5063

    # Test with scaled uint16 integers: Blue=400, Red=800, NIR=4000
    blue_int = xr.DataArray(np.array([[400]], dtype=np.uint16), coords={"y": [1.0], "x": [1.0]}, dims=["y", "x"])
    red_int = xr.DataArray(np.array([[800]], dtype=np.uint16), coords={"y": [1.0], "x": [1.0]}, dims=["y", "x"])
    nir_int = xr.DataArray(np.array([[4000]], dtype=np.uint16), coords={"y": [1.0], "x": [1.0]}, dims=["y", "x"])
    evi_int = calculate_evi(blue_int, red_int, nir_int)
    assert pytest.approx(evi_int.values[0, 0], abs=1e-4) == 0.5063


def test_ndmi_grid_safeguard() -> None:
    """Confirm calculate_ndmi enforces grid safeguards and performs area-weighted downsampling on valid inputs."""
    # 1. Valid 10m NIR (4x4) and 20m SWIR (2x2) in matching extent
    nir_y = np.linspace(100, 130, 4)
    nir_x = np.linspace(500, 530, 4)
    swir_y = np.linspace(105, 125, 2)
    swir_x = np.linspace(505, 525, 2)

    nir_valid = xr.DataArray(np.full((4, 4), 0.5, dtype=np.float32), coords={"y": nir_y, "x": nir_x}, dims=["y", "x"])
    swir_valid = xr.DataArray(np.full((2, 2), 0.2, dtype=np.float32), coords={"y": swir_y, "x": swir_x}, dims=["y", "x"])

    ndmi = calculate_ndmi(nir_valid, swir_valid)
    assert ndmi.sizes["y"] == 2 and ndmi.sizes["x"] == 2
    # (0.50 - 0.20) / (0.50 + 0.20 + 1e-6) = 0.30 / 0.70 ≈ 0.42857
    assert pytest.approx(ndmi.values[0, 0], abs=1e-4) == 0.4286

    # 2. Incompatible extents (no overlap)
    nir_disjoint = xr.DataArray(np.full((4, 4), 0.5), coords={"y": [10, 20, 30, 40], "x": [10, 20, 30, 40]}, dims=["y", "x"])
    with pytest.raises(GridAlignmentError, match="Spatial extents do not overlap"):
        calculate_ndmi(nir_disjoint, swir_valid)

    # 3. Invalid resolution ratio (e.g. 10:1 instead of ~2:1)
    nir_huge = xr.DataArray(np.full((20, 20), 0.5), coords={"y": np.linspace(105, 125, 20), "x": np.linspace(505, 525, 20)}, dims=["y", "x"])
    with pytest.raises(GridAlignmentError, match="Resolution ratio violation"):
        calculate_ndmi(nir_huge, swir_valid)


def test_validate_index_distribution() -> None:
    """Verify that validate_index_distribution audits values, NaNs, and bounds without altering data."""
    data = np.array([[0.2, 0.5, np.nan], [-1.5, 1.2, 0.8]])
    da = xr.DataArray(data, coords={"y": [1.0, 2.0], "x": [10.0, 20.0, 30.0]}, dims=["y", "x"], name="ndvi")

    report = validate_index_distribution(da, index_name="NDVI", expected_bounds=(-1.0, 1.0))

    assert report["index_name"] == "NDVI"
    assert report["total_pixels"] == 6
    assert report["valid_pixels"] == 5
    assert report["nan_pixels"] == 1
    assert report["nan_pct"] == round((1 / 6) * 100.0, 2)
    assert report["min"] == -1.5
    assert report["max"] == 1.2
    assert report["out_of_bounds_count"] == 2  # -1.5 and 1.2
    assert report["out_of_bounds_pct"] == round((2 / 5) * 100.0, 2)
