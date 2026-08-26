"""Unit tests for optical preprocessing, SCL masking, compositing, and diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.preprocessing.optical import apply_scl_mask
from src.preprocessing.masking import apply_cropland_mask
from src.preprocessing.compositing import composite_14d, smooth_temporal_series
from src.preprocessing.diagnostics import generate_quality_report


@pytest.fixture
def synthetic_s2_stack() -> xr.Dataset:
    """Create synthetic multi-temporal Sentinel-2 optical dataset with SCL."""
    times = pd.date_range("2023-04-01", periods=6, freq="5D")
    y = np.linspace(100000, 100100, 10)
    x = np.linspace(500000, 500100, 10)

    # Base reflectance (uint16 scaled integer [0..10000])
    b04_data = np.full((len(times), len(y), len(x)), 1000, dtype=np.uint16)
    b08_data = np.full((len(times), len(y), len(x)), 4000, dtype=np.uint16)

    # SCL: time 0 all class 4 (veg), time 1 all class 9 (cloud), time 2 mixed [4, 5, 7, 3, 0]
    scl_data = np.full((len(times), len(y), len(x)), 4, dtype=np.uint8)
    scl_data[1, :, :] = 9  # High prob cloud
    scl_data[2, 0, 0] = 0  # No data
    scl_data[2, 0, 1] = 3  # Cloud shadow
    scl_data[2, 0, 2] = 7  # Unclassified
    scl_data[2, 0, 3] = 4  # Vegetation
    scl_data[2, 0, 4] = 5  # Bare soil
    scl_data[2, 0, 5] = 6  # Water

    return xr.Dataset(
        data_vars={
            "B04": (["time", "y", "x"], b04_data),
            "B08": (["time", "y", "x"], b08_data),
            "SCL": (["time", "y", "x"], scl_data),
        },
        coords={"time": times, "y": y, "x": x},
    )


def test_scl_mask_classes(synthetic_s2_stack: xr.Dataset) -> None:
    """Verify SCL masking retains classes 4, 5, 6 and sets 0, 1, 2, 3, 7, 8, 9, 10, 11 to NaN."""
    ds_masked = apply_scl_mask(synthetic_s2_stack, valid_classes=[4, 5, 6])

    # Time 0: all class 4 -> all finite
    assert np.all(np.isfinite(ds_masked.B04.isel(time=0).values))

    # Time 1: all class 9 -> all NaN
    assert np.all(np.isnan(ds_masked.B04.isel(time=1).values))

    # Time 2 specific pixels:
    t2_b04 = ds_masked.B04.isel(time=2).values
    assert np.isnan(t2_b04[0, 0])  # Class 0 (No data) -> NaN
    assert np.isnan(t2_b04[0, 1])  # Class 3 (Cloud shadow) -> NaN
    assert np.isnan(t2_b04[0, 2])  # Class 7 (Unclassified) -> NaN
    assert np.isfinite(t2_b04[0, 3])  # Class 4 (Vegetation) -> Valid
    assert np.isfinite(t2_b04[0, 4])  # Class 5 (Bare soil) -> Valid
    assert np.isfinite(t2_b04[0, 5])  # Class 6 (Water) -> Valid


def test_valid_obs_count_tracking(synthetic_s2_stack: xr.Dataset) -> None:
    """Verify valid_obs_count accurately tallies non-null pixel-time observations per 14-day bin."""
    ds_masked = apply_scl_mask(synthetic_s2_stack, valid_classes=[4, 5, 6])
    b04_masked = ds_masked.B04

    comp_da, valid_obs = composite_14d(b04_masked, freq="14D")

    assert valid_obs.name == "valid_obs_count"
    assert valid_obs.attrs["units"] == "count"

    # Bin 0 spans April 1 to April 14 (contains April 1 [valid], April 6 [cloud/masked], April 11 [mostly valid])
    obs_b0 = valid_obs.isel(time=0).values
    # Pixel (0, 0): April 1 is valid (1), April 6 is cloud (0), April 11 is class 0 (0) -> total = 1
    assert obs_b0[0, 0] == 1
    # Pixel (0, 3): April 1 is valid (1), April 6 is cloud (0), April 11 is class 4 (1) -> total = 2
    assert obs_b0[0, 3] == 2


def test_no_implicit_gap_filling() -> None:
    """Confirm an empty composite bin (NaN) surrounded by valid bins remains NaN after smooth_temporal_series()."""
    times = pd.date_range("2023-04-01", periods=5, freq="14D")
    y = [100.0]
    x = [200.0]

    # Create composite with missing middle bin: [0.5, 0.6, NaN, 0.7, 0.8]
    data = np.array([[[0.5]], [[0.6]], [[np.nan]], [[0.7]], [[0.8]]])
    comp_da = xr.DataArray(data, coords={"time": times, "y": y, "x": x}, dims=["time", "y", "x"], name="ndvi")

    smoothed = smooth_temporal_series(comp_da, window=3, min_periods=1)

    # Bin index 2 MUST strictly remain NaN (no implicit gap interpolation)
    assert np.isnan(smoothed.isel(time=2).values[0, 0])
    # Adjacent bins must have smoothed finite values
    assert np.isfinite(smoothed.isel(time=1).values[0, 0])
    assert np.isfinite(smoothed.isel(time=3).values[0, 0])


def test_cropland_mask_configurable() -> None:
    """Verify cropland masking behavior with configurable cultivated_value."""
    y = np.linspace(100, 110, 3)
    x = np.linspace(200, 210, 3)

    feature_data = np.array([[0.2, 0.5, 0.8], [0.3, 0.6, 0.9], [0.4, 0.7, 0.1]])
    feature_da = xr.DataArray(feature_data, coords={"y": y, "x": x}, dims=["y", "x"], name="ndvi")

    # Crop mask with values {0: urban, 1: maize, 2: forest}
    mask_data = np.array([[1, 0, 2], [1, 1, 0], [0, 2, 1]])
    mask_da = xr.DataArray(mask_data, coords={"y": y, "x": x}, dims=["y", "x"], name="crop_mask")

    # Filter with cultivated_value=1 (default)
    masked_1 = apply_cropland_mask(feature_da, mask_da, cultivated_value=1)
    assert np.isfinite(masked_1.values[0, 0])
    assert np.isnan(masked_1.values[0, 1])  # Urban -> NaN
    assert np.isnan(masked_1.values[0, 2])  # Forest -> NaN

    # Filter with custom cultivated_value=2 (e.g. specialized forest mask)
    masked_2 = apply_cropland_mask(feature_da, mask_da, cultivated_value=2)
    assert np.isnan(masked_2.values[0, 0])
    assert np.isfinite(masked_2.values[0, 2])  # Forest -> Valid


def test_quality_diagnostics_report(synthetic_s2_stack: xr.Dataset) -> None:
    """Verify generate_quality_report calculates rejection rate, scene stats, and unfilled bins."""
    ds_masked = apply_scl_mask(synthetic_s2_stack, valid_classes=[4, 5, 6])
    comp_da, valid_obs = composite_14d(ds_masked.B04, freq="14D")

    report = generate_quality_report(
        raw_ds=synthetic_s2_stack,
        masked_ds=ds_masked,
        composite_da=comp_da,
        valid_obs_count=valid_obs,
    )

    assert report["scene_count"] == 6
    assert len(report["acquisition_dates"]) == 6
    assert 0.0 <= report["rejection_rate_scl"] <= 100.0
    assert report["rejection_rate_scl"] > 0.0  # Because time 1 was completely cloud
    assert report["mean_valid_obs_per_bin"] >= 0.0
    assert 0.0 <= report["unfilled_bins_pct"] <= 100.0
