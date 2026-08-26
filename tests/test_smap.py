"""Tests for NASA SMAP Level-4 soil moisture ingestion loader (src/ingestion/smap.py).

Verifies:
1. Synthetic dataset loading with 3-hourly to daily aggregation.
2. Temporal continuity and alignment across all 15 canonical 2023 Long Rains bins.
3. Physical value bounds validation [0.0, 1.0].
4. Monotonic time coordinate validation.
5. Explicit SMAP_L4 provenance tracking.
6. Seamless feeding into downstream hydrology features (resampling and delta_swi_14d).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.features.hydrology import (
    calculate_swi_14d_change,
    resample_soil_moisture_to_calendar,
)
from src.ingestion.smap import (
    load_smap_data,
    validate_smap_dataset,
)


def _create_synthetic_smap_dataset(
    start_date: str = "2023-03-01",
    end_date: str = "2023-09-30",
    freq: str = "3h",
    lat_range: tuple = (0.50, 0.80),
    lon_range: tuple = (35.10, 35.40),
    res: float = 0.08,
) -> xr.Dataset:
    """Create a synthetic 3-hourly SMAP L4 dataset mimicking SPL4SMGP V008."""
    times = pd.date_range(start_date, end_date, freq=freq)
    lats = np.arange(lat_range[0], lat_range[1] + res / 2, res)
    lons = np.arange(lon_range[0], lon_range[1] + res / 2, res)

    np.random.seed(42)
    shape = (len(times), len(lats), len(lons))
    # Relative saturation in [0.15, 0.75]
    sm_rootzone_wetness = np.random.uniform(0.20, 0.70, size=shape).astype(np.float32)
    sm_surface_wetness = np.random.uniform(0.15, 0.80, size=shape).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "sm_rootzone_wetness": (["time", "lat", "lon"], sm_rootzone_wetness),
            "sm_surface_wetness": (["time", "lat", "lon"], sm_surface_wetness),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "product_id": "SPL4SMGP",
            "version": "008",
            "source": "NASA SMAP Level-4",
        },
    )
    return ds


def test_load_smap_data_synthetic_daily_aggregation():
    """Verify 3-hourly SMAP data is properly aggregated to daily means with s0/s1 variable mapping."""
    bbox = (35.15, 0.55, 35.35, 0.75)
    ds_syn = _create_synthetic_smap_dataset(start_date="2023-04-01", end_date="2023-04-05", freq="3h")

    ds_smap = load_smap_data(
        bbox=bbox,
        start_date="2023-04-01",
        end_date="2023-04-05",
        source=ds_syn,
    )

    assert "s0" in ds_smap
    assert "s1" in ds_smap
    assert ds_smap.attrs.get("source") == "SMAP_L4"
    assert "provenance_product_name" in ds_smap.attrs
    assert "SMAP L4" in ds_smap.attrs["provenance_product_name"]

    # Daily aggregation check: 5 days requested -> 5 daily time slices
    assert len(ds_smap.time) == 5
    # Value bounds check
    assert (ds_smap.s1.values >= 0.0).all()
    assert (ds_smap.s1.values <= 1.0).all()


def test_smap_populates_all_15_canonical_2023_bins():
    """Verify continuous 2023 SMAP data populates all 15 canonical 14-day production bins seamlessly."""
    bbox = (35.15, 0.55, 35.35, 0.75)
    # Full Long Rains 2023 window
    ds_syn = _create_synthetic_smap_dataset(start_date="2023-03-01", end_date="2023-09-30", freq="3h")

    ds_smap = load_smap_data(
        bbox=bbox,
        start_date="2023-03-01",
        end_date="2023-09-30",
        source=ds_syn,
    )

    # 15 canonical 14-day bin starts
    target_time_bins = [
        pd.Timestamp(d)
        for d in [
            "2023-03-01", "2023-03-15", "2023-03-29", "2023-04-12", "2023-04-26",
            "2023-05-10", "2023-05-24", "2023-06-07", "2023-06-21", "2023-07-05",
            "2023-07-19", "2023-08-02", "2023-08-16", "2023-08-30", "2023-09-13",
        ]
    ]

    # Pass to existing hydrology temporal resampler
    ds_resampled = resample_soil_moisture_to_calendar(ds_smap, target_time_bins)
    assert len(ds_resampled.time) == 15
    assert list(pd.to_datetime(ds_resampled.time.values)) == target_time_bins

    # Calculate delta_swi_14d
    ds_delta = calculate_swi_14d_change(ds_resampled)
    assert "delta_swi_14d" in ds_delta
    delta_vals = ds_delta.delta_swi_14d.values

    # t=0 is NaN, bins 1..14 are finite continuous rates of change
    assert np.isnan(delta_vals[0]).all()
    for b in range(1, 15):
        assert np.isfinite(delta_vals[b]).all(), f"Bin {b} contains non-finite values"


def test_validate_smap_dataset_physical_bounds():
    """Verify validate_smap_dataset raises ValueError when physical range exceeds [0, 1]."""
    ds = _create_synthetic_smap_dataset(start_date="2023-04-01", end_date="2023-04-02")
    # Remap names to s0, s1
    ds = ds.rename({"sm_surface_wetness": "s0", "sm_rootzone_wetness": "s1"})

    # Introduce invalid value
    ds.s1.values[0, 0, 0] = 1.50
    with pytest.raises(ValueError, match="Physical value range violation"):
        validate_smap_dataset(ds, variables=["s1"])


def test_validate_smap_dataset_non_monotonic_time():
    """Verify validate_smap_dataset raises ValueError on non-monotonic timestamps."""
    ds = _create_synthetic_smap_dataset(start_date="2023-04-01", end_date="2023-04-05")
    ds = ds.rename({"sm_surface_wetness": "s0", "sm_rootzone_wetness": "s1"})

    # Reorder times non-monotonically
    times_shuffled = ds.time.values.copy()
    times_shuffled[0], times_shuffled[1] = times_shuffled[1], times_shuffled[0]
    ds["time"] = times_shuffled

    with pytest.raises(ValueError, match="not monotonically increasing"):
        validate_smap_dataset(ds, variables=["s1"])
