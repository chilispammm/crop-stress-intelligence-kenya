"""Unit tests for Multi-Modal Baseline & Standardized Anomaly Engine (Milestone 4.1 Audit Suite).

All tests are strictly offline, using synthetic in-memory xarray structures, with
zero network dependencies.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
import rioxarray
import xarray as xr

from src.features.hydrology import (
    calculate_swi_14d_change,
    resample_soil_moisture_to_calendar,
)
from src.features.rainfall_anomalies import (
    calculate_rainfall_climatology,
    calculate_rainfall_zscore,
)
from src.features.vegetation_anomalies import (
    aggregate_target_to_climatology_grid,
    calculate_ndvi_zscore,
    filter_ndvi_climatology_qa,
)
from src.utils.exceptions import DataCompletenessError


def test_rainfall_climatology_parameterized_continuity() -> None:
    """Verify parameterized CHIRPS baseline continuity validation across various spans and flaws."""
    # 1. Custom baseline span: 1995-2005 (11 years * 12 months = 132 continuous months)
    valid_dates = pd.date_range("1995-01-01", "2005-12-01", freq="MS")
    ds_valid = xr.Dataset(
        {"rainfall": (["time", "lat", "lon"], np.full((132, 2, 2), 50.0))},
        coords={"time": valid_dates, "lat": [0.5, 0.6], "lon": [35.2, 35.3]},
    )
    mean_ds, std_ds, mask = calculate_rainfall_climatology(ds_valid, baseline_years=(1995, 2005))
    assert mean_ds.sizes["month"] == 12

    # 2. Missing month in 1991-2020 (359 months instead of 360)
    full_dates_360 = pd.date_range("1991-01-01", "2020-12-01", freq="MS")
    dropped_dates = full_dates_360.drop(pd.Timestamp("2005-06-01"))
    ds_missing = xr.Dataset(
        {"rainfall": (["time", "lat", "lon"], np.full((len(dropped_dates), 2, 2), 50.0))},
        coords={"time": dropped_dates, "lat": [0.5, 0.6], "lon": [35.2, 35.3]},
    )
    with pytest.raises(DataCompletenessError, match="incomplete or non-continuous"):
        calculate_rainfall_climatology(ds_missing, baseline_years=(1991, 2020))

    # 3. Unordered timestamps in baseline
    shuffled_dates = list(full_dates_360)
    shuffled_dates[5], shuffled_dates[6] = shuffled_dates[6], shuffled_dates[5]
    ds_unordered = xr.Dataset(
        {"rainfall": (["time", "lat", "lon"], np.full((360, 2, 2), 50.0))},
        coords={"time": shuffled_dates, "lat": [0.5, 0.6], "lon": [35.2, 35.3]},
    )
    with pytest.raises(DataCompletenessError, match="incomplete or non-continuous"):
        calculate_rainfall_climatology(ds_unordered, baseline_years=(1991, 2020))

    # 4. Input dataset spans 1981-2025 (broader than 1991-2020): must succeed and extract exactly 1991-2020
    broad_dates = pd.date_range("1981-01-01", "2025-12-01", freq="MS")
    ds_broad = xr.Dataset(
        {"rainfall": (["time", "lat", "lon"], np.full((len(broad_dates), 2, 2), 50.0))},
        coords={"time": broad_dates, "lat": [0.5, 0.6], "lon": [35.2, 35.3]},
    )
    mean_broad, std_broad, mask_broad = calculate_rainfall_climatology(ds_broad, baseline_years=(1991, 2020))
    assert mean_broad.sizes["month"] == 12
    assert mean_broad.attrs["climatology_baseline"] == "1991-2020"

    # 5. Duplicate month in baseline period
    dup_dates = list(full_dates_360)
    dup_dates[10] = dup_dates[9]  # Create duplicate
    ds_dup = xr.Dataset(
        {"rainfall": (["time", "lat", "lon"], np.full((360, 2, 2), 50.0))},
        coords={"time": dup_dates, "lat": [0.5, 0.6], "lon": [35.2, 35.3]},
    )
    with pytest.raises(DataCompletenessError, match="incomplete or non-continuous"):
        calculate_rainfall_climatology(ds_dup, baseline_years=(1991, 2020))


def test_rainfall_zscore_zero_variance_protection() -> None:
    """Verify that invariant baseline pixels (sigma <= std_epsilon) evaluate to NaN and are audited."""
    dates = pd.date_range("1991-01-01", "2020-12-01", freq="MS")
    # Pixel [0, 0]: sigma = 0.0 (constant 50.0)
    # Pixel [0, 1]: 0.0 < sigma <= 1e-4 (near zero variance: constant + tiny 1e-6 perturbation)
    # Pixel [0, 2]: sigma > 1e-4 (genuine variance)
    data = np.zeros((len(dates), 1, 3), dtype=np.float32)
    data[:, 0, 0] = 50.0
    data[:, 0, 1] = 50.0 + (np.arange(len(dates)) % 2) * 1e-6
    data[:, 0, 2] = np.sin(np.arange(len(dates))) * 20.0 + 50.0

    ds_baseline = xr.Dataset(
        data_vars={"rainfall": (["time", "lat", "lon"], data)},
        coords={"time": dates, "lat": [0.5], "lon": [35.2, 35.3, 35.4]},
    )

    clim_mean, clim_std, valid_std_mask = calculate_rainfall_climatology(
        ds_baseline,
        baseline_years=(1991, 2020),
        std_epsilon=1e-4,
    )

    # Pixel [0, 0] and Pixel [0, 1] must be masked as invalid variance for all 12 months
    assert int((~valid_std_mask).sum().values) == 24  # 12 months * 2 invariant pixels

    # Evaluate target Z-score
    target_dates = pd.date_range("2023-04-01", "2023-05-01", freq="MS")
    target_data = np.full((2, 1, 3), 60.0, dtype=np.float32)
    ds_target = xr.Dataset(
        data_vars={"rainfall": (["time", "lat", "lon"], target_data)},
        coords={"time": target_dates, "lat": [0.5], "lon": [35.2, 35.3, 35.4]},
    )

    z_ds = calculate_rainfall_zscore(ds_target, clim_mean, clim_std, std_epsilon=1e-4)
    z_vals = z_ds.z_rainfall.values

    # Invariant and near-invariant pixels must evaluate strictly to NaN
    assert np.isnan(z_vals[:, 0, 0]).all()
    assert np.isnan(z_vals[:, 0, 1]).all()
    # Variable pixel must evaluate to finite continuous float
    assert np.isfinite(z_vals[:, 0, 2]).all()


def test_ndvi_target_grid_contract() -> None:
    """Verify aggregate_target_to_climatology_grid outputs matching CRS, transform, coordinates, and shape."""
    # Source: 10m grid in EPSG:32736
    src_trans = from_origin(740000, 66100, 10, 10)
    x_src = np.arange(740005, 740100, 10)
    y_src = np.arange(66095, 66000, -10)
    data_src = np.full((len(y_src), len(x_src)), 0.7, dtype=np.float32)

    da_s2 = xr.DataArray(data_src, coords={"y": y_src, "x": x_src}, dims=["y", "x"], name="ndvi")
    da_s2.rio.write_crs("EPSG:32736", inplace=True)
    da_s2.rio.write_transform(src_trans, inplace=True)
    da_s2.rio.write_nodata(np.nan, inplace=True)

    # Reference climatology: 30m grid in EPSG:6933
    dst_trans = from_origin(3396000, 79100, 30, 30)
    x_dst = np.array([3396015.0, 3396045.0, 3396075.0])
    y_dst = np.array([79085.0, 79055.0, 79025.0])
    data_dst = np.zeros((3, 3), dtype=np.float32)

    da_ref = xr.DataArray(data_dst, coords={"y": y_dst, "x": x_dst}, dims=["y", "x"], name="mean_apr")
    da_ref.rio.write_crs("EPSG:6933", inplace=True)
    da_ref.rio.write_transform(dst_trans, inplace=True)
    da_ref.rio.write_nodata(np.nan, inplace=True)

    aligned = aggregate_target_to_climatology_grid(da_s2, da_ref)

    assert str(aligned.rio.crs) == "EPSG:6933"
    assert aligned.sizes["y"] == da_ref.sizes["y"]
    assert aligned.sizes["x"] == da_ref.sizes["x"]
    assert (aligned.y.values == da_ref.y.values).all()
    assert (aligned.x.values == da_ref.x.values).all()
    assert aligned.rio.transform() == dst_trans


def test_ndvi_area_weighted_nodata_handling() -> None:
    """Verify equal-area downsampling: valid-pixel average and all-NaN handling.

    Target 30m cell overlapping nine 10m cells:
    [1, 2, 3]
    [4, NaN, 6]
    [7, 8, 9]
    Mean of 8 valid observations: (1+2+3+4+6+7+8+9)/8 = 40/8 = 5.0.
    """
    src_trans = from_origin(0, 30, 10, 10)
    dst_trans = from_origin(0, 30, 30, 30)

    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float32)

    xs = np.array([5.0, 15.0, 25.0])
    ys = np.array([25.0, 15.0, 5.0])
    da_src = xr.DataArray(data, coords={"y": ys, "x": xs}, dims=["y", "x"]).rio.write_crs("EPSG:32736").rio.write_transform(src_trans)
    da_src.rio.write_nodata(np.nan, inplace=True)

    da_dst = xr.DataArray(np.zeros((1, 1), dtype=np.float32), coords={"y": [15.0], "x": [15.0]}, dims=["y", "x"]).rio.write_crs("EPSG:32736").rio.write_transform(dst_trans)
    da_dst.rio.write_nodata(np.nan, inplace=True)

    res = aggregate_target_to_climatology_grid(da_src, da_dst)
    assert np.isclose(float(res.values[0, 0]), 5.0)

    # All-NaN source
    da_src_all_nan = xr.DataArray(np.full((3, 3), np.nan, dtype=np.float32), coords={"y": ys, "x": xs}, dims=["y", "x"]).rio.write_crs("EPSG:32736").rio.write_transform(src_trans)
    da_src_all_nan.rio.write_nodata(np.nan, inplace=True)

    res_nan = aggregate_target_to_climatology_grid(da_src_all_nan, da_dst)
    assert np.isnan(float(res_nan.values[0, 0]))


def test_ndvi_unequal_overlap_area_weighting() -> None:
    """Verify unequal-overlap area weighting:

    Source A (col 0, 1): 50% overlap, value 0.2
    Source B (col 2):    25% overlap, value 0.6
    Source C (col 3):    25% overlap, value 1.0
    Expected (all valid): 0.50(0.2) + 0.25(0.6) + 0.25(1.0) = 0.50.
    With B = NaN: (0.50*0.2 + 0.25*1.0)/(0.50 + 0.25) = 0.35/0.75 = 0.4667.
    """
    src_trans = from_origin(0, 100, 25, 25)
    dst_trans = from_origin(0, 100, 100, 100)

    data = np.zeros((4, 4), dtype=np.float32)
    data[:, 0:2] = 0.2
    data[:, 2] = 0.6
    data[:, 3] = 1.0

    xs = np.array([12.5, 37.5, 62.5, 87.5])
    ys = np.array([87.5, 62.5, 37.5, 12.5])
    da_src = xr.DataArray(data, coords={"y": ys, "x": xs}, dims=["y", "x"]).rio.write_crs("EPSG:32736").rio.write_transform(src_trans)
    da_src.rio.write_nodata(np.nan, inplace=True)

    da_dst = xr.DataArray(np.zeros((1, 1), dtype=np.float32), coords={"y": [50.0], "x": [50.0]}, dims=["y", "x"]).rio.write_crs("EPSG:32736").rio.write_transform(dst_trans)
    da_dst.rio.write_nodata(np.nan, inplace=True)

    # 1. All valid
    out_valid = aggregate_target_to_climatology_grid(da_src, da_dst)
    assert np.isclose(float(out_valid.values[0, 0]), 0.50)

    # 2. B = NaN
    data_nan = data.copy()
    data_nan[:, 2] = np.nan
    da_src_nan = xr.DataArray(data_nan, coords={"y": ys, "x": xs}, dims=["y", "x"]).rio.write_crs("EPSG:32736").rio.write_transform(src_trans)
    da_src_nan.rio.write_nodata(np.nan, inplace=True)

    out_nan = aggregate_target_to_climatology_grid(da_src_nan, da_dst)
    expected_weighted = (0.50 * 0.2 + 0.25 * 1.0) / (0.50 + 0.25)
    assert np.isclose(float(out_nan.values[0, 0]), expected_weighted, atol=1e-4)


def test_ndvi_zscore_min_obs_qa() -> None:
    """Confirm baseline pixels with count < min_valid_obs evaluate to NaN in Z_NDVI."""
    y = [10.0]
    x = [20.0]
    ds_clim = xr.Dataset(
        data_vars={
            "mean_apr": (["y", "x"], [[0.5]]),
            "stddev_apr": (["y", "x"], [[0.1]]),
            "count_apr": (["y", "x"], [[15]]),  # Below min_valid_obs = 20
        },
        coords={"y": y, "x": x},
    )

    filtered = filter_ndvi_climatology_qa(ds_clim, min_valid_obs=20)
    assert np.isnan(filtered.mean_apr.values[0, 0])
    assert np.isnan(filtered.stddev_apr.values[0, 0])

    # Z-score using filtered baseline must evaluate to NaN
    target_da = xr.DataArray([[0.6]], coords={"y": y, "x": x}, dims=["y", "x"], name="ndvi_aligned")
    z = calculate_ndvi_zscore(target_da, filtered.mean_apr, filtered.stddev_apr)
    assert np.isnan(z.values[0, 0])


def test_ndvi_zscore_preserves_reference_grid() -> None:
    """Verify calculate_ndvi_zscore preserves reference climatology coordinates, CRS, and maps months."""
    times = [pd.Timestamp("2023-03-01"), pd.Timestamp("2023-03-15"), pd.Timestamp("2023-04-12")]
    y = [79015.0, 78985.0]
    x = [3396015.0, 3396045.0]

    # Target composite (3 time steps)
    target_data = np.full((3, 2, 2), 0.6, dtype=np.float32)
    aligned_ndvi = xr.DataArray(
        target_data,
        coords={"time": times, "y": y, "x": x},
        dims=["time", "y", "x"],
        name="ndvi_aligned",
    )
    aligned_ndvi.rio.write_crs("EPSG:6933", inplace=True)

    # Climatology with 12 months
    months = np.arange(1, 13)
    mean_grid = np.zeros((12, 2, 2), dtype=np.float32)
    mean_grid[2, :, :] = 0.40  # March (month 3) mean = 0.40
    mean_grid[3, :, :] = 0.50  # April (month 4) mean = 0.50

    std_grid = np.full((12, 2, 2), 0.10, dtype=np.float32)  # std = 0.10 for all months

    clim_mean = xr.DataArray(mean_grid, coords={"month": months, "y": y, "x": x}, dims=["month", "y", "x"])
    clim_std = xr.DataArray(std_grid, coords={"month": months, "y": y, "x": x}, dims=["month", "y", "x"])

    z_ndvi = calculate_ndvi_zscore(aligned_ndvi, clim_mean, clim_std)

    assert z_ndvi.name == "z_ndvi"
    assert z_ndvi.shape == (3, 2, 2)
    # 2023-03-01 (March): (0.6 - 0.4) / 0.1 = 2.0
    assert np.allclose(z_ndvi.isel(time=0).values, 2.0)
    # 2023-03-15 (March): (0.6 - 0.4) / 0.1 = 2.0
    assert np.allclose(z_ndvi.isel(time=1).values, 2.0)
    # 2023-04-12 (April): (0.6 - 0.5) / 0.1 = 1.0
    assert np.allclose(z_ndvi.isel(time=2).values, 1.0)


def test_grafs_half_open_interval_boundary() -> None:
    """Verify that an observation exactly at t_{i+1} belongs ONLY to [t_{i+1}, t_{i+2}) and not [t_i, t_{i+1})."""
    times = pd.date_range("2023-03-01", "2023-03-30", freq="D")
    s1_vals = np.ones((len(times), 1, 1), dtype=np.float32)

    # Set distinct spike on boundary: 2023-03-15
    idx_15 = list(times).index(pd.Timestamp("2023-03-15"))
    s1_vals[idx_15, 0, 0] = 99.0

    ds = xr.Dataset(
        data_vars={"s1": (["time", "lat", "lon"], s1_vals)},
        coords={"time": times, "lat": [0.5], "lon": [35.2]},
    )

    target_time_bins = [pd.Timestamp("2023-03-01"), pd.Timestamp("2023-03-15")]
    res = resample_soil_moisture_to_calendar(ds, target_time_bins)

    # Bin 0 [2023-03-01, 2023-03-15): 14 days of 1.0 -> mean = 1.0
    assert np.isclose(float(res.s1.isel(time=0).values[0, 0]), 1.0)
    # Bin 1 [2023-03-15, 2023-03-29): includes 2023-03-15 (99.0) -> mean = (99 + 13)/14 = 8.0
    assert np.isclose(float(res.s1.isel(time=1).values[0, 0]), 8.0)


def test_grafs_final_bin_boundary() -> None:
    """Verify that an observation exactly at target_time_bins[-1] + 14 days is NOT included in the final bin."""
    times = pd.date_range("2023-03-01", "2023-03-30", freq="D")
    s1_vals = np.ones((len(times), 1, 1), dtype=np.float32)

    # Set spike on final boundary: 2023-03-29 (t_1 + 14d)
    idx_29 = list(times).index(pd.Timestamp("2023-03-29"))
    s1_vals[idx_29, 0, 0] = 99.0

    ds = xr.Dataset(
        data_vars={"s1": (["time", "lat", "lon"], s1_vals)},
        coords={"time": times, "lat": [0.5], "lon": [35.2]},
    )

    target_time_bins = [pd.Timestamp("2023-03-01"), pd.Timestamp("2023-03-15")]
    res = resample_soil_moisture_to_calendar(ds, target_time_bins)

    # Final Bin 1 [2023-03-15, 2023-03-29): 14 days of 1.0 -> mean = 1.0 (99.0 on 2023-03-29 must be excluded)
    assert np.isclose(float(res.s1.isel(time=1).values[0, 0]), 1.0)


def test_swi_14d_change_evaluation() -> None:
    """Verify calculate_swi_14d_change evaluates s1(t) - s1(t-1) and never s0."""
    times = [pd.Timestamp("2023-04-01"), pd.Timestamp("2023-04-15"), pd.Timestamp("2023-04-29")]
    # Deliberately construct distinct s0 and s1 trajectories
    s0_vals = np.array([
        [[0.10]],
        [[0.90]],  # s0 change: 0.90 - 0.10 = +0.80
        [[0.10]],  # s0 change: 0.10 - 0.90 = -0.80
    ])
    s1_vals = np.array([
        [[0.50]],
        [[0.55]],  # s1 change: 0.55 - 0.50 = +0.05
        [[0.45]],  # s1 change: 0.45 - 0.55 = -0.10
    ])

    ds_resampled = xr.Dataset(
        data_vars={"s0": (["time", "lat", "lon"], s0_vals), "s1": (["time", "lat", "lon"], s1_vals)},
        coords={"time": times, "lat": [0.5], "lon": [35.2]},
    )

    ds_change = calculate_swi_14d_change(ds_resampled)
    delta_vals = ds_change.delta_swi_14d.values

    # t=0 must be NaN
    assert np.isnan(delta_vals[0, 0, 0])
    # t=1 must match s1 diff (+0.05) and NOT s0 diff (+0.80)
    assert np.isclose(delta_vals[1, 0, 0], 0.05)
    assert not np.isclose(delta_vals[1, 0, 0], 0.80)
    # t=2 must match s1 diff (-0.10) and NOT s0 diff (-0.80)
    assert np.isclose(delta_vals[2, 0, 0], -0.10)
    assert not np.isclose(delta_vals[2, 0, 0], -0.80)
