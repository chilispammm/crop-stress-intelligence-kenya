"""Live network integration tests for real data access and preprocessing pipelines.

Marked with `@pytest.mark.network` and `@pytest.mark.integration`.
Run with:
    pytest -m network
Or run offline tests only with:
    pytest -m "not network"
"""

import os
import numpy as np
import pandas as pd
import pystac_client
import pytest
import rasterio
import xarray as xr
import odc.stac

from src.features.hydrology import (
    calculate_swi_14d_change,
    resample_soil_moisture_to_calendar,
)
from src.features.indices import (
    calculate_evi,
    calculate_ndmi,
    calculate_ndvi,
    validate_index_distribution,
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
from src.ingestion.chirps import load_chirps_data
from src.ingestion.grafs import load_grafs_data
from src.ingestion.mask import load_crop_mask
from src.ingestion.sentinel2 import load_sentinel2_data
from src.preprocessing.alignment import MultiResolutionCube
from src.preprocessing.compositing import composite_14d, smooth_temporal_series
from src.preprocessing.diagnostics import generate_quality_report
from src.preprocessing.masking import apply_cropland_mask
from src.preprocessing.optical import apply_scl_mask
from src.utils.aoi import PILOT_AOI_BBOX
from src.diagnostics.screening import (
    DiagnosticCase,
    ACTIONABLE_CASES,
    assign_coarse_to_diagnostic_grid,
    calculate_evidence_confidence,
    classify_diagnostic_grid,
)
from src.export.vector import extract_scouting_zones


@pytest.mark.network
@pytest.mark.integration
def test_live_sentinel2_pixel_retrieval() -> None:
    """Verify live Sentinel-2 STAC query and real pixel retrieval with native integer dtypes."""
    small_bbox = (35.20, 0.60, 35.22, 0.62)
    start_date = "2024-04-01"
    end_date = "2024-04-10"

    ds_s2 = load_sentinel2_data(
        bbox=small_bbox,
        start_date=start_date,
        end_date=end_date,
        bands=["B04", "B08", "SCL"],
        resolution=10,
        crs="EPSG:32736",
    )

    # Validate structure & dimensions
    assert isinstance(ds_s2, xr.Dataset)
    assert "B04" in ds_s2 and "B08" in ds_s2 and "SCL" in ds_s2
    assert "y" in ds_s2.coords and "x" in ds_s2.coords
    assert ds_s2.sizes["y"] > 0 and ds_s2.sizes["x"] > 0
    assert "time" in ds_s2.coords and ds_s2.sizes["time"] >= 1

    # Validate memory integrity (native integer representations preserved)
    assert str(ds_s2.B04.dtype) == "uint16"
    assert str(ds_s2.B08.dtype) == "uint16"
    assert str(ds_s2.SCL.dtype) == "uint8"

    # Validate non-empty actual pixel values
    b04_vals = ds_s2.B04.values
    b08_vals = ds_s2.B08.values
    assert b04_vals.max() > 0, "B04 reflectance array cannot be all zero"
    assert b08_vals.max() > 0, "B08 reflectance array cannot be all zero"


@pytest.mark.network
@pytest.mark.integration
def test_live_chirps_monthly_retrieval() -> None:
    """Verify live CHIRPS Monthly STAC query and real precipitation grid retrieval."""
    small_bbox = (35.15, 0.55, 35.35, 0.75)
    start_date = "2023-01-01"
    end_date = "2023-03-01"

    ds_chirps = load_chirps_data(
        bbox=small_bbox,
        start_date=start_date,
        end_date=end_date,
        frequency="monthly",
    )

    assert isinstance(ds_chirps, xr.Dataset)
    assert "rainfall" in ds_chirps
    assert ds_chirps["rainfall"].attrs["units"] == "mm/month"
    assert ds_chirps.sizes["lat"] >= 2
    assert ds_chirps.sizes["lon"] >= 2
    assert ds_chirps.sizes["time"] >= 1

    vals = ds_chirps.rainfall.values
    assert float(vals.max()) > 0.0, "CHIRPS monthly rainfall should contain positive values"


@pytest.mark.network
@pytest.mark.integration
def test_live_grafs_opendap_retrieval() -> None:
    """Verify live GRAFS OPeNDAP connection and real soil moisture grid retrieval."""
    small_bbox = (35.15, 0.55, 35.35, 0.75)
    start_date = "2022-04-01"
    end_date = "2022-04-03"

    ds_grafs = load_grafs_data(
        bbox=small_bbox,
        start_date=start_date,
        end_date=end_date,
        variables=["s0"],
    )

    assert isinstance(ds_grafs, xr.Dataset)
    assert "s0" in ds_grafs
    assert ds_grafs["s0"].attrs["field_measurement"] is False
    assert ds_grafs.sizes["lat"] >= 2
    assert ds_grafs.sizes["lon"] >= 2
    assert ds_grafs.sizes["time"] >= 1

    vals = ds_grafs.s0.values
    assert 0.0 <= float(vals.min()) <= 1.0
    assert 0.0 <= float(vals.max()) <= 1.0


@pytest.mark.network
@pytest.mark.integration
def test_live_cropland_mask_retrieval() -> None:
    """Verify live Digital Earth Africa Cropland Extent STAC query and real mask retrieval at native 20 m."""
    small_bbox = (35.20, 0.60, 35.22, 0.62)
    da_mask = load_crop_mask(
        bbox=small_bbox,
        resolution=20,
        crs="EPSG:32736",
    )

    assert isinstance(da_mask, xr.DataArray)
    assert da_mask.name == "crop_mask"
    assert "maize_qualification" in da_mask.attrs
    assert da_mask.sizes["y"] > 0 and da_mask.sizes["x"] > 0

    unique_vals = set(int(v) for v in da_mask.values.flatten() if v == v)
    assert unique_vals.issubset({0, 1})


@pytest.mark.network
@pytest.mark.integration
def test_live_multi_resolution_cube_integration() -> None:
    """Verify assembling all 4 real live datasets into MultiResolutionCube without raster flattening."""
    pilot_bbox = (35.15, 0.55, 35.35, 0.75)
    small_optical_bbox = (35.20, 0.60, 35.22, 0.62)

    # 1. Optical (10 m, EPSG:32736)
    ds_s2 = load_sentinel2_data(
        bbox=small_optical_bbox,
        start_date="2024-04-01",
        end_date="2024-04-07",
        bands=["B04", "B08", "SCL"],
        resolution=10,
        crs="EPSG:32736",
    )

    # 2. Climate (~5.5 km, EPSG:4326)
    ds_chirps = load_chirps_data(
        bbox=pilot_bbox,
        start_date="2023-01-01",
        end_date="2023-02-01",
        frequency="monthly",
    )

    # 3. Hydrology (~10 km, EPSG:4326)
    ds_grafs = load_grafs_data(
        bbox=pilot_bbox,
        start_date="2022-04-01",
        end_date="2022-04-02",
        variables=["s0"],
    )

    # 4. Mask (20 m native, EPSG:32736)
    da_mask = load_crop_mask(
        bbox=small_optical_bbox,
        resolution=20,
        crs="EPSG:32736",
    )

    # Instantiate MultiResolutionCube
    cube = MultiResolutionCube(
        optical=ds_s2,
        climate=ds_chirps,
        hydrology=ds_grafs,
        mask=da_mask,
        aoi_bbox=pilot_bbox,
    )

    status = cube.validate_modalities(require_provenance=True)
    assert status["optical"] is True
    assert status["climate"] is True
    assert status["hydrology"] is True
    assert status["mask"] is True

    summary = cube.summary()
    assert summary["modalities"]["optical"]["resolution"] == "10-20 m"
    assert summary["modalities"]["climate"]["resolution"] == "0.05 degree (~5.5 km)"
    assert summary["modalities"]["hydrology"]["resolution"] == "0.1 degree (~10 km)"
    assert summary["modalities"]["mask"]["resolution"] == "20 m"


@pytest.mark.network
@pytest.mark.integration
def test_live_sentinel2_preprocessing_pipeline() -> None:
    """Verify live Sentinel-2 optical masking, NDVI, EVI, NDMI, 14-day compositing, and cropland masking."""
    small_bbox = (35.20, 0.60, 35.22, 0.62)
    start_date = "2023-04-01"
    end_date = "2023-04-20"

    # Ingest 10m and 20m bands for the pilot window
    try:
        # Load 10m bands (B02, B04, B08, SCL)
        ds_s2_10m = load_sentinel2_data(
            bbox=small_bbox,
            start_date=start_date,
            end_date=end_date,
            bands=["B02", "B04", "B08", "SCL"],
            resolution=10,
            crs="EPSG:32736",
        )
        # Load 20m SWIR band (B11)
        ds_s2_20m = load_sentinel2_data(
            bbox=small_bbox,
            start_date=start_date,
            end_date=end_date,
            bands=["B11"],
            resolution=20,
            crs="EPSG:32736",
        )
    except ConnectionError as exc:
        pytest.skip(f"Live STAC endpoint unavailable during test: {exc}")

    # 1. Apply SCL mask
    ds_masked_10m = apply_scl_mask(ds_s2_10m, valid_classes=[4, 5, 6])
    assert ds_masked_10m.B04.attrs.get("scl_masked") is True

    # 2. Resolution-aware biophysical feature engineering
    ndvi = calculate_ndvi(ds_masked_10m.B04, ds_masked_10m.B08)
    evi = calculate_evi(ds_masked_10m.B02, ds_masked_10m.B04, ds_masked_10m.B08)
    ndmi = calculate_ndmi(ds_masked_10m.B08, ds_s2_20m.B11)

    assert ndvi.name == "ndvi" and ndvi.sizes["y"] == ds_masked_10m.sizes["y"]
    assert evi.name == "evi" and evi.sizes["y"] == ds_masked_10m.sizes["y"]
    assert ndmi.name == "ndmi" and ndmi.sizes["y"] == ds_s2_20m.sizes["y"]

    # 3. 14-day temporal compositing and observation accounting
    ndvi_comp, valid_obs = composite_14d(ndvi, freq="14D")
    assert ndvi_comp.sizes["time"] >= 1
    assert valid_obs.sizes["time"] >= 1
    assert valid_obs.name == "valid_obs_count"

    # 4. Rolling temporal smoothing with missingness preservation
    ndvi_smoothed = smooth_temporal_series(ndvi_comp, window=3, min_periods=1)
    assert ndvi_smoothed.attrs["missingness_preservation"] == "strict_nan_retention"

    # 5. Cropland filtering with live mask (native 20 m aligned to 10 m NDVI)
    crop_mask = load_crop_mask(bbox=small_bbox, resolution=20, crs="EPSG:32736")
    ndvi_crop = apply_cropland_mask(ndvi_smoothed, crop_mask, cultivated_value=1)
    assert ndvi_crop.attrs["cropland_masked"] is True

    # 6. Quality report diagnostics
    quality_report = generate_quality_report(
        raw_ds=ds_s2_10m,
        masked_ds=ds_masked_10m,
        composite_da=ndvi_comp,
        valid_obs_count=valid_obs,
    )
    assert "rejection_rate_scl" in quality_report
    assert "mean_valid_obs_per_bin" in quality_report

    # 7. Distribution audit
    dist_report = validate_index_distribution(ndvi_crop, index_name="NDVI")
    assert dist_report["total_pixels"] > 0
    assert dist_report["index_name"] == "NDVI"


@pytest.mark.network
@pytest.mark.integration
def test_live_m4_anomaly_pipeline() -> None:
    """Verify live Multi-Modal Baseline & Standardized Anomaly Pipeline (Milestone 4).

    1. Ingests monthly CHIRPS for Moiben AOI and computes continuous Z_R.
    2. Ingests ndvi_climatology_ls, aggregates real 2023 Sentinel-2 composite onto
       authoritative 30m grid (EPSG:6933), and computes Z_NDVI.
    3. Ingests GRAFS daily soil moisture, resamples to optical target_time_bins, and
       evaluates Delta SWI_14d.
    4. Performs bounded assertions on spatial slices verifying dimensions, CRSs, and valid values.
    """
    pilot_bbox = (35.15, 0.55, 35.35, 0.75)
    small_bbox = (35.20, 0.60, 35.22, 0.62)

    # 1. Live CHIRPS Ingestion & Standardized Rainfall Anomaly (Z_R)
    ds_chirps_2023 = load_chirps_data(
        bbox=pilot_bbox,
        start_date="2023-04-01",
        end_date="2023-05-01",
        frequency="monthly",
    )
    assert "rainfall" in ds_chirps_2023

    # Generate 360-month baseline using realistic spatial dimensions from CHIRPS
    baseline_dates = pd.date_range("1991-01-01", "2020-12-01", freq="MS")
    np.random.seed(42)
    fake_precip = np.random.uniform(20.0, 150.0, size=(len(baseline_dates), ds_chirps_2023.sizes["lat"], ds_chirps_2023.sizes["lon"]))
    ds_chirps_baseline = xr.Dataset(
        {"rainfall": (["time", "lat", "lon"], fake_precip)},
        coords={"time": baseline_dates, "lat": ds_chirps_2023.lat, "lon": ds_chirps_2023.lon},
    )
    clim_mean, clim_std, valid_std_mask = calculate_rainfall_climatology(ds_chirps_baseline, baseline_years=(1991, 2020))
    z_rainfall = calculate_rainfall_zscore(ds_chirps_2023, clim_mean, clim_std, std_epsilon=1e-4)

    assert "z_rainfall" in z_rainfall
    assert z_rainfall.z_rainfall.dims == ds_chirps_2023.rainfall.dims
    assert np.isfinite(z_rainfall.z_rainfall.values).sum() > 0

    # 2. Live ndvi_climatology_ls, Sentinel-2 Ingestion & Standardized Vegetation Anomaly (Z_NDVI)
    try:
        client = pystac_client.Client.open("https://explorer.digitalearth.africa/stac")
        search = client.search(collections=["ndvi_climatology_ls"], bbox=small_bbox, limit=1)
        items = list(search.items())
        assert len(items) > 0, "No ndvi_climatology_ls items found"

        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", AWS_DEFAULT_REGION="af-south-1", AWS_REGION="af-south-1"):
            ds_clim = odc.stac.load(items, bbox=small_bbox, bands=["mean_apr", "stddev_apr", "count_apr"])
    except ConnectionError as exc:
        pytest.skip(f"Live STAC endpoint unavailable: {exc}")

    ds_clim_filtered = filter_ndvi_climatology_qa(ds_clim, min_valid_obs=20)
    ref_mean_apr = ds_clim_filtered.mean_apr.squeeze("time", drop=True) if "time" in ds_clim_filtered.mean_apr.dims else ds_clim_filtered.mean_apr
    ref_std_apr = ds_clim_filtered.stddev_apr.squeeze("time", drop=True) if "time" in ds_clim_filtered.stddev_apr.dims else ds_clim_filtered.stddev_apr

    ds_s2 = load_sentinel2_data(
        bbox=small_bbox,
        start_date="2023-04-01",
        end_date="2023-04-20",
        bands=["B04", "B08", "SCL"],
        resolution=10,
        crs="EPSG:32736",
    )
    ds_s2_masked = apply_scl_mask(ds_s2)
    ndvi_10m = calculate_ndvi(ds_s2_masked.B04, ds_s2_masked.B08)
    ndvi_comp, valid_obs = composite_14d(ndvi_10m)

    # Reproject and aggregate 10m composite onto authoritative 30m EPSG:6933 grid
    ndvi_30m_aligned = aggregate_target_to_climatology_grid(ndvi_comp.isel(time=0), ref_mean_apr)
    assert str(ndvi_30m_aligned.rio.crs) == "EPSG:6933"
    assert ndvi_30m_aligned.sizes["y"] == ref_mean_apr.sizes["y"]
    assert ndvi_30m_aligned.sizes["x"] == ref_mean_apr.sizes["x"]

    # Compute continuous Z_NDVI
    z_ndvi = calculate_ndvi_zscore(ndvi_30m_aligned, ref_mean_apr, ref_std_apr, std_epsilon=1e-4)
    assert z_ndvi.name == "z_ndvi"
    assert np.isfinite(z_ndvi.values).sum() > 0

    # 3. Live GRAFS Soil Moisture & Delta SWI_14d
    ds_grafs = load_grafs_data(
        bbox=pilot_bbox,
        start_date="2022-04-01",
        end_date="2022-04-30",
        variables=["s0", "s1"],
    )
    assert "s0" in ds_grafs and "s1" in ds_grafs

    target_time_bins = [pd.Timestamp("2022-04-01"), pd.Timestamp("2022-04-15")]
    ds_grafs_14d = resample_soil_moisture_to_calendar(ds_grafs, target_time_bins)
    assert len(ds_grafs_14d.time) == 2
    assert list(pd.to_datetime(ds_grafs_14d.time.values)) == target_time_bins

    ds_swi_change = calculate_swi_14d_change(ds_grafs_14d)
    assert "delta_swi_14d" in ds_swi_change
    delta_vals = ds_swi_change.delta_swi_14d.values

    # t=0 must be NaN, t=1 must contain finite continuous differences
    assert np.isnan(delta_vals[0]).all()
    assert np.isfinite(delta_vals[1]).sum() > 0


@pytest.mark.network
@pytest.mark.integration
def test_live_m5_screening_pipeline() -> None:
    """Verify live Multi-Modal Crop Stress Screening Pipeline (Milestone 5).

    Uses same short pilot windows as existing M4 live test to minimise network load.
    Follows the resolution-honest M5 contract:
      - Z_R (CHIRPS ~5.5 km) assigned to 30 m grid via nearest-neighbor lookup only.
      - delta_swi_14d (GRAFS ~10 km) assigned to 30 m grid via nearest-neighbor lookup only.
      - Z_NDVI lives natively on the 30 m EPSG:6933 grid.
      - INSUFFICIENT_EVIDENCE, NORMAL, and actionable cases reported explicitly.
      - Zero scouting zones is a valid result — NOT a pipeline failure.

    Scientific caveat: Diagnostic rules are initial screening assumptions requiring
    future field validation. Classifications are NOT causal crop-stress diagnoses.
    """
    pilot_bbox = (35.15, 0.55, 35.35, 0.75)
    small_bbox = (35.20, 0.60, 35.22, 0.62)

    # 1. Live CHIRPS Ingestion & Z_R (same 2023 April window as M4 live test)
    ds_chirps_2023 = load_chirps_data(
        bbox=pilot_bbox,
        start_date="2023-04-01",
        end_date="2023-05-01",
        frequency="monthly",
    )
    assert "rainfall" in ds_chirps_2023

    baseline_dates = pd.date_range("1991-01-01", "2020-12-01", freq="MS")
    np.random.seed(42)
    fake_precip = np.random.uniform(
        20.0, 150.0,
        size=(len(baseline_dates), ds_chirps_2023.sizes["lat"], ds_chirps_2023.sizes["lon"]),
    )
    ds_chirps_baseline = xr.Dataset(
        {"rainfall": (["time", "lat", "lon"], fake_precip)},
        coords={"time": baseline_dates, "lat": ds_chirps_2023.lat, "lon": ds_chirps_2023.lon},
    )
    clim_mean, clim_std, _ = calculate_rainfall_climatology(ds_chirps_baseline, baseline_years=(1991, 2020))
    z_rainfall_coarse = calculate_rainfall_zscore(ds_chirps_2023, clim_mean, clim_std, std_epsilon=1e-4)
    assert "z_rainfall" in z_rainfall_coarse

    # 2. Live DE Africa ndvi_climatology_ls + Sentinel-2 → Z_NDVI on 30 m EPSG:6933
    try:
        client = pystac_client.Client.open("https://explorer.digitalearth.africa/stac")
        search = client.search(collections=["ndvi_climatology_ls"], bbox=small_bbox, limit=1)
        items = list(search.items())
        assert len(items) > 0, "No ndvi_climatology_ls items found"

        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", AWS_DEFAULT_REGION="af-south-1", AWS_REGION="af-south-1"):
            ds_clim = odc.stac.load(items, bbox=small_bbox, bands=["mean_apr", "stddev_apr", "count_apr"])
    except ConnectionError as exc:
        pytest.skip(f"Live STAC endpoint unavailable: {exc}")

    ds_clim_filtered = filter_ndvi_climatology_qa(ds_clim, min_valid_obs=20)
    ref_mean_apr = (
        ds_clim_filtered.mean_apr.squeeze("time", drop=True)
        if "time" in ds_clim_filtered.mean_apr.dims
        else ds_clim_filtered.mean_apr
    )
    ref_std_apr = (
        ds_clim_filtered.stddev_apr.squeeze("time", drop=True)
        if "time" in ds_clim_filtered.stddev_apr.dims
        else ds_clim_filtered.stddev_apr
    )

    ds_s2 = load_sentinel2_data(
        bbox=small_bbox,
        start_date="2023-04-01",
        end_date="2023-04-20",
        bands=["B04", "B08", "SCL"],
        resolution=10,
        crs="EPSG:32736",
    )
    ds_s2_masked = apply_scl_mask(ds_s2)
    ndvi_10m = calculate_ndvi(ds_s2_masked.B04, ds_s2_masked.B08)
    ndvi_comp, _ = composite_14d(ndvi_10m)

    ndvi_30m_aligned = aggregate_target_to_climatology_grid(ndvi_comp.isel(time=0), ref_mean_apr)
    assert str(ndvi_30m_aligned.rio.crs) == "EPSG:6933"

    z_ndvi_30m = calculate_ndvi_zscore(ndvi_30m_aligned, ref_mean_apr, ref_std_apr, std_epsilon=1e-4)
    z_ndvi_30m.name = "z_ndvi"

    # 3. Live GRAFS → delta_swi_14d (same 2-bin 2022 subset as M4 live test)
    ds_grafs = load_grafs_data(
        bbox=pilot_bbox,
        start_date="2022-04-01",
        end_date="2022-04-30",
        variables=["s0", "s1"],
    )
    target_time_bins = [pd.Timestamp("2022-04-01"), pd.Timestamp("2022-04-15")]
    ds_grafs_14d = resample_soil_moisture_to_calendar(ds_grafs, target_time_bins)
    ds_swi_change = calculate_swi_14d_change(ds_grafs_14d)
    assert "delta_swi_14d" in ds_swi_change

    # Use t=1 (non-NaN bin) for diagnosis
    delta_swi_t1 = ds_swi_change.delta_swi_14d.isel(time=1)
    delta_swi_t1.name = "delta_swi_14d"

    # Also extract absolute SWI (s1) at t=1 for Cases B/C
    swi_t1 = ds_grafs_14d["s1"].isel(time=1)
    swi_t1.name = "s1"

    # Extract a single-month CHIRPS Z_R (only 1 time step for April 2023)
    z_r_single = z_rainfall_coarse["z_rainfall"].isel(time=0)
    z_r_single.name = "z_rainfall"

    # 4. Assign coarse CHIRPS and GRAFS to 30 m diagnostic grid via nearest-neighbor
    # Resolution contract: Z_R (~5.5 km) and delta_swi (~10 km) are NOT interpolated.
    z_r_30m = assign_coarse_to_diagnostic_grid(z_r_single, z_ndvi_30m)
    delta_swi_30m = assign_coarse_to_diagnostic_grid(delta_swi_t1, z_ndvi_30m)
    swi_30m = assign_coarse_to_diagnostic_grid(swi_t1, z_ndvi_30m)

    # Verify nearest-neighbor attribution (no bilinear intermediate values)
    # Normalize to Python float to avoid float32 vs float64 set membership mismatch
    z_r_source_arr = z_r_single.values.flatten().astype(np.float64)
    z_r_source_vals = set(round(float(v), 3) for v in z_r_source_arr if np.isfinite(v))
    z_r_30m_arr = z_r_30m.values.flatten().astype(np.float64)
    z_r_30m_vals = set(round(float(v), 3) for v in z_r_30m_arr if np.isfinite(v))
    # All 30 m values must appear in source coarse cell values (nearest-neighbor lookup)
    assert z_r_30m_vals.issubset(z_r_source_vals), (
        f"Z_R 30 m grid ({z_r_30m_vals}) contains values not present in coarse "
        f"source cells ({z_r_source_vals}) — bilinear interpolation may have been applied."
    )


    # 5. M5 Diagnostic Classification
    diagnostic_da = classify_diagnostic_grid(
        z_ndvi=z_ndvi_30m,
        z_r=z_r_30m,
        delta_swi_14d=delta_swi_30m,
        swi=swi_30m,
        swi_enabled=True,
    )

    assert diagnostic_da.dtype in (np.int8, np.int16, np.int32, np.int64)

    # Count diagnostic states (do not assert specific values — actual data determines these)
    diag_vals = diagnostic_da.values.flatten()
    total_pixels = len(diag_vals)
    insuff_count = int(np.sum(diag_vals == DiagnosticCase.INSUFFICIENT_EVIDENCE.value))
    normal_count = int(np.sum(diag_vals == DiagnosticCase.NORMAL.value))
    case_a_count = int(np.sum(diag_vals == DiagnosticCase.CASE_A.value))
    case_b_count = int(np.sum(diag_vals == DiagnosticCase.CASE_B.value))
    case_c_count = int(np.sum(diag_vals == DiagnosticCase.CASE_C.value))
    case_d_count = int(np.sum(diag_vals == DiagnosticCase.CASE_D.value))
    multi_count = int(np.sum(diag_vals == DiagnosticCase.MULTI_SIGNAL.value))

    print(f"\n=== M5 DIAGNOSTIC DISTRIBUTION ===")
    print(f"Total pixels:           {total_pixels}")
    print(f"INSUFFICIENT_EVIDENCE:  {insuff_count} ({100*insuff_count/total_pixels:.1f}%)")
    print(f"NORMAL:                 {normal_count} ({100*normal_count/total_pixels:.1f}%)")
    print(f"CASE_A:                 {case_a_count}")
    print(f"CASE_B:                 {case_b_count}")
    print(f"CASE_C:                 {case_c_count}")
    print(f"CASE_D:                 {case_d_count}")
    print(f"MULTI_SIGNAL:           {multi_count}")

    # 6. Evidence confidence
    confidence_da = calculate_evidence_confidence([z_ndvi_30m, z_r_30m, delta_swi_30m, swi_30m])
    assert confidence_da.dtype == np.float32
    assert float(confidence_da.values.min()) >= 0.0
    assert float(confidence_da.values.max()) <= 1.0

    # 7. Scouting zone extraction (empty result is valid)
    result_fc = extract_scouting_zones(
        diagnostic_da=diagnostic_da,
        z_ndvi_da=z_ndvi_30m,
        z_r_da=z_r_30m,
        delta_swi_da=delta_swi_30m,
        confidence_da=confidence_da,
        bin_start=pd.Timestamp("2023-04-01"),
        apply_morphology=True,
    )

    # 8. Validate GeoJSON schema & pipeline report
    assert result_fc["type"] == "FeatureCollection"
    assert "features" in result_fc
    assert isinstance(result_fc["features"], list)
    assert "pipeline_report" in result_fc
    report = result_fc["pipeline_report"]
    assert report["pixel_area_m2"] == 900.0
    assert report["area_calculation_crs"] == "EPSG:6933"
    assert report["n_surviving_clusters"] == len(result_fc["features"])

    n_zones = len(result_fc["features"])
    total_area_ha = sum(f["properties"]["area_ha"] for f in result_fc["features"])

    print(f"\n=== M5 SCOUTING ZONES & PIPELINE REPORT ===")
    print(f"Raw actionable pixels:       {report['raw_actionable_pixels']} ({report['raw_actionable_area_ha']} ha)")
    print(f"Post-morphology pixels:     {report['post_morphology_pixels']} ({report['post_morphology_area_ha']} ha)")
    print(f"Connected components:       {report['n_connected_components']}")
    print(f"Components removed by MMU:  {report['n_components_removed_mmu']}")
    print(f"Surviving scouting clusters: {n_zones}")
    print(f"Final total area (ha):      {total_area_ha:.2f}")

    # 9. Validate geometry properties for each surviving zone
    for feature in result_fc["features"]:
        geom = feature["geometry"]
        props = feature["properties"]
        assert geom["type"] in ("Polygon", "MultiPolygon"), f"Unexpected geometry type: {geom['type']}"
        assert props["area_ha"] >= 2.0, f"Zone area {props['area_ha']} below 2 ha MMU"
        assert props["zone_id"].startswith("SC-20230401-")
        assert -90.0 <= (props["centroid_lat"] or 0) <= 90.0
        assert -180.0 <= (props["centroid_lon"] or 0) <= 180.0
        # Independent area verification: geometry area vs pixel-count area
        if props.get("area_ha_match") is not None:
            assert props["area_ha_match"] is True, f"Geometry area mismatch for {props['zone_id']}"

    # 10. Verify Shapely geometry validity for surviving zones
    try:
        from shapely.geometry import shape as shapely_shape
        for feature in result_fc["features"]:
            shp = shapely_shape(feature["geometry"])
            assert shp.is_valid, f"Invalid geometry: {feature['properties']['zone_id']}"
    except ImportError:
        pass  # shapely optional for this check

    # Note: Zero scouting zones is scientifically valid.
    # The pipeline's correctness is determined by schema/contract compliance, not zone count.
    print(f"\nM5 live screening pipeline completed successfully.")
    print(f"Zero zones is a valid result if no actionable evidence above MMU threshold exists.")
