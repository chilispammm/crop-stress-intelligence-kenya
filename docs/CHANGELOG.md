# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0] - 2026-08-25 (Milestone 4: Multi-Modal Baseline & Standardized Anomaly Engine)

### Added
- **Shared Exceptions (`src.utils.exceptions`)**: Created `DataCompletenessError` for validating temporal continuity across long-term climatological baselines.
- **Precipitation Climatology & Standardized Anomaly Engine (`src.features.rainfall_anomalies`)**:
  - `calculate_rainfall_climatology()`: Enforces strict 360-month continuity over the 1991–2020 baseline, groups by calendar month, computes mean ($\mu_{R, m}$) and standard deviation ($\sigma_{R, m}$), and returns `valid_std_mask` with zero-variance protection.
  - `calculate_rainfall_zscore()`: Evaluates continuous standardized precipitation anomalies ($Z_R$) on native 0.05° EPSG:4326 grids, strictly masking invariant baseline pixels ($\sigma \le \text{std\_epsilon}$) to NaN.
- **Resolution-Honest Vegetation Anomaly Engine (`src.features.vegetation_anomalies`)**:
  - `filter_ndvi_climatology_qa()`: Masks baseline pixels where observation count $< \text{min\_valid\_obs}$ (default: 20) to NaN.
  - `aggregate_target_to_climatology_grid()`: Reprojects and aggregates 10 m Sentinel-2 NDVI composites onto the authoritative reference climatology grid (`EPSG:6933`, 30 m) using area-preserving average downsampling with exact nodata masking.
  - `calculate_ndvi_zscore()`: Evaluates continuous standardized NDVI anomalies ($Z_{\text{NDVI}}$) on the authoritative EPSG:6933 reference grid.
- **Soil Hydrology Dynamics Engine (`src.features.hydrology`)**:
  - `resample_soil_moisture_to_calendar()`: Resamples daily GRAFS topsoil ($s_0$) and root-zone ($s_1$) soil water index to match the optical pipeline's canonical 14-day time bins ($[t, t + 14\text{ days})$).
  - `calculate_swi_14d_change()`: Evaluates signed 14-day root-zone moisture change ($\Delta\text{SWI}_{14d}(t) = s_1(t) - s_1(t-1)$), assigning NaN at $t=0$.
- **Test Suite**:
  - 44 offline unit tests passing in `< 3.5s` (`tests/test_anomalies.py`, `tests/test_foundation.py`).
  - Live network integration test `test_live_m4_anomaly_pipeline` verifying real remote execution across CHIRPS, `ndvi_climatology_ls`, and GRAFS.
- **Architecture Decisions**: Added ADR-015 (Cross-CRS Climatology Grid Contract & Area-Preserving Aggregation) and ADR-016 (Zero-Variance Protection, Strict Baseline Continuity, and Calendar-Driven Hydrological Dynamics).

---

## [0.3.0] - 2026-08-21 (Milestone 3: Optical Preprocessing & Biophysical Feature Engineering)

### Added
- **SCL Optical Cloud & Quality Masking (`src.preprocessing.optical`)**: Implemented `apply_scl_mask()` retaining valid classes `[4, 5, 6]` (Vegetation, Bare Soil, Water) and masking `[0, 1, 2, 3, 7, 8, 9, 10, 11]` to NaN across all reflectance bands.
- **Resolution-Aware Biophysical Indices (`src.features.indices`)**:
  - `calculate_ndvi()`: 10 m native resolution from B04 (Red) and B08 (NIR).
  - `calculate_evi()`: Standard 3-band Enhanced Vegetation Index at 10 m native resolution ($2.5 \cdot (\text{NIR} - \text{Red}) / (\text{NIR} + 6\text{Red} - 7.5\text{Blue} + 1.0)$).
  - `calculate_ndmi()`: 20 m derived Normalized Difference Moisture Index with grid alignment verification and area-weighted 10 m $\to$ 20 m downsampling.
  - `GridAlignmentError`: Custom exception safeguarding against CRS mismatch, disjoint extents, inverted orientations, or improper resolution ratios.
  - `validate_index_distribution()`: Distribution auditing and statistical outlier checking without data mutation.
- **Cropland Prior Masking (`src.preprocessing.masking`)**: Implemented `apply_cropland_mask()` supporting configurable `cultivated_value` (default: 1), preserving the native 20 m resolution of the canonical `crop_mask_eastern` product.
- **14-Day Compositing & Observation Accounting (`src.preprocessing.compositing`)**:
  - `composite_14d()`: 14-day median aggregation.
  - `valid_obs_count`: Exact cell-level count of non-null contributing observations.
  - `smooth_temporal_series()`: Centered rolling mean with strict missingness preservation (`where(composite.notnull(), np.nan)`).
- **Quality Diagnostics Matrix (`src.preprocessing.diagnostics`)**: Implemented `generate_quality_report()` computing acquisition timestamps, scene clear pixel percentages, SCL rejection rate, and unfilled composite bin frequencies.
- **Testing Suite**:
  - 35 offline unit tests passing in `< 3s` (`tests/test_preprocessing.py`, `tests/test_indices.py`).
  - Live network integration test `test_live_sentinel2_preprocessing_pipeline` verifying live end-to-end optical preprocessing over the Moiben Pilot AOI.

---

## [0.2.1] - 2026-08-20 (M2.X: Ingestion Hardening & Live Data Verification)

### Added & Verified
- **DE Africa Sentinel-2 Live Verification**: Live STAC discovery and real pixel retrieval verified against `https://explorer.digitalearth.africa/stac` (`s2_l2a`) with AWS `af-south-1` unsigned S3 configuration.
- **Memory Integrity**: Native integer preservation (`uint16` reflectance, `uint8` SCL) confirmed during live ingestion, protecting memory footprint.
- **Dual-Frequency CHIRPS Verification**: Verified both `rainfall_chirps_monthly` and `rainfall_chirps_daily` on DE Africa STAC at native 0.05° scale with explicit `frequency` parameter.
- **GRAFS Live Verification & Hardening**: Live OPeNDAP retrieval verified against NCI THREDDS server with bounded retry resilience, robust time fill-value decoding, and native 0.1° resolution preservation.
- **Cropland Mask Disambiguation**: Canonical product established as Digital Earth Africa Cropland Extent Map 20 m (`crop_mask_eastern`, band `mask`) with ESA WorldCover 10 m designated as separate fallback.
- **Multi-Modal Live Integration Test**: Verified end-to-end assembly of all four real datasets into `MultiResolutionCube` (`pytest -m network`) without raster flattening.

---

## [0.2.0] - 2026-08-20 (M2.5–M2.10: Ingestion, Multi-Resolution Alignment, and Provenance)

### Added
- **M2.5 — GRAFS Soil Moisture Ingestion**: Reusable loader for `s0` (0–5 cm) and `s1` (0–1 m).
- **M2.6 — Agricultural Mask Ingestion**: Binary 20 m cropland mask loader with explicit qualification tags.
- **M2.7 — Multi-Resolution Alignment**: Decoupled `MultiResolutionCube` container and `ModalitySchema`.
- **M2.8 — Testing & Validation**: Offline geometry, containment, range, and provenance tests.
- **M2.9 — Data Provenance**: Mandatory `provenance_*` metadata creation and validation.
- **M2.10 — Documentation Suite**: Updated data dictionary, decisions log, assumptions, and project specification.

---

## [0.1.0] - 2026-08-19 (M1.0: Engineering Foundation)

### Added
- Repository architecture, packaging, baseline configuration loader, offline test framework, and documentation.
