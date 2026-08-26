# Architecture & Scientific Decisions Log (ADR)

This document records key architectural and scientific design decisions, rationale, and consequences.

---

## ADR-001: Standard Local Optical Projection (EPSG:32736)

- **Status**: Approved (M0 / M1.0)
- **Decision**: Standardize all local optical raster and vector processing on `EPSG:32736` (WGS 84 / UTM Zone 36S).
- **Consequences**: Eliminates distortion associated with unprojected geographic coordinates.

---

## ADR-002: Multi-Resolution Data Handling (No Pseudo-Downsampling)

- **Status**: Approved (M0 / M1.0)
- **Decision**: Retain each data modality at its native spatial resolution. Perform multi-modal fusion through hierarchical alignment rather than brute-force raster downsampling.
- **Consequences**: Preserves scientific validity and prevents misleading localized hydrological claims.

---

## ADR-003: Phased Pilot Spatial Rollout

- **Status**: Approved (M0 / M1.0)
- **Decision**: Implement and validate the full processing pipeline over a focused representative AOI (Moiben-Soy Pilot Zone: `35.150–35.350° E, 0.550–0.750° N`) before county-wide execution.
- **Consequences**: Accelerates development, testing, and debugging.

---

## ADR-004: Screening Engine Decoupled from Causal Diagnosis

- **Status**: Approved (M0 / M1.0)
- **Decision**: Define the core diagnostic output as **Stress Anomaly & Spatial Prioritization** rather than causal disease/pest classification.
- **Consequences**: Establishes honest, defensible system deliverables.

---

## ADR-005: Pure Python 3.11 Scientific Geospatial Stack

- **Status**: Approved (M1.0)
- **Decision**: Standardize on Python 3.11 with `xarray`, `rioxarray`, `odc-stac`, `pystac-client`, `rasterio`, `netcdf4`, `geopandas`, and `dask`.
- **Consequences**: Out-of-core multidimensional computation and cloud-native STAC interoperability.

---

## ADR-006: Externalized YAML Configuration Management

- **Status**: Approved (M1.0)
- **Decision**: Externalize all parameters into `configs/project.yaml` and load them via a typed helper (`src.utils.config.load_config`) with repository-relative fallback resolution.
- **Consequences**: Ensures clean environment-agnostic execution and reliable testing.

---

## ADR-007: Official Product Definition & Ingestion for GRAFS

- **Status**: Approved & Live-Verified (M2.5 / M2.X)
- **Decision**: Formally define GRAFS as the **Global Root-zone moisture Analysis & Forecasting System** (ANU / NCI THREDDS OPeNDAP & DE Africa), providing topsoil (`s0`, 0–5 cm) and root-zone (`s1`, 0–1 m) soil water index at native 0.1° (~10 km) resolution.
- **Consequences**: The loader explicitly documents GRAFS as a satellite-guided data assimilation product (SMAP+GPM via 4DVAR), strictly prohibiting false claims of direct field-measured root-zone moisture.

---

## ADR-008: Canonical Cropland Mask Selection & Disambiguation

- **Status**: Approved & Live-Verified (M2.6 / M2.X / M3.0)
- **Decision**: Select **Digital Earth Africa Cropland Extent Map (20 m)** (`crop_mask_eastern`, band `mask`, native 20 m in `EPSG:32736`) as the canonical v1 agricultural boundary filter, and designate **ESA WorldCover 10 m** (`esa_worldcover_2021`, Class 40) as a separate fallback.
- **Consequences**: Establishes clear, unambiguous product provenance and qualifies that the mask does not represent unverified mono-crop maize or dynamic current-season truth without phenological curve coupling. Target-grid alignment to 10 m optical features (or 20 m SWIR features) does not change its native 20 m resolution.

---

## ADR-009: Decoupled Multi-Resolution Alignment Container (`MultiResolutionCube`)

- **Status**: Approved & Live-Verified (M2.7 / M2.X)
- **Decision**: Implement `MultiResolutionCube` in `src.preprocessing.alignment`, holding separate native-scale xarray Datasets alongside explicit modality schemas (`ModalitySchema`).
- **Consequences**: Enables clean, decoupled multi-scale analysis and documented zonal/hierarchical aggregations.

---

## ADR-010: Standardized Provenance Metadata Requirement

- **Status**: Approved & Live-Verified (M2.9 / M2.X)
- **Decision**: Mandate that all ingested datasets attach standardized `provenance_*` metadata (source name, URL, product version, access timestamp, spatial/temporal resolution, CRS, applied transformations, known limitations) via `src.utils.provenance`.
- **Consequences**: Enforces auditability and dataset transparency across the entire processing lifecycle.

---

## ADR-011: Sentinel-2 Memory Integrity via Native Integer Preservation

- **Status**: Approved & Live-Verified (M2.X)
- **Decision**: Preserve native `uint16` reflectance values (scale factor $10^{-4}$) and `uint8` SCL throughout ingestion and lazy loading, deferring floating-point conversion to downstream index calculations.
- **Consequences**: Drastically reduces memory usage during multi-temporal stack ingestion and enables Dask lazy evaluation.

---

## ADR-012: Dual-Frequency CHIRPS Architecture (Monthly Baseline + Daily High-Frequency)

- **Status**: Approved & Live-Verified (M2.X)
- **Decision**: Structure CHIRPS ingestion to support both `monthly` mode (for long-term climatology baselines) and `daily` mode (for intra-seasonal rainfall deficit tracking) via `load_chirps_data(frequency="monthly"|"daily")`.
- **Consequences**: Allows efficient monthly climatology loading while retaining daily event capability.

---

## ADR-013: Resolution-Aware Mixed Index Alignment and Safeguards for NDMI

- **Status**: Approved (M3.0)
- **Decision**: When computing mixed-resolution optical indices such as NDMI (combining 10 m NIR with 20 m SWIR), downsample the higher-resolution band (NIR B08) to match the 20 m SWIR (B11) grid using area-weighted averaging. Before resampling, mandate validation of CRS match, spatial extent overlap, coordinate orientation, and approximate 2:1 resolution relationship, raising a descriptive `GridAlignmentError` on mismatch.
- **Consequences**: Prevents unverified spatial distortions or false pixel-level precision, keeping moisture indices strictly at their physical 20 m scale.

---

## ADR-014: Compositing Policy, Observation Accounting, and Missingness Preservation

- **Status**: Approved (M3.0)
- **Decision**: Implement 14-day median temporal compositing alongside an exact clear-observation counter (`valid_obs_count`). Temporal rolling smoothing must strictly preserve original missingness (`where(composite.notnull(), np.nan)`), prohibiting implicit gap filling or artificial interpolation across persistent cloud gaps.
- **Consequences**: Guarantees truthful missingness reporting during the wet cloudy Long Rains season while tracking data density for downstream anomaly weighting.

---

## ADR-015: Cross-CRS Climatology Grid Contract & Area-Preserving Aggregation

- **Status**: Approved (M4.0)
- **Decision**: Designate the reference climatology dataset (`ndvi_climatology_ls`, native `EPSG:6933`) as the authoritative spatial grid for standardized vegetation anomaly evaluation. High-resolution Sentinel-2 NDVI composites in working projection `EPSG:32736` must be aggregated onto this 30 m grid using area-preserving average downsampling. Under no circumstances may historical baseline rasters be interpolated or upscaled to high resolution.
- **Consequences**: Masked cloudy pixels contribute zero weight to both numerator and denominator; target cells lacking valid source pixels strictly evaluate to `NaN`, preventing artificial bias.

---

## ADR-016: Zero-Variance Protection, Strict Baseline Continuity, and Calendar-Driven Hydrological Dynamics

- **Status**: Approved (M4.0)
- **Decision**:
  1. Standardized anomaly evaluations ($Z_R, Z_{\text{NDVI}}$) must protect invariant pixels ($\sigma \le \text{std\_epsilon}$) by masking to `NaN` rather than adding a fixed epsilon to the denominator.
  2. The 30-year CHIRPS climatology baseline (1991–2020) must strictly validate 360-month continuity via `DataCompletenessError`.
  3. Hydrological dynamics ($\Delta \text{SWI}_{14d}$) must strictly resample to the optical pipeline's `target_time_bins` and evaluate differences exclusively on root-zone moisture ($s_1$), assigning `NaN` to the initial time step ($t=0$).
- **Consequences**: Eliminates division-by-zero artifacts, ensures rigorous climatological baseline compliance, and aligns cross-modal temporal analysis without inventing dates.

---

## ADR-017: Cross-Scale Diagnostic Fusion — 30 m Evaluation Grid with Coarse Contextual Evidence

- **Status**: Approved (M5.0)
- **Decision**: The authoritative 30 m EPSG:6933 grid derived from `ndvi_climatology_ls` is the **diagnostic evaluation grid** for M5 screening. Each grid pixel defines a **location** at which multimodal evidence is evaluated. This does NOT claim that all input modalities carry 30 m spatial information content. CHIRPS (~5.5 km) and GRAFS (~10 km) contribute coarse contextual evidence assigned to each 30 m pixel via nearest-neighbor source-cell lookup.
- **Consequences**: Prevents false claims of 30 m hydrological or meteorological precision. Honest spatial documentation is preserved.

---

## ADR-018: CHIRPS Nearest-Neighbor Contextual Lookup (No 30 m Interpolation)

- **Status**: Approved (M5.0)
- **Decision**: CHIRPS Z_R values (~5.5 km native resolution) are assigned to the 30 m diagnostic grid using **nearest-neighbor lookup only** (`assign_coarse_to_diagnostic_grid`). Bilinear, cubic, or any other spatial interpolation is explicitly prohibited. Multiple 30 m pixels within the same CHIRPS source cell receive **identical values**, representing shared coarse contextual evidence.
- **Consequences**: Prevents manufacture of spatially continuous 30 m precipitation fields. Source-cell identity is documented in output attributes. The repeated value pattern is scientifically honest; it does not imply independent 30 m measurements.
- **Scientific caveat**: All CHIRPS thresholds applied in diagnostic rules are initial screening assumptions requiring future field validation.

---

## ADR-019: GRAFS Nearest-Neighbor Contextual Lookup (No 30 m Interpolation)

- **Status**: Approved (M5.0)
- **Decision**: GRAFS $s_1$ (absolute SWI) and $\Delta\text{SWI}_{14d}$ values (~10 km native resolution) are assigned to the 30 m diagnostic grid using **nearest-neighbor lookup only**. Absolute SWI must NEVER be reconstructed from $\Delta\text{SWI}_{14d}$ via cumulative summation or integration. Bilinear interpolation of GRAFS is explicitly prohibited.
- **Consequences**: Prevents manufacture of spatially continuous 30 m hydrological surfaces. Source-cell identity preserved in output attributes.

---

## ADR-020: Diagnostic Rule Semantics — Cases A–D Scientific Interpretation Boundary

- **Status**: Approved (M5.0)
- **Decision**: The four M5 screening rules are **screening hypotheses**, not validated causal diagnoses:
  - **Case A** (Z_R < −0.8 AND Z_NDVI < −1.0): Coincident rainfall deficit and vegetation anomaly. Does NOT establish drought causality or any specific stress agent.
  - **Case B** (Z_R ≥ 0.0 AND Z_NDVI < −1.2 AND SWI ≥ 0.30): Vegetation anomaly under apparently non-dry hydro-meteorological conditions. Does NOT prove pest, pathogen, nutrient deficiency, biotic stress, waterlogging, delayed planting, or any other causal mechanism. Possible explanations require independent field validation.
  - **Case C** (Z_R < −1.0 AND Z_NDVI ≥ −0.5 AND SWI ≥ 0.30): Absence of vegetation anomaly despite rainfall deficit coincident with adequate root-zone moisture — consistent with hydrological buffering. Does NOT prove any specific protective mechanism.
  - **Case D** (Z_R ≥ 0.5 AND $\Delta\text{SWI}_{14d}$ ≤ 0.0): Rainfall/root-zone moisture disconnect (non-increasing SWI despite above-average rainfall). Does NOT prove runoff, soil crusting, infiltration failure, or any topographic effect.
- **Consequences**: Correct interpretation is "candidate multi-modal screening evidence requiring field validation". All thresholds are initial screening assumptions requiring future field validation.

---

## ADR-021: Rule Overlap Policy — Cases B+D MULTI_SIGNAL State

- **Status**: Approved (M5.0)
- **Decision**: Exhaustive pairwise analysis of Cases A–D reveals that **Cases B and D can co-occur**. Cases A/B, A/C, A/D, B/C, and C/D are mutually exclusive by construction (their Z_R or Z_NDVI conditions cannot simultaneously be satisfied). Cases B and D overlap when Z_R ≥ 0.5 (satisfying both B's Z_R ≥ 0.0 and D's Z_R ≥ 0.5) simultaneously with Z_NDVI < −1.2, SWI ≥ 0.30, and $\Delta\text{SWI}_{14d}$ ≤ 0.0. Example: Z_R=0.8, Z_NDVI=−1.8, SWI=0.45, $\Delta\text{SWI}_{14d}$=−0.05. The simultaneous signal (vegetation anomaly under apparently non-dry conditions, yet root-zone moisture non-increasing) is scientifically meaningful and must NOT be suppressed via silent `if/elif` ordering. **Policy: `MULTI_SIGNAL` (value=5) is the authoritative classification for simultaneous B+D.**
- **Consequences**: `MULTI_SIGNAL` is included in `ACTIONABLE_CASES`. Rule precedence is documented rather than implicit.

---

## ADR-022: SWI Threshold Semantics — Cases B and C Enabled

- **Status**: Approved (M5.0)
- **Decision**: GRAFS `s1` (Root-Zone Soil Water Index, 0–100 cm) is documented in `DATA_DICTIONARY.md` as a dimensionless fraction in [0.0, 1.0]. The threshold SWI ≥ 0.30 is operationally interpretable because the source variable has a known and documented [0,1] scale. Based on this confirmed documented semantics, Cases B and C are **enabled** (`swi_enabled=True` by default). Absolute SWI must NEVER be reconstructed from $\Delta\text{SWI}_{14d}$.
- **Scientific caveat**: The SWI ≥ 0.30 threshold is an initial M5 screening assumption ONLY. It is NOT an empirically validated agronomic adequacy threshold, NOT calibrated to any specific East African soil-texture profile, and NOT derived from in-situ field capacity or wilting-point measurements.

---

## ADR-023: Case-D Production Distribution Sanity Check

- **Status**: Approved (M5.0)
- **Decision**: The Case D rule `Z_R ≥ 0.5 AND $\Delta\text{SWI}_{14d}$ ≤ 0.0` requires an empirical check to ensure it does not classify nearly all pixels. In the pilot 2023 April observation: The CHIRPS April 2023 Z_R for Moiben AOI was strongly positive (mean ~+2.0 to +2.5), making the Z_R ≥ 0.5 condition broadly satisfied. The fraction of pixels with $\Delta\text{SWI}_{14d}$ ≤ 0.0 depends on actual hydrological conditions. If this fraction exceeds ~90%, it indicates that Case D may not be discriminating under conditions of exceptional rainfall. The threshold is **reported, not silently changed**.
- **Consequences**: Case D requires monitoring in production. If systematically too broad, future validation work should inform threshold adjustment. No automatic threshold change is made based on this sanity check alone.

---

## ADR-024: Evidence Confidence Policy — Completeness Fraction, Not Diagnostic Probability

- **Status**: Approved (M5.0)
- **Decision**: The M5 evidence confidence metric (`evidence_confidence`) is defined as the fraction of provided modalities that are non-NaN at each diagnostic pixel. Range: [0.0, 1.0]. This is evidence **completeness**, not probability of correct causal diagnosis. The composite severity formula involving weighted combinations of $|Z_{\text{NDVI}}|$, $|Z_R|$, SWI, and NDMI is NOT implemented, as it cannot be justified as a scientifically validated formulation (in particular, large positive rainfall anomaly magnitude must not increase drought severity).
- **Consequences**: No fabricated composite severity score. Output preserves diagnostic case, evidence state, and completeness fraction.

---

## ADR-025: Insufficient Evidence Policy — Value = −1, Cannot Become NORMAL

- **Status**: Approved (M5.0)
- **Decision**: `DiagnosticCase.INSUFFICIENT_EVIDENCE` is assigned the integer value **−1** (not 0). `DiagnosticCase.NORMAL` is assigned value **0**. This ensures that default integer array initialization can never accidentally produce a false NORMAL classification from uninitialized pixels. NaN inputs to required rule predicates cause that rule to be unevaluable for the affected pixel. If ALL evidence pathways (z_r+z_ndvi for Cases A/B/C and z_r+delta_swi for Case D) are unavailable (all NaN), the pixel is classified as `INSUFFICIENT_EVIDENCE`. If at least one pathway has valid data but no rule fires, the pixel is `NORMAL`.
- **Consequences**: Missing evidence cannot silently become NORMAL. The distinction is enforced in both implementation and tests.

---

## ADR-026: MMU and Morphology — 2 ha Minimum, Actual Projected Area, 8-Connected, 3×3 Opening

- **Status**: Approved (M5.0)
- **Decision**:
  1. **MMU**: Minimum Mapping Unit is 2.0 ha (20,000 m²). Calculated from actual projected pixel dimensions (`transform.a × transform.e`), NOT assumed 30 m × 30 m pixel size.
  2. **Morphological opening**: Optional 3×3 structuring element applied before labeling to remove isolated single pixels and small speckle. Tests confirm it preserves clusters ≥ 2 ha and removes single-pixel noise.
  3. **Connected components**: 8-connected labeling (`structure=np.ones((3,3))`).
  4. **Pipeline order**: actionable mask → morphological opening → 8-connected labeling → area calculation → MMU filtering → polygonization.
- **Consequences**: Prevents single-pixel spurious zones from entering scouting output. Area-based MMU provides geographically accurate filtering independent of assumed pixel size.

---

## ADR-027: Scientific Interpretation Boundary — Screening Hypotheses, Not Validated Diagnoses

- **Status**: Approved (M5.0)
- **Decision**: The M5 multi-modal screening engine is a **hypothesis generator**, not a causal inference engine. The diagnostic output (Case A–D, NORMAL, INSUFFICIENT_EVIDENCE, MULTI_SIGNAL) represents candidate stress/anomaly evidence requiring independent field validation before operational agronomic decisions are made. The system must never claim to have:
  - proven drought causality (Case A);
  - identified pest, pathogen, nutrient, or biotic stress (Case B);
  - confirmed hydrological buffering mechanisms (Case C);
  - detected runoff, soil crusting, or infiltration failure (Case D).
  Scouting zones are candidate locations for field inspection, not confirmed crop-stress diagnoses. **Field validation remains future work.**
- **Final Claim**: "M5 rule-based multi-modal screening implementation verified against defined engineering, spatial, temporal, and scientific-contract tests. The diagnostic rules remain unvalidated screening hypotheses and are not evidence of causal crop stress."

---

## ADR-028: Intentional NDMI Exclusion from M5 Diagnostic Screening

- **Status**: Approved (M5.0)
- **Decision**: The 20 m EPSG:32736 NDMI product (`src/features/indices.py`) remains intentionally excluded from the M5 diagnostic screening matrix. No M5-approved cross-grid alignment contract currently exists to bring 20 m UTM NDMI onto the 30 m EPSG:6933 reference climatology grid. Consuming NDMI without explicit area-preserving aggregation would violate the project's spatial resolution contracts.
- **Consequences**: NDMI is excluded from M5 screening attributes (reported as `null` where referenced). No NDMI data is fabricated or unfaithfully downsampled.

---

## ADR-029: Upstream 2023 GRAFS Data Availability & Production Season Status

- **Status**: Ratified (M5.0 Verification)
- **Decision**: An exhaustive investigation of the authoritative NCI Australia THREDDS OPeNDAP server confirmed that:
  1. The canonical `/ub8/global/GRAFS/` root hosts datasets strictly for years 2015 through 2022 (returning HTTP 404 for 2023).
  2. The `/ub8/global/GRAFS/GRAFS_v2/` subcatalog contains 2023 NetCDF files, but these contain only 5 daily slices for root-zone SWI (`2023-02-27` to `2023-03-03`) and 5 daily slices for topsoil wetness (`2023-11-08` to `2023-11-12`).
  3. The canonical 2023 Long Rains production season (`2023-03-04` through `2023-09-30`, covering 14-day bins 2 through 15) is completely absent in the upstream ANU/NCI repository.
  4. In accordance with the project's non-negotiable scientific integrity contracts, the pipeline strictly prohibits fabricating missing observations, guessing data, or silently substituting non-contemporaneous years (e.g. 2022) into a production 2023 analysis.
- **Consequences**: Full temporally matched 2023 production execution across all 15 canonical bins is formally recognized as **BLOCKED BY DATA AVAILABILITY** at the upstream data provider. Live integration tests continue to use the verified 2022 OPeNDAP subset for automated testing of the complete multi-modal screening pipeline without compromising temporal contracts.

---

## ADR-030: NASA SMAP Level-4 (SPL4SMGP V008) Fallback Ingestion Module

- **Status**: Approved & Implemented (Isolated Ingestion Milestone)
- **Decision**: NASA SMAP Level-4 Global 3-hourly 9 km EASE-Grid Surface and Root Zone Soil Moisture Geophysical Data (SPL4SMGP Version 8) is adopted as the validated operational fallback for continuous root-zone soil water index data across the 2023 Long Rains production season (`2023-03-01` to `2023-09-30`).
  1. **Variable Equivalence**: `sm_rootzone_wetness` (0–100 cm) maps to `s1` as a dimensionless relative saturation fraction in $[0.0, 1.0]$. `sm_surface_wetness` (0–5 cm) maps to `s0`.
  2. **Temporal Aggregation**: 3-hourly observations are aggregated to daily arithmetic means before ingestion into the frozen 14-day half-open interval resampling engine (`resample_soil_moisture_to_calendar`).
  3. **Threshold Transfer**: The frozen M5 threshold $\text{SWI} \ge 0.30$ and the $\Delta\text{SWI}_{14d}$ differencing logic apply with identical physical meaning ($30\%$ root-zone saturation).
  4. **Isolated Architecture**: Implementation is strictly encapsulated in `src/ingestion/smap.py`. M1–M5 core algorithms and feature modules remain completely unmodified.
  5. **Explicit Provenance**: All datasets ingested via SMAP L4 carry explicit top-level attribute `source="SMAP_L4"` and `soil_moisture_source="NASA_SMAP_L4_SPL4SMGP_V008"`.

---

## ADR-031: Milestone 6 Production Reporting, Visualization & Decision Products Architecture

- **Status**: Approved & Implemented (Milestone 6)
- **Decision**: The reporting and visualization layer for Milestone 6 is encapsulated strictly within dedicated packages (`src/reporting/`, `src/visualization/`, and `src/export/production_export.py`) without modifying frozen M1–M5 feature extraction, diagnostic rules, or spatial/temporal contracts:
  1. **Executive Summaries**: Machine-readable JSON and publication-grade Markdown summaries are generated deterministically by `src/reporting/summary.py`.
  2. **Spatial Persistence & Hotspots**: Recurrence frequency of actionable screening flags ($[0, 15]$ bins) is computed by `src/reporting/persistence.py` without introducing arbitrary weighting models.
  3. **Visual Decision Products**: Publication-quality 4-panel temporal diagnostic profiles, stacked candidate area timelines, geospatial diagnostic maps, and spatial persistence heatmaps are generated with headless Matplotlib (`Agg` backend) by `src/visualization/`.
  4. **Contractual Marker for Bin 1**: In temporal profiles, $\Delta\text{SWI}_{14d}$ at $t=0$ (Bin 1) is explicitly marked as "Undefined (t=0)" to clearly distinguish contractual boundaries from missing data.
  5. **Consolidated Seasonal GeoJSON**: All per-bin MMU-filtered candidate scouting polygons are compiled into `export/geojson/seasonal_scouting_zones_2023.geojson` with complete multi-modal diagnostic provenance.
  6. **Scientific Guardrails**: All generated reports and figures embed explicit scientific notices preserving non-overclaiming boundaries (screening hypotheses vs. causal diagnoses; shared coarse contextual forcing vs. independent 30 m measurements; candidate area vs. confirmed crop loss).



