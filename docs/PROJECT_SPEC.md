# Project Specification: Multi-Modal Crop Stress Intelligence System — Kenya

## 1. Executive Summary & Objective

The **Multi-Modal Crop Stress Intelligence System** is an earth observation (EO) and climate data analytics pipeline designed to detect, screen, and spatially prioritize vegetative stress across maize-growing regions of **Uasin Gishu County, Kenya**.

The system operates as an **early-warning and spatial screening framework**, providing field officers, agronomists, and agricultural stakeholders with actionable spatial intelligence on emerging crop stress zones.

---

## 2. Core Scientific & Operational Principles

1. **Multi-Resolution Architecture**:
   - High-resolution optical data (Sentinel-2) is retained and processed at native **10–20 m** resolution (`EPSG:32736`).
   - Gridded climate precipitation (CHIRPS) is retained at native **~5.5 km (0.05°)** scale (`EPSG:4326`).
   - Hydrological soil water index (GRAFS: Global Root-zone moisture Analysis & Forecasting System) is retained at native **~10 km (0.1°)** scale (`EPSG:4326`).
   - Agricultural cropland boundary mask (`crop_mask_eastern`) is ingested at native **20 m** scale (`EPSG:32736`) with explicit non-attribution qualifications. Target-grid alignment to 10 m optical features in downstream masking does **NOT** alter its native 20 m resolution.
   - **Strict Rule**: Coarse datasets must **NOT** be falsely interpolated or resampled to appear as high-resolution (20 m) information.

2. **Screening vs. Causal Diagnosis**:
   - The platform is an **anomaly screening and prioritization engine**, *not* a causal disease, pest, or nutrient diagnosis engine.
   - Anomalies indicate deviations in vegetation vigor, canopy moisture, or biomass accumulation, requiring ground/drone verification for causal attribution.

3. **Phased Spatial Rollout**:
   - Initial implementation targets a focused Area of Interest (AOI) within Uasin Gishu County before scaling county-wide.

4. **Spatial Reference System**:
   - All optical and local spatial analyses use **EPSG:32736** (WGS 84 / UTM Zone 36S).

---

## 3. System Architecture & Component Packages

The repository is modularized under `src/`:

```
src/
├── ingestion/       # Data ingestion loaders for Sentinel-2, CHIRPS, GRAFS, and Crop Mask
├── preprocessing/   # SCL cloud masking, MultiResolutionCube, compositing, alignment schemas
├── features/        # Spectral indices (NDVI, EVI, NDMI), baseline statistics, anomaly metrics
├── diagnostics/     # Multi-modal stress matrix, spatial screening, confidence scoring
└── utils/           # Configuration parsing, AOI validation, provenance tracking, helpers
```

---

## 4. Scope & Capabilities

| In Scope | Deferred / Out of Scope |
| :--- | :--- |
| Focused AOI in Uasin Gishu County | County-wide batch processing |
| Multi-modal ingestion (Sentinel-2, CHIRPS, GRAFS, Mask) | Machine learning yield forecasting |
| Optical index computation & anomaly screening | Automated UAV path routing |
| Offline configuration & reproducible test suite | Production web dashboards / real-time alerting |
| Decoupled multi-resolution data cube management | Causal pathogen/nutrient classification |
| Standardized data provenance metadata | In-situ probe root-zone moisture claims |

---

## 5. Milestone Roadmap

- **M1.0 — Engineering Foundation** *(Completed)*: Repository structure, Python 3.11 environment, config loader, offline test framework, documentation.
- **M2.0 / M2.X — Multi-Modal Ingestion & Live Verification** *(Completed)*: Verified GRAFS soil water index loader, CHIRPS loader (monthly/daily), Sentinel-2 loader, canonical 20 m agricultural mask (`crop_mask_eastern`), decoupled `MultiResolutionCube` alignment schema, AOI validation, and live network integration suite.
- **M3.0 — Optical Preprocessing & Biophysical Feature Engineering** *(Completed)*: SCL cloud/shadow masking, lazy reflectance scaling, 10 m NDVI/EVI, 20 m NDMI with grid alignment safeguards, 14-day median compositing, observation accounting (`valid_obs_count`), missingness-preserving rolling smoothing, cropland prior filtering, and quality diagnostics matrix.
- **M4.0 — Multi-Modal Anomaly Baseline Extraction & Stress Metrics**: Time-series historical baseline extraction, standardized anomaly scoring (Z-scores, VCI), and cross-scale stress metric fusion.
- **M5.0 — Spatial Screening & Diagnostic Matrix**: Multi-modal matrix evaluation combining moisture, hydrological state, and optical response into screening priority categories.
- **M6.0 — Pilot AOI Evaluation**: End-to-end seasonal validation over Uasin Gishu maize cropping calendar.
