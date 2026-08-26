# Crop Stress Intelligence Kenya — Deliverables Guide (`export/`)

Welcome to the public stakeholder deliverables directory for the **End-to-End Multi-Modal Agricultural Early Warning & Crop Health Intelligence Pipeline** (2023 Long Rains Season, Moiben-Soy Agricultural Pilot Zone, Uasin Gishu County, Kenya).

This directory contains standalone, production-ready deliverables formatted for direct consumption by agricultural extension teams, NGO program officers, research reviewers, and GIS analysts without requiring access to the internal Python codebase.

---

## 1. Directory Structure

```
export/
├── README.md                                  # This stakeholder guide & interpretation manual
│
├── figures/                                   # Publication-Grade Visual Decision Products (300 DPI)
│   ├── project_overview_diagram.png           # 01: System architecture & multi-modal pipeline flow
│   ├── satellite_context_regional_2023.png    # 02: Regional satellite context (490.5 km² Pilot AOI)
│   ├── satellite_context_focal_2023.png       # 03: Focal landscape context (503.10 ha Hero image)
│   ├── temporal_diagnostic_profile_2023.png   # 04: Four-panel seasonal trajectory (Z_R, Z_NDVI, SWI, ΔSWI)
│   ├── diagnostic_area_timeline_2023.png      # 05: Stacked timeline of candidate scouting area (ha)
│   ├── satellite_flag_to_triage_2023.png      # 06: Flag-to-Triage progression (Landscape → Anomaly → Polygons)
│   ├── spatial_diagnostic_bin_01.png          # 07: Bin 1 (2023-03-01) 22 Case B clusters (183.87 ha)
│   ├── spatial_persistence_hotspots_2023.png  # 08: Multi-bin spatial recurrence heatmap (0–7 count)
│   ├── spatial_diagnostic_bin_15.png          # 09: Bin 15 (2023-09-13) MULTI_SIGNAL compound (503.10 ha)
│   ├── spatial_diagnostic_bin_10.png          # Supporting: Bin 10 (2023-07-05) Case D saturation (503.10 ha)
│   └── spatial_diagnostic_bin_11..14.png      # Supporting Archive: Intermediate Case D maps
│
├── geojson/                                   # GIS-Ready Vector Scouting Zones (WGS84 / RFC 7946)
│   ├── seasonal_scouting_zones_2023.geojson   # 10: Consolidated seasonal FeatureCollection (28 features)
│   ├── scouting_zones_bin_01_20230301.geojson # Bin 1 candidate polygons (22 features)
│   ├── scouting_zones_bin_10_20230705.geojson # Bin 10 candidate polygon (1 feature)
│   ├── scouting_zones_bin_15_20230913.geojson # Bin 15 candidate polygon (1 feature)
│   └── scouting_zones_bin_02..14.geojson      # Per-bin GeoJSON feature collections
│
└── reports/                                   # Executive Briefings & Metadata Manifests
    ├── portfolio_case_study_2023.md           # Comprehensive portfolio case study & technical presentation
    ├── seasonal_executive_summary_2023.md     # Stakeholder markdown executive summary
    ├── seasonal_executive_summary_2023.json   # Machine-readable JSON execution manifest
    └── satellite_context_metadata.json        # Provenance metadata for satellite context products
```

---

## 2. Satellite Context Products (Milestone 7)

Authentic Earth Observation imagery is integrated into the visual portfolio to provide physical geographic orientation and visual interpretability:

* **Why Imagery is Included**: Contextual satellite imagery allows non-technical stakeholders to answer *"Where is this?"* and *"What does the agricultural landscape physically look like?"* before evaluating mathematical anomaly flags.
* **Analytical vs. Contextual Imagery**:
  * **Analytical Data**: 14-day surface reflectance (B04, B08), standardized anomalies ($Z_{\text{NDVI}}$, $Z_R$), and root-zone soil moisture ($\text{SWI}$, $\Delta\text{SWI}_{14d}$).
  * **Contextual Imagery**: Sentinel-2 Level-2A True-Colour Image (`TCI.tif`, RGB: B04, B03, B02 at native 10 m resolution) acquired on **2023-03-07** (during Canonical Bin 1).
* **Sensor & Source**: Sentinel-2B Multi-Spectral Instrument (MSI), processed to Level-2A Bottom-of-Atmosphere (BOA) reflectance by Digital Earth Africa / Copernicus ESA (`s3://deafrica-sentinel-2/`).
* **Provenance & Metadata**: Full spatial, spectral, and licensing metadata is recorded in [`reports/satellite_context_metadata.json`](reports/satellite_context_metadata.json).
* **Interpretability Notice**: Contextual satellite imagery provides landscape orientation and spatial triage guidance; visual inspection does NOT establish causal plant disease, pest presence, or soil degradation. Any future imagery acquired outside the 2023 season represents current landscape context and is explicitly non-evidence for 2023 conditions.

---

## 3. Portfolio Presentation Order & Visual Narrative

The 10-item canonical presentation order guides stakeholders from system design to physical grounding, seasonal dynamics, and actionable field GIS exports:

| Order | Deliverable | Type | Narrative Question / Purpose |
| :---: | :--- | :---: | :--- |
| **01** | `figures/project_overview_diagram.png` | Diagram | **"What is the system?"** — End-to-end multi-modal architecture & screening workflow. |
| **02** | `figures/satellite_context_regional_2023.png` | Figure | **"Where is this?"** — Authentic Sentinel-2 true-colour view of the 490.5 km² Regional Pilot AOI. |
| **03** | `figures/satellite_context_focal_2023.png` | Figure | **"What does the actual landscape look like?"** — 10 m true-colour hero visual of the 503.10 ha focal block. |
| **04** | `figures/temporal_diagnostic_profile_2023.png` | Figure | **"What happened over the season?"** — 4-panel synchronized time-series ($Z_R$, $Z_{\text{NDVI}}$, SWI, $\Delta\text{SWI}_{14d}$). |
| **05** | `figures/diagnostic_area_timeline_2023.png` | Figure | **"When did screening flags appear?"** — Stacked timeline of candidate scouting area (ha) across 15 bins. |
| **06** | `figures/satellite_flag_to_triage_2023.png` | Visual | **"How does satellite signal become scouting output?"** — 3-panel progression (Landscape $\to$ $Z_{\text{NDVI}}$ $\to$ Polygons). |
| **07** | `figures/spatial_diagnostic_bin_01.png` | Map | **"Early-season fragmented anomaly"** — 22 Case B candidate clusters (183.87 ha) during crop establishment. |
| **08** | `figures/spatial_persistence_hotspots_2023.png` | Map | **"Where did signals recur?"** — Multi-bin spatial recurrence frequency heatmap (0 to 7 bins). |
| **09** | `figures/spatial_diagnostic_bin_15.png` | Map | **"End-of-season compound screening"** — MULTI_SIGNAL compound state (503.10 ha) during crop maturation. |
| **10** | `geojson/seasonal_scouting_zones_2023.geojson` | GeoJSON | **"What can a field team take into GIS?"** — 28 vector scouting polygons with centroids & zonal statistics. |

---

## 4. Spatial Scale Architecture & Disparity Disclosure

The project employs a dual-scale spatial hierarchy:
* **Regional Pilot AOI (`ug_pilot_moiben_01`)**: 0.20° × 0.20° ≈ 490.5 km² (49,050 ha) in `EPSG:4326` ([35.15° E, 0.55° N, 35.35° E, 0.75° N]). Frames macro-climatic and hydrological context.
* **Evaluated Focal Sub-Grid**: 86 rows × 65 columns at 30.0 m in `EPSG:6933` (5,031,000 m² = 503.10 ha). Actively evaluated Sentinel-2 screening footprint.

### Key Scale Disparity Note:
The entire 503.10 ha focal block (2.58 km × 1.95 km) lies within a single coarse CHIRPS cell (~5.5 km) and a single SMAP cell (~9.0 km). Consequently:
1. **Case B** exhibits fine-scale spatial fragmentation (22 clusters) because it incorporates high-resolution 30 m optical data.
2. **Case D** evaluates uniformly across the entire 503.10 ha grid because its controlling inputs ($Z_R$ and $\Delta\text{SWI}_{14d}$) are coarse regional constants. This whole-grid uniformity is a structural property of the sensor resolutions, not proof of identical hydrology across every farm parcel.

---

## 5. Non-Overclaiming Scientific Caveats

1. **Screening vs. Diagnosis**: Candidate scouting polygons represent priority hypotheses to guide extension field scouting, NOT confirmed crop damage, insect pests, or diseases.
2. **No Yield Loss Claims**: The pipeline contains no crop-cutting data or yield models; it does NOT measure or forecast crop failure.
3. **No Soil Physical Diagnostics**: Case D ("Hydrological Disconnect") reflects a satellite rate-of-change rule match; it does NOT prove soil compaction, crusting, hardpans, or overland runoff.
4. **Operational Priors**: $\text{SWI} \ge 0.30$ and $Z_R \ge +0.50$ are screening thresholds, not in-situ calibrated agronomic wilting points or field capacities.
5. **No AI Forecasting**: This project performs deterministic observational screening over satellite observations; it does not deploy black-box predictive forecasting.
