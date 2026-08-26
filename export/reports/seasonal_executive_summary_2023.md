# Seasonal Diagnostic Screening Executive Summary

**Target Season**: 2023 Long Rains (`2023-03-01` to `2023-09-27`)  
**Agricultural Zone**: Moiben-Soy Agricultural Pilot Zone (`ug_pilot_moiben_01`)  
**Bounding Box (WGS84)**: `[35.15, 0.55, 35.35, 0.75]`  
**Generated UTC**: `2026-08-26T18:38:17.384514+00:00`  
**System Version**: `0.4.0` | **Matrix Version**: `M5.0 Frozen Matrix (ADR-021)`  

---

## 1. Executive Summary & Headline Indicators

| Headline Indicator | Seasonal Production Value | Operational Interpretation |
| :--- | :---: | :--- |
| **Canonical Evaluation Bins** | **15** (14-day intervals) | Full contiguous seasonal coverage with zero gaps |
| **Actionable Bins** | **7** of 15 bins | Bins exhibiting MMU-filtered candidate scouting clusters |
| **Total Extracted Clusters** | **28** | Distinct $\ge 2.0\text{ ha}$ connected candidate polygons |
| **Cumulative Candidate Area** | **3202.47 ha** | Sum of actionable polygon areas across all bins |
| **Maximum Single-Bin Area** | **503.10 ha** (Bin 10) | Peak spatial extent of screening flags |
| **First Actionable Date** | `2023-03-01` | Early-season negative vegetation anomaly screening |
| **Last Actionable Date** | `2023-09-13` | Late-season precipitation-to-moisture rate disconnect |

---

## 2. Seasonal Pixel-Level Classification Distribution

| Diagnostic Category | Pixel Count | % of Season | Screening Semantic |
| :--- | :---: | :---: | :--- |
| **NORMAL (0)** | 43,926 | 52.4% | Canopy greenness and moisture within expected climatological ranges |
| **CASE A (1) — Coincident Precipitation & Vegetation Deficit** | 0 | 0.0% | Coincident precipitation deficit ($Z_R \le -0.8$) and vegetation anomaly ($Z_{\text{NDVI}} \le -1.0$) |
| **CASE B (2) — Vegetation Anomaly Under Non-Dry Priors** | 6,384 | 7.6% | Vegetation anomaly ($Z_{\text{NDVI}} \le -1.2$) under non-dry hydro-meteorology ($Z_R \ge 0.0, \text{SWI} \ge 0.30$) |
| **CASE C (3) — Precipitation Deficit Without Vegetation Anomaly** | 0 | 0.0% | Precipitation deficit ($Z_R \le -0.8$) without vegetation anomaly (non-actionable) |
| **CASE D (4) — Precipitation-to-Moisture Disconnect** | 27,137 | 32.4% | Above-normal rainfall ($Z_R \ge 0.5$) with non-increasing root-zone wetness ($\Delta\text{SWI}_{14d} \le 0.0$) |
| **MULTI_SIGNAL (5) — B+D Co-occurrence** | 6,403 | 7.6% | Simultaneous non-dry canopy anomaly and hydrological rate-of-change disconnect |
| **INSUFFICIENT_EVIDENCE (-1)** | 0 | 0.0% | Cloud-obscured or unobserved pixels (default safe state) |

---

## 3. Actionable Scouting Zones Breakdown

| Bin # | Bin Start | Clusters | Candidate Area (ha) | Dominant Case | Operational Priority |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `2023-03-01` | 22 | 183.87 | `CASE_B` | High Priority Scouting |
| **10** | `2023-07-05` | 1 | 503.10 | `CASE_D` | High Priority Scouting |
| **11** | `2023-07-19` | 1 | 503.10 | `CASE_D` | High Priority Scouting |
| **12** | `2023-08-02` | 1 | 503.10 | `CASE_D` | High Priority Scouting |
| **13** | `2023-08-16` | 1 | 503.10 | `CASE_D` | High Priority Scouting |
| **14** | `2023-08-30` | 1 | 503.10 | `CASE_D` | High Priority Scouting |
| **15** | `2023-09-13` | 1 | 503.10 | `MULTI_SIGNAL` | High Priority Scouting |

---

## 4. Multi-Modal Provenance & Data Sources

- **Hydrology / Root-Zone Soil Moisture**: NASA SMAP Level-4 (SPL4SMGP V008)
- **Optical Vegetation Dynamics**: Digital Earth Africa Sentinel-2 L2A
- **Rainfall Climatology**: CHIRPS Monthly Climatology (1991-2020 Baseline)
- **Diagnostic Grid Resolution**: 30 m equal-area projection (`EPSG:6933`)
- **Minimum Mapping Unit (MMU)**: $\ge 2.0\text{ ha}$ ($20,000\text{ m}^2$)

---

## 5. Non-Overclaiming Scientific Caveats & Operating Limits

1. Candidate scouting zones represent priority screening hypotheses to guide field verification, NOT confirmed yield loss or crop damage.
2. Case B reflects vegetation anomalies under non-dry conditions; it does NOT prove insect pests, fungal pathogens, or specific disease presence.
3. Case D reflects above-average rainfall coincident with non-increasing root-zone wetness; it does NOT prove soil crusting, runoff failure, or infiltration impedance.
4. Spatial Scale Disparity Limitation: CHIRPS (5.5 km) and SMAP (9 km) data provide coarse regional forcing broadcast to the 30 m grid via nearest neighbor. Over the 503.10 ha focal block, regional hydro-meteorological conditions are spatially uniform.
5. Evaluated Spatial Extent: The 503.10 ha (86x65, 30 m) grid represents the intentional frozen focal pilot evaluation block within the broader 490.5 km² regional pilot AOI.
6. SWI >= 0.30 and Z_R >= 0.50 are operational screening priors, not empirically calibrated in-situ agronomic adequacy thresholds.
