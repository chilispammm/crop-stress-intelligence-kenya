# Scientific Assumptions & Methodological Caveats

This document outlines the core scientific assumptions, limitations, and methodological boundaries governing the **Crop Stress Intelligence System — Kenya**.

---

## 1. Multi-Resolution Integrity & Alignment

- **Assumption**: Earth observation datasets capture phenomena at distinct spatial and temporal scales. Sentinel-2 captures field-level canopy structure (10–20 m), whereas CHIRPS (~5.5 km) and GRAFS (~10 km) capture regional atmospheric and hydrological forcing.
- **Rule**: Coarse climate and soil moisture rasters must **never** be naively bilinear- or bicubic-resampled to 20 m to produce pseudo-field-level rainfall or soil moisture maps. 
- **Methodological Handling**: Multi-modal analysis is conducted via hierarchical alignment (e.g., evaluating high-resolution field anomalies in the context of their enclosing regional climate/soil water index cells) using decoupled multi-resolution data structures (`MultiResolutionCube`).

---

## 2. GRAFS Hydrological Nature & Scientific Limitations

- **System Definition**: GRAFS (**Global Root-zone moisture Analysis & Forecasting System**) is a near-real-time data assimilation product developed by ANU and hosted by NCI Australia.
- **Assimilation Architecture**: Assimilates satellite microwave radiometry from the **Soil Moisture Active/Passive (SMAP)** mission into a first-order Antecedent Precipitation Index (API) model driven by **Global Precipitation Measurement (GPM)** satellite rainfall via 4DVAR.
- **Crucial Qualification**: GRAFS provides satellite-guided regional topsoil (`s0`, 0–5 cm) and root-zone (`s1`, 0–1 m) soil water index estimates. It is **NOT** direct in-situ field probe measurement.
- **Scale Limitation**: At ~10 km resolution, GRAFS reflects regional hydrological regimes and cannot capture intra-field soil texture variations, microtopography, or tile drainage effects.

---

## 3. Agricultural Mask Qualifications & Ground Truth Non-Equivalence

- **Mask Scope**: A 20 m agricultural mask (Digital Earth Africa Cropland Extent `crop_mask_eastern`, native 20 m in `EPSG:32736`) identifies cultivated/arable land parcels relative to non-crop land cover (forest, shrubland, urban, water). Target-grid alignment to 10 m optical features does **NOT** alter its native 20 m resolution.
- **Caveat 1 — Crop Type Non-Specificity**: General cropland masks do **NOT** distinguish maize specifically from other seasonal crops (wheat, barley, beans, vegetables, or seasonal fallow). In Uasin Gishu, maize is the dominant staple during the Long Rains (>75% cultivated area), but general crop pixels cannot be assumed to be maize without qualification.
- **Caveat 2 — Static vs. Dynamic Truth**: A static or historical cropland mask does **NOT** constitute current-year ground truth. Farmers rotate crops or leave parcels fallow.
- **Protocol**: The crop mask must serve as a spatial boundary filter that is strictly combined with **in-season phenological growth curves** (Long Rains March–November) to screen actively cultivated maize parcels.

---

## 4. Screening vs. Causal Attribution

- **Assumption**: Spectral vegetation indices (e.g., NDVI, EVI, NDMI) detect physiological symptoms of stress (chlorophyll loss, moisture deficit, biomass reduction) but are **non-specific**.
- **Caveat**: A vegetative anomaly can be caused by moisture stress, nutrient deficiency, pest infestation (e.g., Fall Armyworm), disease (e.g., Maize Lethal Necrosis), waterlogging, or weed competition.
- **Rule**: The system classifies fields into **stress priority tiers** (Low, Moderate, High, Severe) to guide ground scouting and drone verification, explicitly avoiding unsupported causal claims.

---

## 5. Maize Cropping Calendar & Phenology in Uasin Gishu

- **Typical Calendar (Long Rains Season)**:
  - **Land Preparation & Planting**: March – April
  - **Emergence & Early Vegetative**: April – May
  - **Peak Vegetative & Tasseling / Silking**: June – July
  - **Grain Fill & Maturation**: August – September
  - **Harvesting**: October – November
- **Phenological Baseline**: Anomaly calculations compare current reflectance to historical or expected baseline values for the *same phenological window*, avoiding false alarms during normal pre-planting fallow or post-harvest dry-down.

---

## 6. Optical Cloud Gaps & SCL Masking

- **Limitation**: Uasin Gishu experiences heavy cloud cover during peak Long Rains (April–July), creating potential observation gaps.
- **Handling**: Strict Scene Classification Layer (SCL) filtering (masking cloud, shadows, and cirrus) paired with 14-day median compositing and exact observation accounting (`valid_obs_count`). Centered rolling smoothing strictly retains original missingness (`where(composite.notnull(), np.nan)`).

---

## 7. Standardized Anomaly Baselines & Temporal Alignment (Milestone 4.1)

- **Standardized Anomaly Formulation**: Continuous Z-score anomaly indices ($Z_R$, $Z_{\text{NDVI}}$) are computed against multi-decadal historical reference baselines without artificial smoothing or categorization.
- **Zero-Variance Policy**: Baseline pixels exhibiting invariant or near-zero variance ($\sigma \le 10^{-4}$) are masked to `NaN` rather than inflating anomalies with an arbitrary epsilon term in the denominator.
- **Half-Open Temporal Compositing**: All 14-day temporal aggregations (optical compositing and GRAFS soil moisture resampling) strictly follow half-open interval boundaries $[t, t + 14\text{ days})$. An observation occurring exactly on an internal boundary $t_{i+1}$ belongs exclusively to $[t_{i+1}, t_{i+2})$ and does not contribute to $[t_i, t_{i+1})$.
- **Root-Zone Moisture Dynamics**: The $\Delta\text{SWI}_{14d}$ metric is evaluated exclusively on root-zone soil water index ($s_1, 0\text{–}100\text{ cm}$) series: $\Delta\text{SWI}_{14d}(t) = s_1(t) - s_1(t-1)$, with $t=0$ assigned `NaN`.
- **Area-Preserving Baseline Contract**: High-resolution Sentinel-2 NDVI composites in working projection `EPSG:32736` are aggregated onto the authoritative reference climatology grid (`EPSG:6933`, 30 m) using area-weighted downsampling. Historical baseline rasters are **never** upscaled to the Sentinel-2 grid.
