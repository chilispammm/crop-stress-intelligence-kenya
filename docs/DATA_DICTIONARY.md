# Data Dictionary

This document defines all input datasets, intermediate variables, derived spectral indices, standardized anomaly metrics, and diagnostic outputs utilized in the **Crop Stress Intelligence System — Kenya**.

---

## 1. Primary Ingestion Modalities (Live Verified)

### 1.1. Sentinel-2 Level-2A (Optical Surface Reflectance & SCL)

| Variable / Band | Description | Native Spatial Resolution | Temporal Frequency | Native Dtype & Range | Units | Live Verified Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `B02` | Blue (490 nm) | 10 m | 5 days | `uint16` `[0, 10000]` | Scaled Integer (scale: $10^{-4}$) | DE Africa STAC (`s2_l2a`) |
| `B03` | Green (560 nm) | 10 m | 5 days | `uint16` `[0, 10000]` | Scaled Integer (scale: $10^{-4}$) | DE Africa STAC (`s2_l2a`) |
| `B04` | Red (665 nm) | 10 m | 5 days | `uint16` `[0, 10000]` | Scaled Integer (scale: $10^{-4}$) | DE Africa STAC (`s2_l2a`) |
| `B05` | Red Edge 1 (705 nm) | 20 m | 5 days | `uint16` `[0, 10000]` | Scaled Integer (scale: $10^{-4}$) | DE Africa STAC (`s2_l2a`) |
| `B06` | Red Edge 2 (740 nm) | 20 m | 5 days | `uint16` `[0, 10000]` | Scaled Integer (scale: $10^{-4}$) | DE Africa STAC (`s2_l2a`) |
| `B07` | Red Edge 3 (783 nm) | 20 m | 5 days | `uint16` `[0, 10000]` | Scaled Integer (scale: $10^{-4}$) | DE Africa STAC (`s2_l2a`) |
| `B08` | NIR Broad (842 nm) | 10 m | 5 days | `uint16` `[0, 10000]` | Scaled Integer (scale: $10^{-4}$) | DE Africa STAC (`s2_l2a`) |
| `B11` | SWIR 1 (1610 nm) | 20 m | 5 days | `uint16` `[0, 10000]` | Scaled Integer (scale: $10^{-4}$) | DE Africa STAC (`s2_l2a`) |
| `B12` | SWIR 2 (2190 nm) | 20 m | 5 days | `uint16` `[0, 10000]` | Scaled Integer (scale: $10^{-4}$) | DE Africa STAC (`s2_l2a`) |
| `SCL` | Scene Classification Layer | 20 m | 5 days | `uint8` `[0–11]` | Class Enum | DE Africa STAC (`s2_l2a`) |

*Memory Integrity Note*: Surface reflectance values are preserved as native `uint16` integers and `uint8` for SCL during ingestion. Conversion to float occurs lazily only during index/anomaly calculation.

---

### 1.2. Scene Classification Layer (SCL) Quality Masking Classes

| SCL Value | Class Name | Masking Status in Preprocessing | Description / Action |
| :---: | :--- | :---: | :--- |
| `0` | NO_DATA | **Invalid (Masked to NaN)** | Missing / fill sensor observations |
| `1` | SATURATED_OR_DEFECTIVE | **Invalid (Masked to NaN)** | Detector saturation or artifact |
| `2` | CAST_SHADOW | **Invalid (Masked to NaN)** | Topographic or cloud cast shadow |
| `3` | CLOUD_SHADOWS | **Invalid (Masked to NaN)** | Cloud shadow footprint |
| `4` | VEGETATION | **Valid (Preserved)** | Green agricultural canopy & natural vegetation |
| `5` | NOT_VEGETATED | **Valid (Preserved)** | Bare agricultural soil & dry ground |
| `6` | WATER | **Valid (Preserved)** | Inland open water & reservoirs |
| `7` | UNCLASSIFIED | **Invalid (Masked to NaN)** | Ambiguous spectral signature |
| `8` | CLOUD_MEDIUM_PROBABILITY | **Invalid (Masked to NaN)** | Medium-confidence optical cloud |
| `9` | CLOUD_HIGH_PROBABILITY | **Invalid (Masked to NaN)** | High-confidence opaque optical cloud |
| `10` | THIN_CIRRUS | **Invalid (Masked to NaN)** | High-altitude ice-crystal cirrus |
| `11` | SNOW_OR_ICE | **Invalid (Masked to NaN)** | High-reflectance snow or ice |

---

### 1.3. CHIRPS (Precipitation — Monthly & Daily Modes)

| Variable | Description | Native Spatial Resolution | Temporal Frequency | Data Type / Range | Units | Live Verified Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rainfall` (Monthly) | Monthly Total Precipitation | ~5.5 km (0.05°) | Monthly | `float32` `[0.0, 1000.0]` | `mm/month` | DE Africa STAC (`rainfall_chirps_monthly`) |
| `rainfall` (Daily) | Daily Total Precipitation | ~5.5 km (0.05°) | Daily | `float32` `[0.0, 500.0]` | `mm/day` | DE Africa STAC (`rainfall_chirps_daily`) |

---

### 1.4. GRAFS (Global Root-zone moisture Analysis & Forecasting System)

| Variable | Description | Native Spatial Resolution | Temporal Frequency | Data Type / Range | Units | Live Verified Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `s0` | Topsoil Relative Wetness (0–5 cm) | ~10 km (0.1°) | Daily | `float32` `[0.0, 1.0]` | Dimensionless fraction | NCI THREDDS OPeNDAP Server |
| `s1` | Root-zone Soil Water Index (0–1 m) | ~10 km (0.1°) | Daily | `float32` `[0.0, 1.0]` | Dimensionless fraction | NCI THREDDS OPeNDAP Server |

*Scientific Qualification*: Satellite data assimilation product (SMAP+GPM via 4DVAR), **NOT** direct in-situ root-zone probe measurement.

---

### 1.5. Cropland Mask

#### Canonical Product (v1 Primary)
- **Product Name**: Digital Earth Africa Cropland Extent Map (20 m)
- **Provider**: Digital Earth Africa
- **Collection ID**: `crop_mask_eastern`
- **Band Name**: `mask`
- **Native Resolution**: 20 m (`EPSG:32736`)
- **Dtype & Values**: `uint8` (`0` = non-crop, `1` = cropland, `NaN` = nodata)
- **Temporal Reference**: Annual / Static Baseline (2019)
- **Scientific Limitation**: Identifies general cultivated arable parcels. Does **NOT** directly classify mono-crop maize without phenological time-series verification.
- **Resolution Principle**: Target-grid alignment to 10 m features (or 20 m SWIR features) during masking does **NOT** alter the underlying 20 m native resolution of this product.

#### Fallback Product (Defined Separately)
- **Product Name**: ESA WorldCover 10 m
- **Provider**: European Space Agency
- **Collection ID**: `esa_worldcover_2021`
- **Band Name**: `map` (Class `40` = Cropland)

---

## 2. Derived Biophysical Indices (Milestone 3)

| Feature / Index | Full Name | Input Bands & Scale | Derived Spatial Scale | Formulation | Units / Range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ndvi` | Normalized Difference Vegetation Index | B04 (10 m), B08 (10 m) | 10 m native | $\frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + 10^{-6}}$ | Dimensionless `[-1.0, 1.0]` |
| `evi` | Enhanced Vegetation Index | B02 (10 m), B04 (10 m), B08 (10 m) | 10 m native | $2.5 \cdot \frac{\text{NIR} - \text{Red}}{\text{NIR} + 6\text{Red} - 7.5\text{Blue} + 1.0}$ | Dimensionless nominal `[-1.0, 1.0]` |
| `ndmi` | Normalized Difference Moisture Index | B08 (10 m $\to$ 20 m), B11 (20 m) | 20 m derived | $\frac{\text{NIR}_{20\text{m}} - \text{SWIR}_{20\text{m}}}{\text{NIR}_{20\text{m}} + \text{SWIR}_{20\text{m}} + 10^{-6}}$ | Dimensionless `[-1.0, 1.0]` |

---

## 3. Temporal Compositing & Quality Metrics (Milestone 3)

| Variable | Description | Aggregation Method | Units | Formula / Logic |
| :--- | :--- | :--- | :--- | :--- |
| `composite_14d` | 14-day median biophysical composite | 14-day temporal median | Dimensionless | `da.resample(time="14D").median()` |
| `valid_obs_count` | Contributing clear observation tally | 14-day temporal sum | Count | `da.notnull().resample(time="14D").sum()` |
| `smoothed_14d` | Centered rolling window smoothed series | Centered rolling mean (window=3) | Dimensionless | `comp.rolling(time=3, center=True).mean().where(comp.notnull())` |
| `rejection_rate_scl` | Total optical pixel rejection rate | Statistical summary | Percent (%) | $\frac{\text{Total Invalid SCL Pixels}}{\text{Total Input Pixels}} \times 100$ |
| `clear_pixel_pct_per_date` | Scene-level clear sky percentage | Spatial percentage | Percent (%) | $\frac{\text{Valid Unmasked Pixels}}{\text{Spatial Scene Size}} \times 100$ |
| `unfilled_bins_pct` | Persistent data gap frequency | Statistical summary | Percent (%) | $\frac{\text{NaN Composite Cells}}{\text{Total Spatiotemporal Cells}} \times 100$ |

---

## 4. Multi-Resolution Alignment Schema

| Modality | Native Spatial Scale | Native Temporal Scale | Native Coordinate System | Analysis Role |
| :--- | :--- | :--- | :--- | :--- |
| **Sentinel-2 L2A** | 10–20 m | 5 days | `EPSG:32736` (UTM 36S) | Field-scale canopy structure & vigor |
| **CHIRPS** | 0.05° (~5.5 km) | Monthly / Daily | `EPSG:4326` | Regional meteorological drought forcing |
| **GRAFS** | 0.1° (~10 km) | Daily | `EPSG:4326` | Regional root-zone soil water index |
| **Crop Mask** | 20 m | Static / Annual | `EPSG:32736` | Spatial masking of cultivated parcel bounds |

---

## 5. Standardized Anomaly & Hydrological Dynamics Metrics (Milestone 4)

| Metric / Variable | Full Name | Grid Scale & CRS | Formulation / Logic | Units / Range | Zero-Variance Policy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `z_rainfall` ($Z_R$) | Standardized Precipitation Anomaly | ~5.5 km (0.05°) `EPSG:4326` | $Z_R = \frac{R_{x, m} - \mu_{R, m}}{\sigma_{R, m}}$ | Continuous Dimensionless | Masked to `NaN` if $\sigma \le 10^{-4}$ |
| `z_ndvi` ($Z_{\text{NDVI}}$) | Standardized Vegetation Anomaly | 30 m `EPSG:6933` | $Z_{\text{NDVI}} = \frac{\text{NDVI}_{\text{aligned}, m} - \mu_{\text{NDVI}, m}}{\sigma_{\text{NDVI}, m}}$ | Continuous Dimensionless | Masked to `NaN` if $\sigma \le 10^{-4}$ or $\text{count} < 20$ |
| `delta_swi_14d` ($\Delta\text{SWI}_{14d}$) | 14-day Root-Zone Moisture Change | ~10 km (0.1°) `EPSG:4326` | $\Delta\text{SWI}_{14d}(t) = s_1(t) - s_1(t-1)$ | Dimensionless fraction `[-1.0, 1.0]` | $t=0$ evaluated as `NaN` |
| `valid_std_mask` | Valid Baseline Variance Mask | Modality-specific grid | $\sigma > \text{std\_epsilon}$ | Boolean | Excluded pixels tallied and audited |
