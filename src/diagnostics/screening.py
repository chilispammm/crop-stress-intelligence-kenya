"""Multi-Modal Crop Stress Diagnostic Screening Engine (Milestone 5).

Transforms validated multimodal M4 feature products into diagnostic screening
classifications, evidence states, and confidence metrics.

Scientific Interpretation Boundary
------------------------------------
This module implements **screening hypotheses**, not causal inference:

  - Case A: coincident rainfall deficit and vegetation anomaly.
    Does NOT prove drought causality or any specific stress agent.
  - Case B: vegetation anomaly under apparently non-dry hydro-meteorological
    conditions (Z_R ≥ 0.0, SWI ≥ 0.30).
    Does NOT prove pest, pathogen, nutrient deficiency, biotic stress,
    waterlogging, delayed planting, or any other causal mechanism.
    Possible explanations require independent field validation.
  - Case C: absence of vegetation anomaly despite rainfall deficit
    coincident with adequate root-zone moisture — consistent with
    hydrological buffering. Does NOT prove any specific mechanism.
  - Case D: rainfall-to-root-zone moisture disconnect (non-increasing
    SWI despite above-average rainfall). Does NOT prove runoff,
    soil crusting, infiltration failure, or any topographic effect.

Correct interpretation of all cases:
  "Candidate multi-modal screening evidence requiring field validation."

Resolution Contract
--------------------
- Z_NDVI operates on the authoritative 30 m EPSG:6933 diagnostic grid.
- Z_R (CHIRPS, ~5.5 km) and delta_swi_14d (GRAFS, ~10 km) are assigned to the
  30 m diagnostic grid using nearest-neighbor lookup ONLY.
- Each 30 m diagnostic pixel receives the value of its containing coarse
  source cell. Repeated values across multiple 30 m pixels represent SHARED
  contextual evidence, NOT independent 30 m environmental measurements.
- No bilinear or other spatial interpolation of CHIRPS or GRAFS is performed.
- Source-cell provenance is documented in `assign_coarse_to_diagnostic_grid`
  output attributes.

Rule Overlap Analysis (exhaustive)
------------------------------------
Case A:  Z_R < -0.8  AND  Z_NDVI < -1.0
Case B:  Z_R >= 0.0  AND  Z_NDVI < -1.2  AND  SWI >= 0.30
Case C:  Z_R < -1.0  AND  Z_NDVI >= -0.5  AND  SWI >= 0.30
Case D:  Z_R >= 0.5  AND  delta_swi_14d <= 0.0

Pair analysis:
  A ∩ B: A needs Z_R < -0.8; B needs Z_R >= 0.0  → IMPOSSIBLE (mutually exclusive)
  A ∩ C: A needs Z_NDVI < -1.0; C needs Z_NDVI >= -0.5  → IMPOSSIBLE
  A ∩ D: A needs Z_R < -0.8; D needs Z_R >= 0.5  → IMPOSSIBLE
  B ∩ C: B needs Z_R >= 0.0; C needs Z_R < -1.0  → IMPOSSIBLE
  C ∩ D: C needs Z_R < -1.0; D needs Z_R >= 0.5  → IMPOSSIBLE
  B ∩ D: B needs Z_R >= 0.0, D needs Z_R >= 0.5 → POSSIBLE simultaneously.
         Example: Z_R=0.8, Z_NDVI=-1.8, SWI=0.45, delta_swi=-0.05
         → Case B fires AND Case D fires → MULTI_SIGNAL required.

Policy: simultaneous B+D signals are scientifically meaningful (vegetation
anomaly under apparently non-dry conditions coinciding with non-increasing
root-zone moisture). Hiding this via if/elif ordering would be dishonest.
MULTI_SIGNAL is the correct representation.

Evidence Contract
------------------
  INSUFFICIENT_EVIDENCE (-1): Required evidence for relevant rules is unavailable
                               (NaN inputs). This is explicitly distinct from NORMAL.
                               Missing evidence can NEVER silently become NORMAL.
  NORMAL (0):                  At least one evidence pathway has valid data AND no
                               rule fires.
  A pixel needs evidence from at least one evaluable pathway (z_r+z_ndvi OR
  z_r+delta_swi) to qualify as NORMAL; otherwise it is INSUFFICIENT_EVIDENCE.

SWI Threshold Justification and Limitations
---------------------------------------------
GRAFS s1 is documented in DATA_DICTIONARY.md as a dimensionless fraction [0.0, 1.0]
representing the root-zone (0–100 cm) Soil Water Index.

SWI >= 0.30 is operationally interpretable because the source variable has a
known and documented [0,1] scale.

However, this threshold is:
  - an initial M5 screening assumption ONLY;
  - NOT an empirically validated agronomic adequacy threshold;
  - NOT calibrated to any specific East African soil-texture profile;
  - NOT derived from in-situ field capacity or wilting-point measurements.

Its use is analogous to applying a conservative prior: it screens for pixels
where root-zone moisture is at or above 30% of the documented scale.
All threshold-based rules require future field validation before operational use.

NDMI Exclusion
---------------
The 20 m EPSG:32736 NDMI product (from src/features/indices.py) is intentionally
excluded from M5 screening because no M5-approved cross-grid alignment contract
currently exists to bring NDMI onto the 30 m EPSG:6933 diagnostic grid.
Consuming NDMI without explicit alignment would violate the resolution-honest
contract. NDMI must NOT be added in this verification pass.
"""

from __future__ import annotations

from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr


class DiagnosticCase(IntEnum):
    """Enumeration of M5 diagnostic screening states.

    IMPORTANT: INSUFFICIENT_EVIDENCE must be -1 (not 0) so that missing data
    can NEVER accidentally map to NORMAL via default integer initialization.
    """
    INSUFFICIENT_EVIDENCE = -1
    NORMAL = 0
    CASE_A = 1    # Coupled Agro-Meteorological Stress
    CASE_B = 2    # Vegetation Stress Under Non-Dry Conditions
    CASE_C = 3    # Hydrological Buffering (protective, not automatically actionable)
    CASE_D = 4    # Potential Hydrological Disconnect / Recharge Lag
    MULTI_SIGNAL = 5  # Multiple rules simultaneously satisfied (B + D)


#: Cases that represent actionable candidate scouting states.
#: Case C (Hydrological Buffering) is a protective state, NOT automatically actionable.
ACTIONABLE_CASES: frozenset = frozenset([
    DiagnosticCase.CASE_A,
    DiagnosticCase.CASE_B,
    DiagnosticCase.CASE_D,
    DiagnosticCase.MULTI_SIGNAL,
])


def assign_coarse_to_diagnostic_grid(
    coarse_da: xr.DataArray,
    reference_da: xr.DataArray,
) -> xr.DataArray:
    """Assign coarse-resolution contextual evidence to the 30 m diagnostic grid.

    This function performs **nearest-neighbor lookup** of coarse source-cell values
    onto the 30 m EPSG:6933 diagnostic grid.  It does NOT perform bilinear or any
    other form of spatial interpolation.

    Multiple 30 m diagnostic pixels that fall within the same coarse source cell
    receive identical values, representing **shared coarse contextual evidence**,
    not independent 30 m measurements.

    Parameters
    ----------
    coarse_da : xr.DataArray
        Coarse-resolution DataArray in EPSG:4326 (e.g. CHIRPS ~5.5 km, GRAFS ~10 km).
        Must have 'lat' and 'lon' coordinates.
    reference_da : xr.DataArray
        Reference DataArray on the 30 m EPSG:6933 diagnostic grid.
        Must have 'y' and 'x' (or 'lat'/'lon') coordinates.
        Used only to extract destination coordinate arrays.

    Returns
    -------
    xr.DataArray
        Coarse values broadcast onto the reference_da grid using nearest-neighbor
        lookup. Spatial dimensions and coordinates match reference_da.

    Notes
    -----
    Coarse source-cell identity is documented in output attributes. The result
    must never be interpreted as spatially continuous 30 m information.
    """
    # Determine coordinate names on reference grid
    y_name = "y" if "y" in reference_da.coords else "lat"
    x_name = "x" if "x" in reference_da.coords else "lon"

    dst_y = reference_da[y_name].values
    dst_x = reference_da[x_name].values

    # Reference grid is in EPSG:6933 (projected), but coarse_da is in EPSG:4326.
    # We need to reproject the destination coordinates to lat/lon for the lookup.
    try:
        from pyproj import Transformer
        if reference_da.rio.crs is not None:
            src_crs_str = str(reference_da.rio.crs)
        else:
            src_crs_str = "EPSG:6933"

        transformer = Transformer.from_crs(src_crs_str, "EPSG:4326", always_xy=True)
        # dst_x, dst_y are projected coordinates; transform to lon, lat
        dst_lon, dst_lat = transformer.transform(dst_x, dst_y)
    except Exception:
        # If CRS info unavailable, fall back to treating coords as lat/lon directly
        dst_lat = dst_y
        dst_lon = dst_x

    # Nearest-neighbor lookup: for each destination (lat, lon), find the closest
    # coarse source cell. This is a broadcast/repeat of coarse cell values.
    coarse_lat = coarse_da["lat"].values if "lat" in coarse_da.coords else coarse_da.coords[coarse_da.dims[-2]].values
    coarse_lon = coarse_da["lon"].values if "lon" in coarse_da.coords else coarse_da.coords[coarse_da.dims[-1]].values

    # Vectorized nearest-neighbor assignment using numpy
    # For each dst lat/lon, find the index of the nearest coarse cell
    lat_idx = np.argmin(np.abs(coarse_lat[:, None] - dst_lat[None, :]), axis=0)  # shape: (n_dst_y,)
    lon_idx = np.argmin(np.abs(coarse_lon[:, None] - dst_lon[None, :]), axis=0)  # shape: (n_dst_x,)

    # Broadcast coarse values onto destination grid
    # coarse_da may have time dimension: handle both 2D and 3D
    if "time" in coarse_da.dims:
        n_time = coarse_da.sizes["time"]
        coarse_vals = coarse_da.values  # (time, lat, lon)
        dst_arr = coarse_vals[:, lat_idx[:, None], lon_idx[None, :]]  # (time, n_dst_y, n_dst_x)

        time_coord = coarse_da["time"]
        coords = {"time": time_coord, y_name: reference_da[y_name], x_name: reference_da[x_name]}
        result = xr.DataArray(
            dst_arr.astype(np.float32),
            coords=coords,
            dims=["time", y_name, x_name],
            name=coarse_da.name,
        )
    else:
        coarse_vals = coarse_da.values  # (lat, lon)
        dst_arr = coarse_vals[lat_idx[:, None], lon_idx[None, :]]  # (n_dst_y, n_dst_x)

        coords = {y_name: reference_da[y_name], x_name: reference_da[x_name]}
        result = xr.DataArray(
            dst_arr.astype(np.float32),
            coords=coords,
            dims=[y_name, x_name],
            name=coarse_da.name,
        )

    result.attrs.update(coarse_da.attrs)
    result.attrs["coarse_cell_lookup"] = "nearest-neighbor (NOT spatial interpolation)"
    result.attrs["coarse_source_resolution"] = coarse_da.attrs.get("native_resolution", "coarse")
    result.attrs["coarse_assignment_policy"] = (
        "Each 30 m diagnostic pixel is assigned the value of its containing coarse "
        "source cell. Repeated values represent shared contextual evidence, not "
        "independent 30 m measurements."
    )
    return result


def classify_diagnostic_grid(
    z_ndvi: xr.DataArray,
    z_r: xr.DataArray,
    delta_swi_14d: xr.DataArray,
    swi: Optional[xr.DataArray] = None,
    swi_enabled: bool = True,
) -> xr.DataArray:
    """Classify each 30 m diagnostic pixel into a DiagnosticCase.

    Implements Cases A–D with explicit multi-signal detection for the B+D overlap.
    Missing evidence (NaN) produces INSUFFICIENT_EVIDENCE, never NORMAL.

    Parameters
    ----------
    z_ndvi : xr.DataArray
        Standardized NDVI anomaly on the 30 m EPSG:6933 diagnostic grid.
    z_r : xr.DataArray
        Standardized precipitation anomaly, assigned to the 30 m grid via
        nearest-neighbor coarse-cell lookup (NOT interpolated).
    delta_swi_14d : xr.DataArray
        14-day root-zone SWI change, assigned to the 30 m grid via nearest-neighbor.
    swi : xr.DataArray, optional
        Absolute root-zone SWI (s1, dimensionless [0,1]), assigned to the 30 m grid.
        Required for Cases B and C. If None and swi_enabled=True, Cases B and C
        are skipped (treated as insufficient evidence per pixel).
    swi_enabled : bool, default True
        Whether to evaluate Cases B and C (requires valid SWI/s1 semantics).
        Set to False to disable SWI-dependent cases globally.

    Returns
    -------
    xr.DataArray
        Integer-encoded DataArray (int8) with DiagnosticCase values.
        Dimensions match z_ndvi.

    Notes
    -----
    Rule Overlap:
        Cases A/B, A/C, A/D, B/C, C/D are mutually exclusive by construction.
        Cases B and D can co-occur → MULTI_SIGNAL.

    Evidence Requirements:
        Case A: z_ndvi, z_r (both required)
        Case B: z_ndvi, z_r, swi (all three required)
        Case C: z_ndvi, z_r, swi (all three required)
        Case D: z_r, delta_swi_14d (both required)
    """
    # Extract numpy arrays for vectorized operations
    ndvi_arr = z_ndvi.values.astype(np.float32)
    zr_arr = z_r.values.astype(np.float32)
    delta_arr = delta_swi_14d.values.astype(np.float32)
    swi_arr = swi.values.astype(np.float32) if (swi is not None and swi_enabled) else None

    shape = ndvi_arr.shape

    # Initialize output to INSUFFICIENT_EVIDENCE (-1)
    # This is intentionally -1 so that uninitialized cells never read as NORMAL (0)
    out = np.full(shape, DiagnosticCase.INSUFFICIENT_EVIDENCE, dtype=np.int8)

    # --- Validity masks (True where data is available for evaluation) ---
    ndvi_valid = ~np.isnan(ndvi_arr)
    zr_valid = ~np.isnan(zr_arr)
    delta_valid = ~np.isnan(delta_arr)
    swi_valid = (~np.isnan(swi_arr)) if swi_arr is not None else np.zeros(shape, dtype=bool)

    # --- Case predicates (only where inputs are valid) ---
    # Case A: Z_R < -0.8 AND Z_NDVI < -1.0
    case_a_pred = (zr_arr < -0.8) & (ndvi_arr < -1.0)
    case_a_sufficient = zr_valid & ndvi_valid
    case_a = case_a_pred & case_a_sufficient

    # Case B: Z_R >= 0.0 AND Z_NDVI < -1.2 AND SWI >= 0.30
    if swi_enabled and swi_arr is not None:
        case_b_pred = (zr_arr >= 0.0) & (ndvi_arr < -1.2) & (swi_arr >= 0.30)
        case_b_sufficient = zr_valid & ndvi_valid & swi_valid
        case_b = case_b_pred & case_b_sufficient
    else:
        case_b = np.zeros(shape, dtype=bool)

    # Case C: Z_R < -1.0 AND Z_NDVI >= -0.5 AND SWI >= 0.30
    if swi_enabled and swi_arr is not None:
        case_c_pred = (zr_arr < -1.0) & (ndvi_arr >= -0.5) & (swi_arr >= 0.30)
        case_c_sufficient = zr_valid & ndvi_valid & swi_valid
        case_c = case_c_pred & case_c_sufficient
    else:
        case_c = np.zeros(shape, dtype=bool)

    # Case D: Z_R >= 0.5 AND delta_swi_14d <= 0.0
    case_d_pred = (zr_arr >= 0.5) & (delta_arr <= 0.0)
    case_d_sufficient = zr_valid & delta_valid
    case_d = case_d_pred & case_d_sufficient

    # --- Count simultaneously firing cases per pixel ---
    n_firing = case_a.astype(np.int8) + case_b.astype(np.int8) + case_c.astype(np.int8) + case_d.astype(np.int8)

    # --- Determine pixel-level evidence availability ---
    # A pixel has sufficient evidence if AT LEAST ONE case can be evaluated
    # (even if that case does not fire), OR if any combination of inputs
    # covers the NORMAL state (at minimum z_ndvi and z_r available)
    has_any_evidence = (ndvi_valid & zr_valid) | (zr_valid & delta_valid)

    # --- Assign classifications ---

    # Step 1: Pixels with NO evidence remain INSUFFICIENT_EVIDENCE (already set)
    # Step 2: Pixels with evidence but no firing case → NORMAL
    normal_mask = has_any_evidence & (n_firing == 0)
    out[normal_mask] = DiagnosticCase.NORMAL

    # Step 3: Single-case fires
    out[case_a & (n_firing == 1)] = DiagnosticCase.CASE_A
    out[case_b & (n_firing == 1)] = DiagnosticCase.CASE_B
    out[case_c & (n_firing == 1)] = DiagnosticCase.CASE_C
    out[case_d & (n_firing == 1)] = DiagnosticCase.CASE_D

    # Step 4: Multi-signal (B + D overlap is the only possible overlap)
    out[n_firing >= 2] = DiagnosticCase.MULTI_SIGNAL

    result = xr.DataArray(
        out,
        coords=z_ndvi.coords,
        dims=z_ndvi.dims,
        name="diagnostic_case",
    )
    result.attrs.update({
        "standard_name": "crop_stress_diagnostic_case",
        "long_name": "M5 Crop Stress Diagnostic Screening Case",
        "diagnostic_cases": {
            "INSUFFICIENT_EVIDENCE": -1,
            "NORMAL": 0,
            "CASE_A": 1,
            "CASE_B": 2,
            "CASE_C": 3,
            "CASE_D": 4,
            "MULTI_SIGNAL": 5,
        },
        "case_a_rule": "Z_R < -0.8 AND Z_NDVI < -1.0",
        "case_b_rule": "Z_R >= 0.0 AND Z_NDVI < -1.2 AND SWI >= 0.30",
        "case_c_rule": "Z_R < -1.0 AND Z_NDVI >= -0.5 AND SWI >= 0.30",
        "case_d_rule": "Z_R >= 0.5 AND delta_swi_14d <= 0.0",
        "multi_signal_pairs": "B+D (only possible overlap)",
        "swi_enabled": swi_enabled,
        "scientific_caveat": (
            "These rules are initial screening assumptions requiring future "
            "field validation. Classifications are NOT causal diagnoses."
        ),
        "resolution_contract": (
            "Z_NDVI is at native 30 m EPSG:6933. Z_R and delta_swi_14d are "
            "coarse CHIRPS/GRAFS values assigned via nearest-neighbor lookup. "
            "Repeated coarse values represent shared contextual evidence."
        ),
    })
    return result


def calculate_evidence_confidence(
    modality_arrays: Sequence[xr.DataArray],
) -> xr.DataArray:
    """Compute per-pixel evidence completeness fraction.

    Measures the fraction of provided modalities that are non-NaN at each pixel.
    This is evidence COMPLETENESS, not probability of correct diagnosis.

    Parameters
    ----------
    modality_arrays : sequence of xr.DataArray
        One DataArray per modality (e.g. [z_ndvi, z_r, delta_swi_14d]).
        All must have compatible dimensions/coordinates.

    Returns
    -------
    xr.DataArray
        float32 DataArray in [0.0, 1.0] representing fraction of available modalities.
        Shape matches the broadcast shape of the input arrays.
    """
    if not modality_arrays:
        raise ValueError("modality_arrays must be non-empty.")

    n_total = len(modality_arrays)
    # Count non-NaN per pixel across modalities
    valid_count = sum(~np.isnan(da.values.astype(np.float32)) for da in modality_arrays)

    confidence = valid_count.astype(np.float32) / float(n_total)

    result = xr.DataArray(
        confidence,
        coords=modality_arrays[0].coords,
        dims=modality_arrays[0].dims,
        name="evidence_confidence",
    )
    result.attrs.update({
        "standard_name": "evidence_completeness_fraction",
        "long_name": "Evidence Completeness Fraction (M5)",
        "units": "dimensionless",
        "range": "[0.0, 1.0]",
        "interpretation": (
            "Fraction of provided modalities with valid (non-NaN) observations. "
            "This is evidence completeness, NOT probability of correct diagnosis."
        ),
        "n_modalities_total": n_total,
    })
    return result
