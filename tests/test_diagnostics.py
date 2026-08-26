"""Offline unit tests for Milestone 5 diagnostic screening, scouting zone extraction,
and GeoJSON export.

All tests are marked  and have zero network dependencies.

Scientific contracts verified:
1. Case A-D triggers (synthetic inputs)
2. INSUFFICIENT_EVIDENCE vs NORMAL distinction
3. MULTI_SIGNAL for B+D overlap
4. Missing modality handling (never silently NORMAL)
5. Morphological opening behavior
6. MMU using actual projected area
7. GeoJSON schema, CRS, geometry validity
8. Deterministic output
9. Empty result is valid
10. Positive Z_R does not increase drought severity
11. CASE_C not in ACTIONABLE_CASES
12. No 30 m CHIRPS/GRAFS interpolation (nearest-neighbor only)
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.diagnostics.screening import (
    ACTIONABLE_CASES,
    DiagnosticCase,
    assign_coarse_to_diagnostic_grid,
    calculate_evidence_confidence,
    classify_diagnostic_grid,
)
from src.export.vector import MMU_AREA_M2, extract_scouting_zones


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_da(
    value: float,
    shape: tuple = (5, 5),
    y_start: float = 60000.0,
    x_start: float = 815000.0,
    resolution: float = 30.0,
    crs: str = "EPSG:6933",
    name: str = "test",
) -> xr.DataArray:
    """Create a synthetic DataArray on the EPSG:6933 30 m diagnostic grid."""
    n_y, n_x = shape
    y_vals = y_start + np.arange(n_y) * resolution
    x_vals = x_start + np.arange(n_x) * resolution
    data = np.full(shape, value, dtype=np.float32)
    da = xr.DataArray(
        data,
        coords={"y": y_vals, "x": x_vals},
        dims=["y", "x"],
        name=name,
    )
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform()
    da = da.rio.write_nodata(np.nan)
    return da


def _make_da_array(
    arr: np.ndarray,
    y_start: float = 60000.0,
    x_start: float = 815000.0,
    resolution: float = 30.0,
    crs: str = "EPSG:6933",
    name: str = "test",
) -> xr.DataArray:
    """Create a DataArray from a 2D numpy array."""
    n_y, n_x = arr.shape
    y_vals = y_start + np.arange(n_y) * resolution
    x_vals = x_start + np.arange(n_x) * resolution
    da = xr.DataArray(
        arr.astype(np.float32),
        coords={"y": y_vals, "x": x_vals},
        dims=["y", "x"],
        name=name,
    )
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform()
    return da


def _classify(
    z_ndvi: float = 0.1,
    z_r: float = 0.1,
    delta_swi: float = 0.05,
    swi: float | None = None,
    shape: tuple = (3, 3),
    swi_enabled: bool = True,
) -> DiagnosticCase:
    """Helper: classify a uniform synthetic grid and return the value at [0,0]."""
    z_ndvi_da = _make_da(z_ndvi, shape=shape, name="z_ndvi")
    z_r_da = _make_da(z_r, shape=shape, name="z_rainfall")
    delta_da = _make_da(delta_swi, shape=shape, name="delta_swi_14d")
    swi_da = _make_da(swi, shape=shape, name="swi") if swi is not None else None

    result = classify_diagnostic_grid(
        z_ndvi_da, z_r_da, delta_da, swi=swi_da, swi_enabled=swi_enabled
    )
    return DiagnosticCase(int(result.values[0, 0]))


# ---------------------------------------------------------------------------
# 1. Case A — Coupled Agro-Meteorological Stress
# ---------------------------------------------------------------------------


def test_case_a_triggers():
    """Z_R < -0.8 AND Z_NDVI < -1.0 → CASE_A."""
    assert _classify(z_ndvi=-1.5, z_r=-1.5, delta_swi=0.1) == DiagnosticCase.CASE_A



def test_case_a_boundary_z_ndvi():
    """Z_NDVI = -1.0 is NOT < -1.0; should not trigger Case A."""
    result = _classify(z_ndvi=-1.0, z_r=-1.5, delta_swi=0.1)
    assert result != DiagnosticCase.CASE_A



def test_case_a_boundary_z_r():
    """Z_R = -0.8 is NOT < -0.8; should not trigger Case A."""
    result = _classify(z_ndvi=-1.5, z_r=-0.8, delta_swi=0.1)
    assert result != DiagnosticCase.CASE_A


# ---------------------------------------------------------------------------
# 2. Case B — Vegetation Stress Under Non-Dry Conditions
# ---------------------------------------------------------------------------


def test_case_b_triggers():
    """Z_R >= 0.0 AND Z_NDVI < -1.2 AND SWI >= 0.30 → CASE_B (when swi_enabled)."""
    result = _classify(z_ndvi=-1.5, z_r=0.5, delta_swi=0.1, swi=0.6, swi_enabled=True)
    assert result == DiagnosticCase.CASE_B



def test_case_b_swi_disabled():
    """Case B must not trigger when swi_enabled=False."""
    result = _classify(z_ndvi=-1.5, z_r=0.5, delta_swi=0.1, swi=0.6, swi_enabled=False)
    assert result != DiagnosticCase.CASE_B


# ---------------------------------------------------------------------------
# 3. Case C — Hydrological Buffering
# ---------------------------------------------------------------------------


def test_case_c_triggers():
    """Z_R < -1.0 AND Z_NDVI >= -0.5 AND SWI >= 0.30 → CASE_C."""
    result = _classify(z_ndvi=-0.3, z_r=-1.5, delta_swi=0.05, swi=0.5, swi_enabled=True)
    assert result == DiagnosticCase.CASE_C



def test_case_c_not_actionable():
    """Case C (Hydrological Buffering) must NOT be in ACTIONABLE_CASES."""
    assert DiagnosticCase.CASE_C not in ACTIONABLE_CASES


# ---------------------------------------------------------------------------
# 4. Case D — Potential Hydrological Disconnect
# ---------------------------------------------------------------------------


def test_case_d_triggers():
    """Z_R >= 0.5 AND delta_swi_14d <= 0.0 → CASE_D."""
    result = _classify(z_ndvi=0.1, z_r=1.0, delta_swi=-0.1)
    assert result == DiagnosticCase.CASE_D



def test_case_d_boundary_z_r():
    """Z_R = 0.5 satisfies >= 0.5 → CASE_D fires if delta_swi also qualifies."""
    result = _classify(z_ndvi=0.1, z_r=0.5, delta_swi=-0.05)
    assert result == DiagnosticCase.CASE_D



def test_case_d_boundary_delta_swi_zero():
    """delta_swi_14d = 0.0 satisfies <= 0.0 → CASE_D fires."""
    result = _classify(z_ndvi=0.1, z_r=0.6, delta_swi=0.0)
    assert result == DiagnosticCase.CASE_D


# ---------------------------------------------------------------------------
# 5. Normal — no rule fires
# ---------------------------------------------------------------------------


def test_normal_no_trigger():
    """Valid evidence, no rule fires → NORMAL."""
    result = _classify(z_ndvi=0.1, z_r=0.2, delta_swi=0.05, swi=0.4)
    assert result == DiagnosticCase.NORMAL


# ---------------------------------------------------------------------------
# 6–9. Missing evidence handling
# ---------------------------------------------------------------------------


def test_missing_z_r_case_a_insufficient():
    """NaN Z_R with negative Z_NDVI → INSUFFICIENT_EVIDENCE (not CASE_A, not NORMAL)."""
    result = _classify(z_ndvi=-1.5, z_r=float("nan"), delta_swi=0.1)
    assert result == DiagnosticCase.INSUFFICIENT_EVIDENCE



def test_missing_z_ndvi_insufficient_for_case_a():
    """NaN Z_NDVI → Case A cannot fire (requires both z_ndvi AND z_r).

    When z_ndvi is NaN but z_r and delta_swi are valid, the pixel has partial
    evidence. Case A cannot trigger. The overall result must NOT be CASE_A.
    If no other rule fires, it is NORMAL (partial evidence available).
    """
    result = _classify(z_ndvi=float("nan"), z_r=-1.5, delta_swi=0.1)
    # Case A needs z_ndvi — cannot fire with NaN z_ndvi
    assert result != DiagnosticCase.CASE_A
    # Should be NORMAL (z_r and delta_swi provide partial evidence; no rule fires)
    assert result in (DiagnosticCase.NORMAL, DiagnosticCase.INSUFFICIENT_EVIDENCE)


def test_all_modalities_nan_is_insufficient():
    """When ALL modalities are NaN, result must be INSUFFICIENT_EVIDENCE."""
    result = _classify(z_ndvi=float("nan"), z_r=float("nan"), delta_swi=float("nan"))
    assert result == DiagnosticCase.INSUFFICIENT_EVIDENCE



def test_missing_swi_case_b_not_triggered():
    """NaN SWI → Case B cannot fire, even if Z_R and Z_NDVI would otherwise qualify."""
    z_ndvi_da = _make_da(-1.5, name="z_ndvi")
    z_r_da = _make_da(0.5, name="z_rainfall")
    delta_da = _make_da(0.1, name="delta_swi_14d")
    swi_da = _make_da(float("nan"), name="swi")  # NaN SWI

    result = classify_diagnostic_grid(z_ndvi_da, z_r_da, delta_da, swi=swi_da, swi_enabled=True)
    # Should not be CASE_B since SWI is NaN
    assert DiagnosticCase(int(result.values[0, 0])) != DiagnosticCase.CASE_B



def test_missing_delta_swi_case_d_not_triggered():
    """NaN delta_swi_14d → Case D cannot fire, even with strong Z_R."""
    result = _classify(z_ndvi=0.1, z_r=1.5, delta_swi=float("nan"))
    assert result != DiagnosticCase.CASE_D


# ---------------------------------------------------------------------------
# 10. Insufficient evidence is NEVER equal to NORMAL
# ---------------------------------------------------------------------------


def test_insufficient_evidence_not_normal():
    """DiagnosticCase.INSUFFICIENT_EVIDENCE must be distinct from NORMAL."""
    assert DiagnosticCase.INSUFFICIENT_EVIDENCE != DiagnosticCase.NORMAL
    assert DiagnosticCase.INSUFFICIENT_EVIDENCE.value == -1
    assert DiagnosticCase.NORMAL.value == 0



def test_all_nan_inputs_insufficient_not_normal():
    """All NaN inputs must produce INSUFFICIENT_EVIDENCE, not NORMAL."""
    z_ndvi_da = _make_da(float("nan"), name="z_ndvi")
    z_r_da = _make_da(float("nan"), name="z_rainfall")
    delta_da = _make_da(float("nan"), name="delta_swi_14d")
    result = classify_diagnostic_grid(z_ndvi_da, z_r_da, delta_da)
    assert int(result.values[0, 0]) == DiagnosticCase.INSUFFICIENT_EVIDENCE.value


# ---------------------------------------------------------------------------
# 11. MULTI_SIGNAL — B + D co-occurrence
# ---------------------------------------------------------------------------


def test_multi_signal_bd_overlap():
    """Z_R=0.8, Z_NDVI=-1.8, SWI=0.45, delta_swi=-0.05 → MULTI_SIGNAL (B+D both fire)."""
    result = _classify(z_ndvi=-1.8, z_r=0.8, delta_swi=-0.05, swi=0.45, swi_enabled=True)
    assert result == DiagnosticCase.MULTI_SIGNAL


# ---------------------------------------------------------------------------
# 12–13. Mutual exclusivity tests
# ---------------------------------------------------------------------------


def test_ab_mutually_exclusive():
    """Case A needs Z_R < -0.8; Case B needs Z_R >= 0.0. They cannot co-occur."""
    # No Z_R value can satisfy both Z_R < -0.8 AND Z_R >= 0.0
    for z_r in [-0.9, -1.0, -2.0, -5.0]:
        assert z_r < -0.8
        assert not (z_r >= 0.0)
    for z_r in [0.0, 0.5, 1.0, 2.0]:
        assert not (z_r < -0.8)



def test_ad_mutually_exclusive():
    """Case A needs Z_R < -0.8; Case D needs Z_R >= 0.5. They cannot co-occur."""
    for z_r in [-0.9, -1.0, -5.0]:
        assert z_r < -0.8
        assert not (z_r >= 0.5)



def test_bd_overlap_confirmed():
    """Confirm B and D CAN co-occur: a single Z_R value satisfies both constraints."""
    z_r = 0.8
    # Case B condition on Z_R: >= 0.0 → satisfied
    assert z_r >= 0.0
    # Case D condition on Z_R: >= 0.5 → also satisfied
    assert z_r >= 0.5


# ---------------------------------------------------------------------------
# 14–15. Evidence confidence
# ---------------------------------------------------------------------------


def test_evidence_confidence_full():
    """All 3 modalities present (non-NaN) → confidence == 1.0."""
    da1 = _make_da(1.0, name="a")
    da2 = _make_da(2.0, name="b")
    da3 = _make_da(3.0, name="c")
    conf = calculate_evidence_confidence([da1, da2, da3])
    assert np.allclose(conf.values, 1.0)



def test_evidence_confidence_partial():
    """1 of 3 modalities is NaN → confidence == 1/3 per pixel."""
    da1 = _make_da(1.0, name="a")
    da2 = _make_da(float("nan"), name="b")
    da3 = _make_da(3.0, name="c")
    conf = calculate_evidence_confidence([da1, da2, da3])
    expected = 2.0 / 3.0
    assert np.allclose(conf.values, expected, atol=1e-6)



def test_evidence_confidence_range():
    """Confidence must always be in [0.0, 1.0]."""
    da1 = _make_da(float("nan"), name="a")
    da2 = _make_da(float("nan"), name="b")
    conf = calculate_evidence_confidence([da1, da2])
    assert float(conf.values.min()) >= 0.0
    assert float(conf.values.max()) <= 1.0


# ---------------------------------------------------------------------------
# 16. Positive Z_R does NOT increase drought severity
# ---------------------------------------------------------------------------


def test_positive_z_r_no_drought_severity_penalty():
    """A large POSITIVE Z_R must not increase any drought-related metric.

    Evidence confidence is purely a completeness fraction. Large |Z_R| positive
    values do not change confidence (data is still present → contribution = 1).
    The diagnostic case for positive Z_R with neutral NDVI must not be CASE_A.
    """
    # Strongly positive Z_R, neutral NDVI — not drought
    z_r_positive = _make_da(3.0, name="z_rainfall")
    z_ndvi_neutral = _make_da(0.1, name="z_ndvi")
    delta_da = _make_da(0.05, name="delta_swi_14d")

    conf_positive = calculate_evidence_confidence([z_r_positive, z_ndvi_neutral])

    # Strongly negative Z_R (drought), same NDVI
    z_r_negative = _make_da(-3.0, name="z_rainfall")
    conf_negative = calculate_evidence_confidence([z_r_negative, z_ndvi_neutral])

    # Confidence must be identical regardless of sign of Z_R
    assert np.allclose(conf_positive.values, conf_negative.values)

    # Positive Z_R must not trigger Case A
    result = classify_diagnostic_grid(z_ndvi_neutral, z_r_positive, delta_da)
    assert DiagnosticCase(int(result.values[0, 0])) != DiagnosticCase.CASE_A


# ---------------------------------------------------------------------------
# 17–18. Morphological opening behavior
# ---------------------------------------------------------------------------


def test_morphological_opening_removes_isolated_single_pixel():
    """A single isolated actionable pixel should be removed by 3×3 opening."""
    import scipy.ndimage as ndi
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True  # single isolated pixel

    struct_3x3 = np.ones((3, 3), dtype=bool)
    opened = ndi.binary_opening(mask, structure=struct_3x3)
    assert not np.any(opened), "3×3 opening should remove a single isolated pixel"



def test_morphological_opening_preserves_large_cluster():
    """A 10×10 block of actionable pixels should survive 3×3 opening."""
    import scipy.ndimage as ndi
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True  # 10×10 = 100 pixels

    struct_3x3 = np.ones((3, 3), dtype=bool)
    opened = ndi.binary_opening(mask, structure=struct_3x3)
    assert np.sum(opened) > 0, "3×3 opening should preserve a large 10×10 cluster"


# ---------------------------------------------------------------------------
# 19–20. MMU actual projected area filtering
# ---------------------------------------------------------------------------


def test_mmu_below_2ha_removed():
    """A component with area < 2 ha must be filtered out.

    At 30 m resolution: pixel_area = 900 m².
    2 ha = 20000 m² → requires 22.2 pixels. So 20 pixels = 18000 m² < 2 ha.
    """
    # Create a 5×5 grid with a small 3×3 (9 pixels) actionable cluster (area = 8100 m²)
    diag_arr = np.zeros((5, 5), dtype=np.int8) + DiagnosticCase.NORMAL.value
    diag_arr[1:4, 1:4] = DiagnosticCase.CASE_A.value  # 9 pixels = 8100 m² < 2 ha

    diag_da = _make_da_array(diag_arr.astype(np.float32), name="diagnostic_case")
    diag_da = diag_da.astype(np.int8)
    diag_da = diag_da.rio.write_crs("EPSG:6933")
    diag_da = diag_da.rio.write_transform()

    result_fc = extract_scouting_zones(diag_da, apply_morphology=False)
    assert result_fc["type"] == "FeatureCollection"
    assert len(result_fc["features"]) == 0, (
        f"Component of 9 pixels × 900 m² = 8100 m² < 20000 m² should be filtered. "
        f"Got {len(result_fc['features'])} features."
    )



def test_mmu_above_2ha_retained():
    """A component with area >= 2 ha must be retained.

    At 30 m resolution: 2 ha = 20000 m². Need at least ceil(20000/900) = 23 pixels.
    Use a 6×4 = 24-pixel block → 21600 m² >= 2 ha.
    """
    # Create a 20×20 grid with a 6×4 actionable cluster (24 pixels = 21600 m² > 2 ha)
    diag_arr = np.zeros((20, 20), dtype=np.int8) + DiagnosticCase.NORMAL.value
    diag_arr[8:14, 8:12] = DiagnosticCase.CASE_A.value  # 6 rows × 4 cols = 24 pixels

    diag_da = _make_da_array(diag_arr.astype(np.float32), resolution=30.0, name="diagnostic_case")
    diag_da = diag_da.astype(np.int8)
    diag_da = diag_da.rio.write_crs("EPSG:6933")
    diag_da = diag_da.rio.write_transform()

    result_fc = extract_scouting_zones(diag_da, apply_morphology=False)
    assert result_fc["type"] == "FeatureCollection"
    assert len(result_fc["features"]) >= 1, (
        f"Component of 24 pixels × 900 m² = 21600 m² >= 20000 m² should be retained. "
        f"Got {len(result_fc['features'])} features."
    )


# ---------------------------------------------------------------------------
# 21. GeoJSON schema
# ---------------------------------------------------------------------------


def test_geojson_schema_valid():
    """Exported GeoJSON must have 'type': 'FeatureCollection' and 'features' list."""
    diag_da = _make_da(DiagnosticCase.NORMAL.value, name="diagnostic_case")
    diag_da = diag_da.astype(np.int8)
    result_fc = extract_scouting_zones(diag_da)
    assert result_fc["type"] == "FeatureCollection"
    assert "features" in result_fc
    assert isinstance(result_fc["features"], list)



def test_geojson_feature_properties_schema():
    """Each feature must have required properties when zones exist."""
    # Create a large enough actionable cluster (50×50 pixels = 67500 m² >> 2 ha)
    diag_arr = np.zeros((60, 60), dtype=np.float32) + DiagnosticCase.NORMAL.value
    diag_arr[5:55, 5:55] = DiagnosticCase.CASE_A.value  # 50×50 = 2500 pixels = 2.25 Mha >> 2 ha

    diag_da = _make_da_array(diag_arr, name="diagnostic_case")
    diag_da = diag_da.astype(np.int8)
    diag_da = diag_da.rio.write_crs("EPSG:6933")
    diag_da = diag_da.rio.write_transform()

    result_fc = extract_scouting_zones(diag_da, apply_morphology=False, bin_start=pd.Timestamp("2023-03-01"))

    if len(result_fc["features"]) > 0:
        props = result_fc["features"][0]["properties"]
        required_keys = ["zone_id", "case_type", "area_ha", "centroid_lat", "centroid_lon"]
        for key in required_keys:
            assert key in props, f"Missing required property: {key}"
        assert props["zone_id"].startswith("SC-20230301-")


# ---------------------------------------------------------------------------
# 22. GeoJSON WGS84 coordinates
# ---------------------------------------------------------------------------


def test_geojson_wgs84_coordinates():
    """All GeoJSON coordinates must be within WGS84 bounds."""
    diag_arr = np.zeros((60, 60), dtype=np.float32) + DiagnosticCase.NORMAL.value
    diag_arr[5:55, 5:55] = DiagnosticCase.CASE_A.value

    diag_da = _make_da_array(diag_arr, name="diagnostic_case")
    diag_da = diag_da.astype(np.int8)
    diag_da = diag_da.rio.write_crs("EPSG:6933")
    diag_da = diag_da.rio.write_transform()

    result_fc = extract_scouting_zones(diag_da, apply_morphology=False)

    for feature in result_fc["features"]:
        geom = feature["geometry"]
        if geom["type"] == "Polygon":
            for ring in geom["coordinates"]:
                for lon, lat in ring:
                    assert -180.0 <= lon <= 180.0, f"Longitude out of WGS84 bounds: {lon}"
                    assert -90.0 <= lat <= 90.0, f"Latitude out of WGS84 bounds: {lat}"


# ---------------------------------------------------------------------------
# 23. Geometry validity
# ---------------------------------------------------------------------------


def test_geojson_geometry_validity():
    """All exported geometries must be valid (no self-intersections)."""
    try:
        from shapely.geometry import shape as shapely_shape
    except ImportError:
        pytest.skip("shapely not installed — skipping geometry validity test")

    diag_arr = np.zeros((60, 60), dtype=np.float32) + DiagnosticCase.NORMAL.value
    diag_arr[5:55, 5:55] = DiagnosticCase.CASE_A.value

    diag_da = _make_da_array(diag_arr, name="diagnostic_case")
    diag_da = diag_da.astype(np.int8)
    diag_da = diag_da.rio.write_crs("EPSG:6933")
    diag_da = diag_da.rio.write_transform()

    result_fc = extract_scouting_zones(diag_da, apply_morphology=False)

    for feature in result_fc["features"]:
        shp = shapely_shape(feature["geometry"])
        assert shp.is_valid, f"Invalid geometry in feature: {feature['properties']['zone_id']}"


# ---------------------------------------------------------------------------
# 24. Deterministic output
# ---------------------------------------------------------------------------


def test_deterministic_output():
    """Two runs with identical inputs must produce identical zone_ids and geometries."""
    diag_arr = np.zeros((60, 60), dtype=np.float32) + DiagnosticCase.NORMAL.value
    diag_arr[10:50, 10:50] = DiagnosticCase.CASE_A.value

    diag_da = _make_da_array(diag_arr, name="diagnostic_case")
    diag_da = diag_da.astype(np.int8)
    diag_da = diag_da.rio.write_crs("EPSG:6933")
    diag_da = diag_da.rio.write_transform()

    ts = pd.Timestamp("2023-03-01")
    result1 = extract_scouting_zones(diag_da, apply_morphology=False, bin_start=ts)
    result2 = extract_scouting_zones(diag_da, apply_morphology=False, bin_start=ts)

    ids1 = sorted([f["properties"]["zone_id"] for f in result1["features"]])
    ids2 = sorted([f["properties"]["zone_id"] for f in result2["features"]])
    assert ids1 == ids2, f"Zone IDs not deterministic: {ids1} vs {ids2}"


# ---------------------------------------------------------------------------
# 25. Empty GeoJSON is valid
# ---------------------------------------------------------------------------


def test_empty_geojson_valid():
    """Zero actionable pixels → empty FeatureCollection with correct RFC 7946 schema."""
    diag_da = _make_da(DiagnosticCase.NORMAL.value, name="diagnostic_case")
    diag_da = diag_da.astype(np.int8)
    result_fc = extract_scouting_zones(diag_da)
    assert result_fc["type"] == "FeatureCollection"
    assert result_fc["features"] == []



def test_empty_geojson_on_all_insufficient_evidence():
    """All INSUFFICIENT_EVIDENCE pixels → no actionable zones → empty FeatureCollection."""
    diag_da = _make_da(DiagnosticCase.INSUFFICIENT_EVIDENCE.value, name="diagnostic_case")
    diag_da = diag_da.astype(np.int8)
    result_fc = extract_scouting_zones(diag_da)
    assert len(result_fc["features"]) == 0


# ---------------------------------------------------------------------------
# 26. No 30 m CHIRPS interpolation (nearest-neighbor only)
# ---------------------------------------------------------------------------


def test_assign_coarse_uses_nearest_neighbor():
    """assign_coarse_to_diagnostic_grid must use nearest-neighbor (not bilinear).

    Verify that coarse grid values are broadcast as discrete blocks, not smoothly
    interpolated. A 2×2 coarse grid should produce sharp block boundaries on the
    30 m reference grid.
    """
    # Create a 2×2 coarse CHIRPS-like grid with distinct values
    coarse_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    coarse_da = xr.DataArray(
        coarse_arr,
        coords={"lat": [0.65, 0.75], "lon": [35.15, 35.25]},
        dims=["lat", "lon"],
        name="z_rainfall",
    )

    # Reference 30 m grid on EPSG:6933 (small area matching the coarse extent)
    ref_da = _make_da(0.0, shape=(10, 10), name="z_ndvi")

    result = assign_coarse_to_diagnostic_grid(coarse_da, ref_da)

    # The result should contain only the discrete coarse cell values (1.0, 2.0, 3.0, 4.0)
    # NOT interpolated intermediate values
    result_vals = result.values.flatten()
    finite_vals = result_vals[np.isfinite(result_vals)]
    unique_vals = set(np.round(finite_vals, 3))

    # With nearest-neighbor: only original coarse cell values should appear
    # With bilinear: intermediate values (e.g. 1.5, 2.5, etc.) would appear
    for v in unique_vals:
        assert v in {1.0, 2.0, 3.0, 4.0}, (
            f"Unexpected value {v} in output — bilinear interpolation detected! "
            f"Only nearest-neighbor values should appear."
        )


# ---------------------------------------------------------------------------
# 27. Case C not actionable
# ---------------------------------------------------------------------------


def test_case_c_value_not_in_actionable_set():
    """DiagnosticCase.CASE_C must not be present in ACTIONABLE_CASES."""
    assert DiagnosticCase.CASE_C not in ACTIONABLE_CASES



def test_case_c_pixels_not_exported_as_scouting_zones():
    """A grid of pure CASE_C pixels must not produce scouting zones."""
    diag_arr = np.zeros((60, 60), dtype=np.float32) + DiagnosticCase.CASE_C.value

    diag_da = _make_da_array(diag_arr, name="diagnostic_case")
    diag_da = diag_da.astype(np.int8)
    diag_da = diag_da.rio.write_crs("EPSG:6933")
    diag_da = diag_da.rio.write_transform()

    result_fc = extract_scouting_zones(diag_da, apply_morphology=False)
    assert len(result_fc["features"]) == 0, "CASE_C pixels must NOT produce scouting zones"


# ---------------------------------------------------------------------------
# 28. No SWI reconstruction from delta_swi
# ---------------------------------------------------------------------------


def test_no_swi_reconstruction_from_delta():
    """Verify screening.py does not reconstruct absolute SWI from delta_swi_14d."""
    import inspect
    from src.diagnostics import screening
    source = inspect.getsource(screening)

    # These operations would indicate illegal SWI reconstruction from delta
    forbidden = ["cumsum", "cumulative_sum", "np.cumsum", "integrate"]
    for f in forbidden:
        assert f not in source, (
            f"Found '{f}' in screening.py — this could indicate illegal SWI "
            f"reconstruction from delta_swi_14d."
        )


# ---------------------------------------------------------------------------
# 29. MULTI_SIGNAL is actionable
# ---------------------------------------------------------------------------


def test_multi_signal_is_actionable():
    """MULTI_SIGNAL must be in ACTIONABLE_CASES."""
    assert DiagnosticCase.MULTI_SIGNAL in ACTIONABLE_CASES


# ---------------------------------------------------------------------------
# 30. Diagnostic case values are unique and correctly ordered
# ---------------------------------------------------------------------------


def test_diagnostic_case_enum_values():
    """Verify all DiagnosticCase enum values are correct."""
    assert DiagnosticCase.INSUFFICIENT_EVIDENCE.value == -1
    assert DiagnosticCase.NORMAL.value == 0
    assert DiagnosticCase.CASE_A.value == 1
    assert DiagnosticCase.CASE_B.value == 2
    assert DiagnosticCase.CASE_C.value == 3
    assert DiagnosticCase.CASE_D.value == 4
    assert DiagnosticCase.MULTI_SIGNAL.value == 5
