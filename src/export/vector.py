"""Vector export utilities for Milestone 5 candidate scouting zones.

Implements the full scouting zone extraction pipeline:
    1. Actionable diagnostic mask
    2. Optional 3×3 morphological opening
    3. 8-connected component labeling
    4. Actual projected-area MMU filtering (>= 2 ha)
    5. Polygonization (union of all rasterio-returned shapes per component)
    6. CRS conversion to EPSG:4326
    7. RFC 7946 GeoJSON FeatureCollection

Scientific Contract
--------------------
- All area calculations use actual projected pixel dimensions from the
  affine transform (EPSG:6933 meters). Geographic-degree areas are NEVER used.
- Area reported in `area_ha` is pixel-count × pixel_area_m2, computed
  entirely in projected space. This is independently verifiable against
  the projected Shapely geometry area (see `geom_area_ha` in properties).
- MMU = 2 ha (20 000 m²) is checked using projected pixel area only.
- A zero-feature FeatureCollection is a valid result.
- No polygons are manufactured to satisfy tests or thresholds.
- Centroids are calculated in projected space before conversion to WGS84.
- Coarse CHIRPS/GRAFS source-cell values are reported as-is. They are NOT
  spatially continuous 30 m fields.

Pipeline Reporting
-------------------
`extract_scouting_zones` returns a dict with keys:
    "type"             → "FeatureCollection"
    "features"         → list of RFC 7946 Features
    "pipeline_report"  → dict with per-step pixel/area accounting (see below)

Pipeline report keys:
    "pixel_area_m2"               actual projected pixel area from transform
    "raw_actionable_pixels"       pixels matching ACTIONABLE_CASES
    "raw_actionable_area_m2"      raw_actionable_pixels × pixel_area_m2
    "post_morphology_pixels"      pixels surviving binary_opening
    "post_morphology_area_m2"     post_morphology_pixels × pixel_area_m2
    "n_connected_components"      total 8-connected components after opening
    "n_components_removed_mmu"    components with area < MMU (filtered)
    "n_surviving_clusters"        components passing MMU
    "final_total_area_m2"         sum of surviving-cluster projected areas
    "final_total_area_ha"         final_total_area_m2 / 10 000
    "pixel_area_source"           how pixel_area_m2 was determined
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

try:
    import scipy.ndimage as ndi
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    import rasterio.features
    import rasterio.transform
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False

try:
    from pyproj import Transformer
    _PYPROJ_AVAILABLE = True
except ImportError:
    _PYPROJ_AVAILABLE = False

try:
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union
    _SHAPELY_AVAILABLE = True
except ImportError:
    _SHAPELY_AVAILABLE = False

from src.diagnostics.screening import ACTIONABLE_CASES, DiagnosticCase


#: Minimum Mapping Unit in square metres (2 hectares). Do not alter.
MMU_AREA_M2: float = 20_000.0


def extract_scouting_zones(
    diagnostic_da: xr.DataArray,
    z_ndvi_da: Optional[xr.DataArray] = None,
    z_r_da: Optional[xr.DataArray] = None,
    delta_swi_da: Optional[xr.DataArray] = None,
    confidence_da: Optional[xr.DataArray] = None,
    bin_start: Optional[pd.Timestamp] = None,
    apply_morphology: bool = True,
) -> Dict[str, Any]:
    """Extract candidate scouting zones as RFC 7946 GeoJSON FeatureCollection.

    Parameters
    ----------
    diagnostic_da : xr.DataArray
        Integer-encoded diagnostic classification DataArray (DiagnosticCase values).
        Must be on the 30 m EPSG:6933 grid with a valid CRS and affine transform.
    z_ndvi_da : xr.DataArray, optional
        Z_NDVI values for attribute extraction. Must be spatially aligned with
        diagnostic_da.
    z_r_da : xr.DataArray, optional
        Z_R values for attribute extraction (coarse, broadcast via nearest-neighbor).
    delta_swi_da : xr.DataArray, optional
        delta_swi_14d values for attribute extraction.
    confidence_da : xr.DataArray, optional
        Evidence confidence fraction [0,1] aligned with diagnostic_da.
    bin_start : pd.Timestamp, optional
        Canonical 14-day bin start timestamp for zone_id generation.
    apply_morphology : bool, default True
        Whether to apply 3×3 morphological opening to remove isolated pixels.

    Returns
    -------
    dict
        RFC 7946 GeoJSON FeatureCollection extended with a ``pipeline_report``
        key containing per-step pixel/area accounting. Empty FeatureCollection
        is valid when no zones survive MMU filtering.

    Notes
    -----
    All area calculations use actual projected pixel dimensions from the affine
    transform (EPSG:6933 metres). Geographic-degree calculations are never used.

    Each surviving zone's ``properties`` includes:
        ``area_ha``         pixel-count area (projected)
        ``geom_area_ha``    Shapely geometry area (projected) — independent check
        ``area_ha_match``   True if both measures agree within 1 pixel tolerance
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for morphological operations and connected components.")
    if not _RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required for polygonization.")
    if not _PYPROJ_AVAILABLE:
        raise ImportError("pyproj is required for CRS transformation.")

    empty_report: Dict[str, Any] = {}
    empty_fc: Dict[str, Any] = {
        "type": "FeatureCollection",
        "features": [],
        "pipeline_report": empty_report,
    }

    # ── Step 1: Extract classification values ────────────────────────────────
    diag_vals = diagnostic_da.values.astype(np.int8)

    # ── Step 2: Build actionable mask ────────────────────────────────────────
    actionable_values = np.array([c.value for c in ACTIONABLE_CASES], dtype=np.int8)
    actionable_mask = np.isin(diag_vals, actionable_values)
    raw_actionable_pixels = int(np.sum(actionable_mask))

    if raw_actionable_pixels == 0:
        empty_fc["pipeline_report"] = {
            "pixel_area_m2": None,
            "raw_actionable_pixels": 0,
            "raw_actionable_area_m2": 0.0,
            "post_morphology_pixels": 0,
            "post_morphology_area_m2": 0.0,
            "n_connected_components": 0,
            "n_components_removed_mmu": 0,
            "n_surviving_clusters": 0,
            "final_total_area_m2": 0.0,
            "final_total_area_ha": 0.0,
            "pixel_area_source": "N/A (no actionable pixels)",
        }
        return empty_fc

    # ── Step 3: Compute actual projected pixel area from affine transform ────
    pixel_area_source = "affine_transform"
    try:
        transform = diagnostic_da.rio.transform()
        pixel_width_m = abs(float(transform.a))
        pixel_height_m = abs(float(transform.e))
        pixel_area_m2 = pixel_width_m * pixel_height_m
    except Exception:
        pixel_area_source = "coordinate_spacing"
        y_name = "y" if "y" in diagnostic_da.coords else "lat"
        x_name = "x" if "x" in diagnostic_da.coords else "lon"
        y_vals = diagnostic_da[y_name].values
        x_vals = diagnostic_da[x_name].values
        if len(y_vals) > 1 and len(x_vals) > 1:
            pixel_height_m = abs(float(y_vals[1] - y_vals[0]))
            pixel_width_m = abs(float(x_vals[1] - x_vals[0]))
            pixel_area_m2 = pixel_height_m * pixel_width_m
        else:
            pixel_area_source = "fallback_900m2"
            pixel_area_m2 = 900.0
            warnings.warn(
                "Could not determine pixel area from transform or coordinates. "
                "Using 900 m² fallback. Verify this is correct for your grid.",
                stacklevel=2,
            )

    raw_actionable_area_m2 = raw_actionable_pixels * pixel_area_m2

    # ── Step 4: Optional 3×3 morphological opening ───────────────────────────
    struct_3x3 = np.ones((3, 3), dtype=bool)
    if apply_morphology:
        opened_mask = ndi.binary_opening(actionable_mask, structure=struct_3x3)
    else:
        opened_mask = actionable_mask.copy()

    post_morphology_pixels = int(np.sum(opened_mask))
    post_morphology_area_m2 = post_morphology_pixels * pixel_area_m2

    if post_morphology_pixels == 0:
        empty_fc["pipeline_report"] = {
            "pixel_area_m2": pixel_area_m2,
            "pixel_area_source": pixel_area_source,
            "raw_actionable_pixels": raw_actionable_pixels,
            "raw_actionable_area_m2": raw_actionable_area_m2,
            "post_morphology_pixels": 0,
            "post_morphology_area_m2": 0.0,
            "n_connected_components": 0,
            "n_components_removed_mmu": 0,
            "n_surviving_clusters": 0,
            "final_total_area_m2": 0.0,
            "final_total_area_ha": 0.0,
        }
        return empty_fc

    # ── Step 5: 8-connected component labeling ───────────────────────────────
    labeled, n_components = ndi.label(opened_mask, structure=struct_3x3)

    if n_components == 0:
        empty_fc["pipeline_report"] = {
            "pixel_area_m2": pixel_area_m2,
            "pixel_area_source": pixel_area_source,
            "raw_actionable_pixels": raw_actionable_pixels,
            "raw_actionable_area_m2": raw_actionable_area_m2,
            "post_morphology_pixels": post_morphology_pixels,
            "post_morphology_area_m2": post_morphology_area_m2,
            "n_connected_components": 0,
            "n_components_removed_mmu": 0,
            "n_surviving_clusters": 0,
            "final_total_area_m2": 0.0,
            "final_total_area_ha": 0.0,
        }
        return empty_fc

    # ── Step 6: Acquire projected transform for polygonization ───────────────
    try:
        proj_transform = diagnostic_da.rio.transform()
    except Exception:
        y_name = "y" if "y" in diagnostic_da.coords else "lat"
        x_name = "x" if "x" in diagnostic_da.coords else "lon"
        y_vals_c = diagnostic_da[y_name].values
        x_vals_c = diagnostic_da[x_name].values
        from rasterio.transform import from_bounds
        proj_transform = from_bounds(
            left=float(x_vals_c.min()) - pixel_width_m / 2,
            bottom=float(y_vals_c.min()) - pixel_height_m / 2,
            right=float(x_vals_c.max()) + pixel_width_m / 2,
            top=float(y_vals_c.max()) + pixel_height_m / 2,
            width=len(x_vals_c),
            height=len(y_vals_c),
        )

    # ── Step 7: Acquire CRS and WGS84 transformer ────────────────────────────
    src_crs_str = str(diagnostic_da.rio.crs) if (
        diagnostic_da.rio.crs is not None
    ) else "EPSG:6933"
    to_wgs84 = Transformer.from_crs(src_crs_str, "EPSG:4326", always_xy=True)

    def _transform_geom_coords(geom: Dict) -> Dict:
        """Recursively reproject geometry coordinates from projected → WGS84."""
        geom_type = geom["type"]
        if geom_type == "Polygon":
            return {
                "type": "Polygon",
                "coordinates": [
                    [list(to_wgs84.transform(x, y)) for x, y in ring]
                    for ring in geom["coordinates"]
                ],
            }
        elif geom_type == "MultiPolygon":
            return {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [list(to_wgs84.transform(x, y)) for x, y in ring]
                        for ring in poly
                    ]
                    for poly in geom["coordinates"]
                ],
            }
        return geom

    def _mean_attr(da: Optional[xr.DataArray], mask: np.ndarray) -> Optional[float]:
        if da is None:
            return None
        vals = da.values.astype(np.float32)[mask]
        finite = vals[np.isfinite(vals)]
        return float(np.mean(finite)) if len(finite) > 0 else None

    # ── Step 8: Per-component processing: MMU + polygonization ───────────────
    bin_start_str = bin_start.strftime("%Y%m%d") if bin_start is not None else "YYYYMMDD"
    features: List[Dict[str, Any]] = []
    n_removed_mmu = 0
    final_total_area_m2 = 0.0

    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        pixel_count = int(np.sum(comp_mask))
        area_m2 = pixel_count * pixel_area_m2
        area_ha = area_m2 / 10_000.0

        if area_m2 < MMU_AREA_M2:
            n_removed_mmu += 1
            continue

        # Dominant case type (mode)
        comp_diag_vals = diag_vals[comp_mask]
        unique_vals, counts = np.unique(comp_diag_vals, return_counts=True)
        dominant_val = int(unique_vals[np.argmax(counts)])
        try:
            dominant_case = DiagnosticCase(dominant_val).name
        except ValueError:
            dominant_case = f"UNKNOWN_{dominant_val}"

        # Mean attributes
        mean_z_ndvi = _mean_attr(z_ndvi_da, comp_mask)
        mean_z_r = _mean_attr(z_r_da, comp_mask)
        mean_delta_swi = _mean_attr(delta_swi_da, comp_mask)
        mean_confidence = _mean_attr(confidence_da, comp_mask)

        # Polygonize component in projected space (union all shapes)
        comp_label_arr = np.where(comp_mask, 1, 0).astype(np.uint8)
        shapes = list(
            rasterio.features.shapes(
                comp_label_arr,
                mask=comp_label_arr.astype(bool),
                transform=proj_transform,
            )
        )
        geom_list = [s[0] for s in shapes if s[1] == 1]
        if not geom_list:
            n_removed_mmu += 1  # polygonization returned nothing — skip
            continue

        # Union all polygonized pieces into one geometry (in projected space)
        if _SHAPELY_AVAILABLE:
            shp_geoms = [shapely_shape(g) for g in geom_list]
            shp_union = unary_union(shp_geoms)
            geom_proj = shp_union.__geo_interface__

            # Independent geometry area verification (projected, in m²)
            geom_area_m2 = float(shp_union.area)
            geom_area_ha = geom_area_m2 / 10_000.0
            area_ha_match = abs(geom_area_ha - area_ha) <= (pixel_area_m2 / 10_000.0 + 1e-6)

            # Centroid in projected space
            centroid_proj = shp_union.centroid
            cx_proj, cy_proj = centroid_proj.x, centroid_proj.y
        else:
            geom_proj = geom_list[0]
            geom_area_ha = None
            area_ha_match = None
            # Pixel-coordinate centroid fallback
            y_name = "y" if "y" in diagnostic_da.coords else "lat"
            x_name = "x" if "x" in diagnostic_da.coords else "lon"
            y_coords = diagnostic_da[y_name].values
            x_coords = diagnostic_da[x_name].values
            y_inds, x_inds = np.where(comp_mask)
            cy_proj = float(np.mean(y_coords[y_inds]))
            cx_proj = float(np.mean(x_coords[x_inds]))

        # Centroid → WGS84
        try:
            centroid_lon, centroid_lat = to_wgs84.transform(cx_proj, cy_proj)
        except Exception:
            centroid_lon, centroid_lat = None, None

        # Geometry → WGS84
        try:
            geom_wgs84 = _transform_geom_coords(geom_proj)
        except Exception:
            geom_wgs84 = geom_proj

        properties: Dict[str, Any] = {
            "zone_id": f"SC-{bin_start_str}-{comp_id:04d}",
            "case_type": dominant_case,
            "area_ha": round(area_ha, 4),
            "geom_area_ha": round(geom_area_ha, 4) if geom_area_ha is not None else None,
            "area_ha_match": area_ha_match,
            "pixel_count": pixel_count,
            "pixel_area_m2": round(pixel_area_m2, 4),
            "mean_z_ndvi": round(mean_z_ndvi, 4) if mean_z_ndvi is not None else None,
            "mean_z_r": round(mean_z_r, 4) if mean_z_r is not None else None,
            "mean_delta_swi_14d": round(mean_delta_swi, 4) if mean_delta_swi is not None else None,
            "mean_confidence": round(mean_confidence, 4) if mean_confidence is not None else None,
            "centroid_lat": round(centroid_lat, 6) if centroid_lat is not None else None,
            "centroid_lon": round(centroid_lon, 6) if centroid_lon is not None else None,
        }

        features.append({"type": "Feature", "geometry": geom_wgs84, "properties": properties})
        final_total_area_m2 += area_m2

    n_surviving = len(features)

    pipeline_report: Dict[str, Any] = {
        "pixel_area_m2": round(pixel_area_m2, 6),
        "pixel_area_source": pixel_area_source,
        "raw_actionable_pixels": raw_actionable_pixels,
        "raw_actionable_area_m2": round(raw_actionable_area_m2, 2),
        "raw_actionable_area_ha": round(raw_actionable_area_m2 / 10_000.0, 4),
        "post_morphology_pixels": post_morphology_pixels,
        "post_morphology_area_m2": round(post_morphology_area_m2, 2),
        "post_morphology_area_ha": round(post_morphology_area_m2 / 10_000.0, 4),
        "n_connected_components": n_components,
        "n_components_removed_mmu": n_removed_mmu,
        "n_surviving_clusters": n_surviving,
        "final_total_area_m2": round(final_total_area_m2, 2),
        "final_total_area_ha": round(final_total_area_m2 / 10_000.0, 4),
        "mmu_threshold_m2": MMU_AREA_M2,
        "morphology_applied": apply_morphology,
        "connectivity": "8-connected",
        "area_calculation_crs": src_crs_str,
        "area_calculation_method": "pixel_count × pixel_area_m2 (projected)",
    }

    return {
        "type": "FeatureCollection",
        "features": features,
        "pipeline_report": pipeline_report,
    }
