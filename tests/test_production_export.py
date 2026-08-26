"""Unit and integration tests for M6 production export orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.export.production_export import execute_m6_seasonal_pipeline


def test_execute_m6_seasonal_pipeline(tmp_path: Path):
    """Verify full M6 production export pipeline runs end-to-end and outputs all deliverables."""
    out_dir = tmp_path / "m6_export"

    manifest = execute_m6_seasonal_pipeline(
        output_dir=out_dir,
        aoi_id="ug_pilot_moiben_01",
        aoi_name="Moiben-Soy Agricultural Pilot Zone",
        aoi_bbox=(35.15, 0.55, 35.35, 0.75),
        seed=42,
    )

    assert manifest["status"] == "SUCCESS"
    assert manifest["n_bins_executed"] == 15
    assert manifest["total_clusters"] > 0
    assert manifest["cumulative_candidate_area_ha"] > 0.0

    artifacts = manifest["exported_artifacts"]

    # Verify Summary JSON
    summary_json = Path(artifacts["summary_json"])
    assert summary_json.exists()
    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["temporal_scope"]["n_canonical_bins"] == 15
        assert len(data["scientific_caveats"]) >= 5

    # Verify Summary Markdown
    summary_md = Path(artifacts["summary_markdown"])
    assert summary_md.exists()
    with open(summary_md, "r", encoding="utf-8") as f:
        md_text = f.read()
        assert "# Seasonal Diagnostic Screening Executive Summary" in md_text
        assert "Non-Overclaiming Scientific Caveats" in md_text

    # Verify Seasonal GeoJSON
    seasonal_geojson = Path(artifacts["seasonal_geojson"])
    assert seasonal_geojson.exists()
    with open(seasonal_geojson, "r", encoding="utf-8") as f:
        fc = json.load(f)
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == manifest["total_clusters"]

    # Verify Generated Figures
    profile_png = Path(artifacts["temporal_profile_plot"])
    timeline_png = Path(artifacts["area_timeline_plot"])
    persist_png = Path(artifacts["persistence_heatmap_plot"])

    assert profile_png.exists()
    assert timeline_png.exists()
    assert persist_png.exists()

    # Verify actionable bin maps
    figs_dir = Path(artifacts["figures_directory"])
    spatial_maps = list(figs_dir.glob("spatial_diagnostic_bin_*.png"))
    assert len(spatial_maps) > 0


def test_derive_canonical_diagnostic_grid_coords_spatial_intersection():
    """Verify dynamically derived reference grid intersects and falls inside canonical AOI bounds."""
    from src.export.production_export import derive_canonical_diagnostic_grid_coords
    import pyproj
    from shapely.geometry import box

    aoi_bbox = (35.15, 0.55, 35.35, 0.75)
    aoi_geom = box(*aoi_bbox)

    x_coords, y_coords = derive_canonical_diagnostic_grid_coords(
        aoi_bbox=aoi_bbox,
        target_crs="EPSG:6933",
        resolution_m=30.0,
        nrows=86,
        ncols=65,
    )

    assert len(x_coords) == 65
    assert len(y_coords) == 86
    assert np.isclose(x_coords[1] - x_coords[0], 30.0)
    assert np.isclose(y_coords[1] - y_coords[0], 30.0)

    # Reproject grid corner bounds back to EPSG:4326
    tf_to_4326 = pyproj.Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)
    min_lon, min_lat = tf_to_4326.transform(x_coords[0], y_coords[0])
    max_lon, max_lat = tf_to_4326.transform(x_coords[-1], y_coords[-1])

    grid_bbox_geom = box(min_lon, min_lat, max_lon, max_lat)

    # Regression assertion 1: Reference grid MUST intersect the canonical AOI
    assert grid_bbox_geom.intersects(aoi_geom), "Reference grid does not intersect canonical AOI!"

    # Regression assertion 2: Reference grid MUST be contained within AOI bounds (not displaced)
    assert aoi_bbox[0] <= min_lon <= aoi_bbox[2]
    assert aoi_bbox[0] <= max_lon <= aoi_bbox[2]
    assert aoi_bbox[1] <= min_lat <= aoi_bbox[3]
    assert aoi_bbox[1] <= max_lat <= aoi_bbox[3]

    # Regression assertion 3: Longitude MUST NOT have a ~26.7 degree displacement (e.g. at 8.45 E)
    assert abs(min_lon - 35.25) < 0.20, f"Grid longitude {min_lon} is displaced from Uasin Gishu (~35.25 E)!"
    assert abs(min_lon - 8.45) > 20.0, "Regression: grid coordinate at ~8.45 E (Gulf of Guinea offset) detected!"


def test_exported_geojson_spatial_containment_and_alignment(tmp_path: Path):
    """Regression test: verify exported GeoJSON polygons intersect AOI and plot viewport."""
    from shapely.geometry import box, shape as shapely_shape

    out_dir = tmp_path / "spatial_check"
    aoi_bbox = (35.15, 0.55, 35.35, 0.75)
    aoi_geom = box(*aoi_bbox)

    manifest = execute_m6_seasonal_pipeline(
        output_dir=out_dir,
        aoi_bbox=aoi_bbox,
        seed=42,
    )

    seasonal_geojson = Path(manifest["exported_artifacts"]["seasonal_geojson"])
    with open(seasonal_geojson, "r", encoding="utf-8") as f:
        fc = json.load(f)

    assert len(fc["features"]) > 0

    for feat in fc["features"]:
        geom = shapely_shape(feat["geometry"])
        props = feat["properties"]

        # Regression check: Every polygon must intersect the pilot AOI
        assert geom.intersects(aoi_geom), f"Polygon {props['zone_id']} does not intersect AOI!"

        # Regression check: Centroid coordinates must be in Uasin Gishu (Lon ~35.25), NOT West Africa (~8.45)
        c_lon = props["centroid_lon"]
        c_lat = props["centroid_lat"]
        assert 35.15 <= c_lon <= 35.35, f"Centroid lon {c_lon} is outside AOI bounds!"
        assert 0.55 <= c_lat <= 0.75, f"Centroid lat {c_lat} is outside AOI bounds!"
        assert abs(c_lon - 8.45) > 20.0, f"Displacement error: polygon centroid is near 8.45 E!"

    # Verify per-bin GeoJSONs
    for bin_idx in [1, 10, 15]:
        bin_files = list((out_dir / "geojson").glob(f"scouting_zones_bin_{bin_idx:02d}_*.geojson"))
        assert len(bin_files) == 1
        with open(bin_files[0], "r", encoding="utf-8") as bf:
            bin_fc = json.load(bf)
            assert "grid_bbox" in bin_fc["properties"]
            g_bb = bin_fc["properties"]["grid_bbox"]
            assert aoi_geom.contains(box(*g_bb))
            for feat in bin_fc["features"]:
                g = shapely_shape(feat["geometry"])
                assert g.intersects(aoi_geom)

    # Seasonal GeoJSON must also contain grid_bbox
    assert "grid_bbox" in fc["properties"]
    s_g_bb = fc["properties"]["grid_bbox"]
    assert aoi_geom.contains(box(*s_g_bb))

