"""Unit tests for AOI validation, containment, and GeoJSON boundaries."""

import pytest
from shapely.geometry import Polygon, box

from src.utils.aoi import (
    KENYA_BBOX,
    PILOT_AOI_BBOX,
    UASIN_GISHU_BBOX,
    bbox_to_polygon,
    check_containment,
    load_aoi_geojson,
    validate_aoi_geometry,
    validate_bbox,
)


def test_validate_bbox_valid() -> None:
    """Verify that a valid bounding box passes validation."""
    valid = (35.0, 0.2, 35.5, 0.8)
    assert validate_bbox(valid) == (35.0, 0.2, 35.5, 0.8)


def test_validate_bbox_invalid_ranges() -> None:
    """Verify that coordinates outside geographic boundaries raise ValueError."""
    with pytest.raises(ValueError, match="out of range"):
        validate_bbox((200.0, 0.0, 35.0, 1.0))

    with pytest.raises(ValueError, match="min_lon .* strictly less"):
        validate_bbox((36.0, 0.0, 35.0, 1.0))

    with pytest.raises(ValueError, match="min_lat .* strictly less"):
        validate_bbox((35.0, 1.0, 36.0, 0.0))


def test_bbox_to_polygon() -> None:
    """Verify conversion from bbox tuple to Shapely Polygon."""
    poly = bbox_to_polygon((35.0, 0.0, 36.0, 1.0))
    assert isinstance(poly, Polygon)
    assert poly.area == pytest.approx(1.0)


def test_validate_aoi_geometry() -> None:
    """Verify geometry validation on Shapely polygons, dicts, and tuples."""
    poly = box(35.0, 0.0, 35.5, 0.5)
    assert validate_aoi_geometry(poly).equals(poly)

    geojson_dict = {"type": "Polygon", "coordinates": [[[35.0, 0.0], [35.5, 0.0], [35.5, 0.5], [35.0, 0.5], [35.0, 0.0]]]}
    assert validate_aoi_geometry(geojson_dict).is_valid


def test_check_containment_uasin_gishu() -> None:
    """Verify AOI containment check inside Uasin Gishu County bounds."""
    # Inside Uasin Gishu
    pilot_aoi = (35.15, 0.55, 35.35, 0.75)
    assert check_containment(pilot_aoi, UASIN_GISHU_BBOX) is True

    # Far outside Uasin Gishu (e.g. coastal Mombasa)
    coastal_aoi = (39.5, -4.1, 39.8, -3.9)
    assert check_containment(coastal_aoi, UASIN_GISHU_BBOX) is False
    # But still inside Kenya
    assert check_containment(coastal_aoi, KENYA_BBOX) is True


def test_load_canonical_county_geojson() -> None:
    """Verify loading and validating canonical Uasin Gishu County GeoJSON."""
    county_geojson = load_aoi_geojson("configs/aoi/uasin_gishu_county.geojson")
    assert county_geojson["type"] == "FeatureCollection"
    assert county_geojson["features"][0]["properties"]["county_name"] == "Uasin Gishu"
    assert county_geojson["features"][0]["properties"]["optical_crs"] == "EPSG:32736"

    geom = validate_aoi_geometry(county_geojson)
    assert geom.is_valid
    assert geom.area > 0.1  # Significant geographic polygon area


def test_load_pilot_aoi_geojson() -> None:
    """Verify loading, validating, and containment of the pilot AOI GeoJSON."""
    pilot_geojson = load_aoi_geojson("configs/aoi/pilot_aoi.geojson")
    assert pilot_geojson["type"] == "FeatureCollection"
    assert pilot_geojson["features"][0]["properties"]["aoi_id"] == "ug_pilot_moiben_01"

    geom = validate_aoi_geometry(pilot_geojson)
    assert geom.is_valid

    # Verify pilot AOI is spatially contained within Uasin Gishu County bounds
    assert check_containment(geom, UASIN_GISHU_BBOX) is True
