"""Area of Interest (AOI) validation, containment, and GeoJSON loader utilities.

Provides geometric validation and containment checks against regional boundaries
(Uasin Gishu County and Kenya bounding extents).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import shapely.geometry
from shapely.geometry import Polygon, box, shape

from src.utils.config import get_repo_root


# Canonical bounding extents for spatial validation
KENYA_BBOX: Tuple[float, float, float, float] = (33.9, -4.7, 41.9, 5.5)  # (min_lon, min_lat, max_lon, max_lat)
UASIN_GISHU_BBOX: Tuple[float, float, float, float] = (34.88, 0.17, 35.58, 0.94)  # (min_lon, min_lat, max_lon, max_lat)
PILOT_AOI_BBOX: Tuple[float, float, float, float] = (35.15, 0.55, 35.35, 0.75)  # Moiben-Soy Agricultural Pilot Zone


def validate_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Validate bounding box coordinate format and geometric integrity.

    Parameters
    ----------
    bbox : tuple of float
        (min_lon, min_lat, max_lon, max_lat)

    Returns
    -------
    tuple of float
        Validated bounding box tuple.

    Raises
    ------
    ValueError
        If coordinates are out of valid global range [-180..180, -90..90] or if min >= max.
    """
    if len(bbox) != 4:
        raise ValueError(f"Bounding box must have 4 elements (min_lon, min_lat, max_lon, max_lat), got {len(bbox)}.")

    min_lon, min_lat, max_lon, max_lat = bbox

    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise ValueError(f"Longitude coordinates [{min_lon}, {max_lon}] out of range [-180, 180].")

    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise ValueError(f"Latitude coordinates [{min_lat}, {max_lat}] out of range [-90, 90].")

    if min_lon >= max_lon:
        raise ValueError(f"min_lon ({min_lon}) must be strictly less than max_lon ({max_lon}).")

    if min_lat >= max_lat:
        raise ValueError(f"min_lat ({min_lat}) must be strictly less than max_lat ({max_lat}).")

    return (float(min_lon), float(min_lat), float(max_lon), float(max_lat))


def bbox_to_polygon(bbox: Tuple[float, float, float, float]) -> Polygon:
    """Convert a bounding box tuple to a Shapely Polygon.

    Parameters
    ----------
    bbox : tuple of float
        (min_lon, min_lat, max_lon, max_lat)

    Returns
    -------
    shapely.geometry.Polygon
    """
    valid_bbox = validate_bbox(bbox)
    return box(*valid_bbox)


def validate_aoi_geometry(geometry: Any) -> shapely.geometry.base.BaseGeometry:
    """Validate that a geometry object is valid and non-empty.

    Parameters
    ----------
    geometry : shapely geometry, dict (GeoJSON), or tuple (bbox)

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Valid Shapely geometry.

    Raises
    ------
    ValueError
        If geometry is invalid, empty, or cannot be parsed.
    """
    if isinstance(geometry, tuple) and len(geometry) == 4:
        geom = bbox_to_polygon(geometry)
    elif isinstance(geometry, dict):
        if geometry.get("type") == "FeatureCollection" and "features" in geometry:
            features = geometry["features"]
            if not features:
                raise ValueError("GeoJSON FeatureCollection contains no features.")
            geom = shape(features[0]["geometry"])
        elif geometry.get("type") == "Feature" and "geometry" in geometry:
            geom = shape(geometry["geometry"])
        else:
            geom = shape(geometry)
    elif isinstance(geometry, shapely.geometry.base.BaseGeometry):
        geom = geometry
    else:
        raise ValueError(f"Unsupported geometry type: {type(geometry).__name__}")

    if not geom.is_valid:
        raise ValueError("AOI geometry is topologically invalid.")
    if geom.is_empty:
        raise ValueError("AOI geometry is empty.")

    return geom


def check_containment(
    aoi_geom: Any,
    reference_bbox: Tuple[float, float, float, float] = UASIN_GISHU_BBOX,
    buffer_deg: float = 0.05,
) -> bool:
    """Check whether an AOI geometry is spatially contained within a reference bounding box.

    Parameters
    ----------
    aoi_geom : shapely geometry, dict, or bbox tuple
        Geometry to test.
    reference_bbox : tuple of float, default UASIN_GISHU_BBOX
        Reference bounding box (min_lon, min_lat, max_lon, max_lat).
    buffer_deg : float, default 0.05
        Tolerance buffer in degrees around reference extent.

    Returns
    -------
    bool
        True if AOI is contained within the buffered reference extent, False otherwise.
    """
    geom = validate_aoi_geometry(aoi_geom)
    ref_poly = box(
        reference_bbox[0] - buffer_deg,
        reference_bbox[1] - buffer_deg,
        reference_bbox[2] + buffer_deg,
        reference_bbox[3] + buffer_deg,
    )
    return bool(ref_poly.contains(geom) or ref_poly.intersects(geom))


def load_aoi_geojson(geojson_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse an AOI GeoJSON file with path resolution.

    Parameters
    ----------
    geojson_path : str or Path
        Relative or absolute path to GeoJSON file.

    Returns
    -------
    dict
        Parsed GeoJSON dictionary.

    Raises
    ------
    FileNotFoundError
        If the GeoJSON file does not exist.
    """
    path = Path(geojson_path)
    if path.is_absolute():
        resolved_path = path
    elif path.exists():
        resolved_path = path.resolve()
    else:
        resolved_path = (get_repo_root() / path).resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(f"AOI GeoJSON file not found at: '{resolved_path}'")

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate topology
    validate_aoi_geometry(data)
    return data
