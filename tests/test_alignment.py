"""Unit tests for multi-resolution alignment container and modality schemas."""

import pytest
import xarray as xr

from src.preprocessing.alignment import (
    MultiResolutionCube,
    SCHEMA_CHIRPS,
    SCHEMA_GRAFS,
    SCHEMA_MASK,
    SCHEMA_SENTINEL2,
)
from src.utils.provenance import create_provenance_metadata, attach_provenance


def test_modality_schemas_properties() -> None:
    """Verify modality schema properties reflect multi-resolution design."""
    assert SCHEMA_SENTINEL2.native_spatial_resolution == "10-20 m"
    assert SCHEMA_SENTINEL2.native_crs == "EPSG:32736"

    assert SCHEMA_CHIRPS.native_spatial_resolution == "0.05 degree (~5.5 km)"
    assert SCHEMA_CHIRPS.native_crs == "EPSG:4326"

    assert SCHEMA_GRAFS.native_spatial_resolution == "0.1 degree (~10 km)"
    assert SCHEMA_GRAFS.native_crs == "EPSG:4326"

    assert SCHEMA_MASK.native_spatial_resolution == "20 m"
    assert SCHEMA_MASK.native_crs == "EPSG:32736"


def test_multi_resolution_cube_lifecycle() -> None:
    """Verify initializing, validating, and summarizing MultiResolutionCube."""
    # Create minimal datasets with attached provenance
    prov = create_provenance_metadata(
        source_name="Test Source",
        product_name="Test Product",
        product_version="v1.0",
        spatial_resolution="10 m",
        temporal_resolution="Daily",
        native_crs="EPSG:4326",
        known_limitations=["None"],
    )

    opt_ds = xr.Dataset(data_vars={"B04": (["lat", "lon"], [[0.2]])}, coords={"lat": [0.0], "lon": [35.0]})
    attach_provenance(opt_ds, prov)

    cube = MultiResolutionCube(optical=opt_ds, aoi_bbox=(35.0, 0.0, 35.5, 0.5))
    validation_status = cube.validate_modalities(require_provenance=True)
    assert validation_status["optical"] is True

    summary = cube.summary()
    assert "optical" in summary["modalities"]
    assert summary["aoi_bbox"] == (35.0, 0.0, 35.5, 0.5)
