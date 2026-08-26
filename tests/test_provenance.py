"""Unit tests for data provenance creation and validation."""

import pytest
import xarray as xr

from src.utils.provenance import (
    attach_provenance,
    create_provenance_metadata,
    validate_provenance,
)


def test_create_and_validate_provenance() -> None:
    """Verify complete provenance metadata generation and attachment."""
    prov = create_provenance_metadata(
        source_name="European Space Agency",
        product_name="Sentinel-2 L2A",
        product_version="2.0",
        spatial_resolution="10 m",
        temporal_resolution="5 days",
        native_crs="EPSG:32736",
        known_limitations=["Cloud cover"],
        source_url="https://dataspace.copernicus.eu",
    )

    assert prov["provenance_source_name"] == "European Space Agency"
    assert "provenance_access_timestamp_utc" in prov

    ds = xr.Dataset()
    attach_provenance(ds, prov)
    # Should not raise
    validate_provenance(ds)


def test_validate_provenance_missing_field() -> None:
    """Verify that incomplete provenance raises ValueError."""
    ds = xr.Dataset(attrs={"provenance_source_name": "Incomplete"})
    with pytest.raises(ValueError, match="missing mandatory provenance attributes"):
        validate_provenance(ds)
