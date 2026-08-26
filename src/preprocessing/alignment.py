"""Multi-resolution dataset alignment conventions and multi-modal container.

IMPORTANT SCIENTIFIC PRINCIPLE:
    Alignment does NOT mean forcing all datasets onto a common 20 m raster.
    Coarse climate (CHIRPS ~5.5 km) and hydrological (GRAFS ~10 km) grids must
    retain their native spatial scales to preserve physical truth.

This module provides:
1. `MultiResolutionCube`: A decoupled multi-modal container holding native-scale datasets.
2. `ModalitySchema`: Schema descriptor specifying spatial/temporal resolutions, units, and CRS.
3. Metadata validation ensuring cross-dataset alignment conventions are maintained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import xarray as xr

from src.utils.provenance import validate_provenance


@dataclass(frozen=True)
class ModalitySchema:
    """Schema specification for an individual data modality."""

    modality_name: str
    native_spatial_resolution: str
    native_temporal_resolution: str
    native_crs: str
    canonical_variables: Tuple[str, ...]
    primary_units: Dict[str, str]
    nodata_convention: str
    description: str


# Canonical Modality Schemas
SCHEMA_SENTINEL2 = ModalitySchema(
    modality_name="optical_sentinel2",
    native_spatial_resolution="10-20 m",
    native_temporal_resolution="5 days",
    native_crs="EPSG:32736",
    canonical_variables=("B02", "B03", "B04", "B08", "B11", "SCL"),
    primary_units={"B02": "reflectance", "B04": "reflectance", "B08": "reflectance", "SCL": "class_enum"},
    nodata_convention="NaN (float32)",
    description="High-resolution multispectral optical reflectance for canopy vigor & moisture screening.",
)

SCHEMA_CHIRPS = ModalitySchema(
    modality_name="climate_chirps",
    native_spatial_resolution="0.05 degree (~5.5 km)",
    native_temporal_resolution="Daily",
    native_crs="EPSG:4326",
    canonical_variables=("precip",),
    primary_units={"precip": "mm"},
    nodata_convention="NaN (float32)",
    description="Quasi-global precipitation estimates for rainfall deficit & drought screening.",
)

SCHEMA_GRAFS = ModalitySchema(
    modality_name="hydrology_grafs",
    native_spatial_resolution="0.1 degree (~10 km)",
    native_temporal_resolution="Daily",
    native_crs="EPSG:4326",
    canonical_variables=("s0", "s1"),
    primary_units={"s0": "dimensionless", "s1": "dimensionless"},
    nodata_convention="NaN (float32)",
    description="Satellite-guided topsoil and root-zone soil water index data assimilation.",
)

SCHEMA_MASK = ModalitySchema(
    modality_name="agricultural_mask",
    native_spatial_resolution="20 m",
    native_temporal_resolution="Static / Annual",
    native_crs="EPSG:32736",
    canonical_variables=("crop_mask",),
    primary_units={"crop_mask": "binary_flag"},
    nodata_convention="NaN (float32)",
    description="Cropland extent filter (native 20 m in EPSG:32736) isolating agricultural fields from non-crop land cover.",
)


@dataclass
class MultiResolutionCube:
    """Multi-modal container retaining each Earth observation dataset at native resolution.

    Attributes
    ----------
    optical : xr.Dataset, optional
        Sentinel-2 optical reflectance dataset at native 10-20 m scale.
    climate : xr.Dataset, optional
        CHIRPS precipitation dataset at native ~5.5 km scale.
    hydrology : xr.Dataset, optional
        GRAFS soil moisture dataset at native ~10 km scale.
    mask : xr.DataArray, optional
        Agricultural cropland mask at native 20 m scale.
    aoi_bbox : tuple of float, optional
        Common geographic bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    """

    optical: Optional[xr.Dataset] = None
    climate: Optional[xr.Dataset] = None
    hydrology: Optional[xr.Dataset] = None
    mask: Optional[xr.DataArray] = None
    aoi_bbox: Optional[Tuple[float, float, float, float]] = None

    def validate_modalities(self, require_provenance: bool = True) -> Dict[str, bool]:
        """Validate presence, coordinates, and provenance across all initialized modalities.

        Parameters
        ----------
        require_provenance : bool, default True
            If True, validates that standardized provenance metadata is attached.

        Returns
        -------
        dict[str, bool]
            Validation status for each loaded modality.
        """
        status = {}

        if self.optical is not None:
            assert isinstance(self.optical, xr.Dataset)
            if require_provenance:
                validate_provenance(self.optical)
            status["optical"] = True

        if self.climate is not None:
            assert isinstance(self.climate, xr.Dataset)
            if require_provenance:
                validate_provenance(self.climate)
            status["climate"] = True

        if self.hydrology is not None:
            assert isinstance(self.hydrology, xr.Dataset)
            if require_provenance:
                validate_provenance(self.hydrology)
            status["hydrology"] = True

        if self.mask is not None:
            assert isinstance(self.mask, (xr.DataArray, xr.Dataset))
            if require_provenance:
                validate_provenance(self.mask)
            status["mask"] = True

        return status

    def summary(self) -> Dict[str, Any]:
        """Return a structured summary of modalities, dimensions, and native resolutions."""
        res: Dict[str, Any] = {"aoi_bbox": self.aoi_bbox, "modalities": {}}
        if self.optical is not None:
            res["modalities"]["optical"] = {
                "variables": list(self.optical.data_vars),
                "sizes": dict(self.optical.sizes),
                "resolution": SCHEMA_SENTINEL2.native_spatial_resolution,
                "crs": SCHEMA_SENTINEL2.native_crs,
            }
        if self.climate is not None:
            res["modalities"]["climate"] = {
                "variables": list(self.climate.data_vars),
                "sizes": dict(self.climate.sizes),
                "resolution": SCHEMA_CHIRPS.native_spatial_resolution,
                "crs": SCHEMA_CHIRPS.native_crs,
            }
        if self.hydrology is not None:
            res["modalities"]["hydrology"] = {
                "variables": list(self.hydrology.data_vars),
                "sizes": dict(self.hydrology.sizes),
                "resolution": SCHEMA_GRAFS.native_spatial_resolution,
                "crs": SCHEMA_GRAFS.native_crs,
            }
        if self.mask is not None:
            res["modalities"]["mask"] = {
                "name": self.mask.name,
                "sizes": dict(self.mask.sizes),
                "resolution": SCHEMA_MASK.native_spatial_resolution,
                "crs": SCHEMA_MASK.native_crs,
            }
        return res
