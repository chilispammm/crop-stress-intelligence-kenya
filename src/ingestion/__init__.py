"""Data ingestion subpackage for multi-modal Earth Observation & climate datasets."""

from src.ingestion.chirps import (
    DEFAULT_CHIRPS_VARIABLES,
    load_chirps_data,
    validate_chirps_dataset,
)
from src.ingestion.grafs import (
    DEFAULT_GRAFS_VARIABLES,
    VARIABLE_METADATA as GRAFS_VARIABLE_METADATA,
    build_grafs_url,
    load_grafs_data,
    validate_grafs_dataset,
)
from src.ingestion.mask import (
    load_crop_mask,
    validate_crop_mask,
)
from src.ingestion.sentinel2 import (
    DEFAULT_S2_BANDS,
    load_sentinel2_data,
    validate_sentinel2_dataset,
)

__all__ = [
    "DEFAULT_CHIRPS_VARIABLES",
    "DEFAULT_GRAFS_VARIABLES",
    "DEFAULT_S2_BANDS",
    "GRAFS_VARIABLE_METADATA",
    "build_grafs_url",
    "load_chirps_data",
    "load_crop_mask",
    "load_grafs_data",
    "load_sentinel2_data",
    "validate_chirps_dataset",
    "validate_crop_mask",
    "validate_grafs_dataset",
    "validate_sentinel2_dataset",
]
