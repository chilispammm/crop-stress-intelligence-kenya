"""Data preprocessing and alignment package."""

from src.preprocessing.alignment import (
    MultiResolutionCube,
    ModalitySchema,
    SCHEMA_CHIRPS,
    SCHEMA_GRAFS,
    SCHEMA_MASK,
    SCHEMA_SENTINEL2,
)
from src.preprocessing.optical import apply_scl_mask, DEFAULT_VALID_SCL_CLASSES
from src.preprocessing.masking import apply_cropland_mask
from src.preprocessing.compositing import composite_14d, smooth_temporal_series
from src.preprocessing.diagnostics import generate_quality_report

__all__ = [
    "DEFAULT_VALID_SCL_CLASSES",
    "ModalitySchema",
    "MultiResolutionCube",
    "SCHEMA_CHIRPS",
    "SCHEMA_GRAFS",
    "SCHEMA_MASK",
    "SCHEMA_SENTINEL2",
    "apply_cropland_mask",
    "apply_scl_mask",
    "composite_14d",
    "generate_quality_report",
    "smooth_temporal_series",
]
