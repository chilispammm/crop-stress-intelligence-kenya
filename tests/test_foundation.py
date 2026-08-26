"""M1.0 / M2.0 / M3.0 / M4.0 Engineering Foundation test suite.

Verifies:
1. Python package modules import cleanly.
2. Configuration files can be loaded.
3. Required project configuration keys exist with valid types and values.
4. Missing configuration paths raise expected errors.
5. Cropland metadata contract reflects canonical specification.
All tests are strictly offline and perform no network/disk downloads.
"""

import pytest
import src
import src.diagnostics
import src.features
import src.ingestion
import src.preprocessing
import src.utils
from src.utils.config import get_repo_root, load_config


def test_package_imports() -> None:
    """Verify that all core top-level packages and subpackages import successfully."""
    assert hasattr(src, "__version__"), "src package must expose __version__"
    assert src.__version__ == "0.4.0", f"Unexpected version: {src.__version__}"
    assert src.ingestion is not None
    assert src.preprocessing is not None
    assert src.features is not None
    assert src.diagnostics is not None
    assert src.utils is not None


def test_repo_root_resolution() -> None:
    """Verify that the repository root path can be determined and contains configs directory."""
    repo_root = get_repo_root()
    assert repo_root.exists(), f"Repository root directory does not exist: {repo_root}"
    assert (repo_root / "configs").is_dir(), "configs directory must exist at repository root"
    assert (repo_root / "configs" / "project.yaml").is_file(), "configs/project.yaml must exist"


def test_load_default_config() -> None:
    """Verify that the default project configuration loads without errors."""
    config = load_config()
    assert isinstance(config, dict), "Configuration must load as a dictionary"
    assert len(config) > 0, "Configuration must not be empty"


def test_required_config_keys() -> None:
    """Verify that mandatory project configuration sections and keys exist with expected values."""
    config = load_config()

    # Project section
    assert "project" in config, "Missing 'project' section in config"
    project = config["project"]
    assert project.get("name") == "crop-stress-intelligence-kenya", (
        f"Unexpected project name: {project.get('name')}"
    )
    assert project.get("version") == "0.4.0", (
        f"Unexpected project version: {project.get('version')}"
    )

    # Study section
    assert "study" in config, "Missing 'study' section in config"
    study = config["study"]
    assert study.get("country") == "Kenya", (
        f"Unexpected country: {study.get('country')}"
    )
    assert study.get("region") == "Uasin Gishu", (
        f"Unexpected region: {study.get('region')}"
    )
    assert study.get("crop") == "maize", (
        f"Unexpected crop: {study.get('crop')}"
    )

    # Spatial section
    assert "spatial" in config, "Missing 'spatial' section in config"
    spatial = config["spatial"]
    assert spatial.get("optical_crs") == "EPSG:32736", (
        f"Unexpected optical CRS: {spatial.get('optical_crs')}"
    )

    # Climatology section
    assert "climatology" in config, "Missing 'climatology' section in config"
    clim = config["climatology"]
    assert clim.get("baseline_years") == [1991, 2020]
    assert clim.get("expected_baseline_months") == 360
    assert clim.get("min_valid_observations") == 20
    assert clim.get("std_epsilon") == 0.0001


def test_cropland_metadata_contract() -> None:
    """Verify crop_mask_eastern is configured as 20 m native with storage CRS EPSG:6933 and working CRS EPSG:32736."""
    config = load_config()
    crop_mask_cfg = config.get("sources", {}).get("crop_mask", {})
    assert crop_mask_cfg.get("collection_id") == "crop_mask_eastern"
    assert crop_mask_cfg.get("native_resolution_m") == 20
    assert crop_mask_cfg.get("storage_crs") == "EPSG:6933"
    assert crop_mask_cfg.get("working_crs") == "EPSG:32736"


def test_missing_config_raises_file_not_found() -> None:
    """Verify that attempting to load a non-existent configuration file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent_file.yaml")
