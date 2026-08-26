"""Configuration loading and management utilities."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


def get_repo_root() -> Path:
    """Return the absolute path to the repository root directory.

    Determined relative to this module's location (src/utils/config.py).

    Returns
    -------
    Path
        Absolute path to the repository root directory.
    """
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load and parse a YAML configuration file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the configuration YAML file. If not specified, defaults to
        `configs/project.yaml` relative to the repository root. If a relative
        path is provided, it is resolved relative to the current working
        directory first, and then relative to the repository root if not found.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    ValueError
        If the configuration file is empty or does not define a mapping.
    """
    if config_path is None:
        resolved_path = get_repo_root() / "configs" / "project.yaml"
    else:
        path = Path(config_path)
        if path.is_absolute():
            resolved_path = path
        elif path.exists():
            resolved_path = path.resolve()
        else:
            resolved_path = (get_repo_root() / path).resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found at: '{resolved_path}'"
        )

    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(
            f"Configuration file at '{resolved_path}' is empty."
        )

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file at '{resolved_path}' must define a top-level mapping, got {type(config).__name__}."
        )

    return config
