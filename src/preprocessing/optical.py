"""Scene Classification Layer (SCL) optical cloud and quality masking.

Implements strict quality filtering based on Copernicus Sentinel-2 L2A SCL:
    Default valid classes: 4 (Vegetation), 5 (Not-vegetated / Bare Soil), 6 (Water).
    Default invalid classes: 0 (No Data), 1 (Saturated/Defective), 2 (Cast Shadow),
                             3 (Cloud Shadow), 7 (Unclassified), 8 (Cloud Medium Prob),
                             9 (Cloud High Prob), 10 (Thin Cirrus), 11 (Snow/Ice).
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np
import xarray as xr


DEFAULT_VALID_SCL_CLASSES: List[int] = [4, 5, 6]


def apply_scl_mask(
    ds_s2: xr.Dataset,
    valid_classes: Optional[Sequence[int]] = None,
    scl_var: str = "SCL",
) -> xr.Dataset:
    """Apply Scene Classification Layer (SCL) quality mask to Sentinel-2 reflectance bands.

    Masks out clouds, cloud shadows, cirrus, saturated/defective pixels, and unclassified
    areas by setting them to NaN.

    Parameters
    ----------
    ds_s2 : xr.Dataset
        xarray Dataset containing Sentinel-2 optical bands and the SCL variable.
    valid_classes : sequence of int, optional
        List of SCL integer classes to retain as valid clear-sky observations.
        Defaults to [4, 5, 6] (Vegetation, Bare Soil, Water).
        Classes 0 (No Data) and 7 (Unclassified) are invalid by default.
    scl_var : str, default 'SCL'
        Name of the Scene Classification Layer variable in `ds_s2`.

    Returns
    -------
    xr.Dataset
        A copy of `ds_s2` with invalid pixels set to NaN across all reflectance variables.
        The SCL variable is preserved for downstream diagnostic accounting.

    Raises
    ------
    KeyError
        If `scl_var` is not found in `ds_s2.data_vars`.
    """
    if scl_var not in ds_s2.data_vars:
        raise KeyError(
            f"Scene Classification Layer variable '{scl_var}' not found in dataset data variables: "
            f"{list(ds_s2.data_vars)}"
        )

    if valid_classes is None:
        valid_classes = DEFAULT_VALID_SCL_CLASSES

    valid_set = set(valid_classes)
    scl_da = ds_s2[scl_var]

    # Create base boolean mask from SCL
    scl_valid = scl_da.isin(list(valid_set))

    ds_masked = ds_s2.copy(deep=True)

    for var_name in list(ds_masked.data_vars):
        if var_name == scl_var:
            continue

        var_da = ds_masked[var_name]

        # Check if spatial dimensions/coordinates match SCL
        if var_da.shape == scl_valid.shape:
            mask_aligned = scl_valid
        else:
            # Reindex / align 20m SCL mask to 10m optical grid using nearest neighbor
            try:
                mask_aligned = scl_valid.reindex_like(var_da, method="nearest")
            except Exception:
                mask_aligned = scl_valid.interp_like(var_da, method="nearest") > 0.5

        # Mask invalid pixels to NaN (ensuring float representation for NaN storage)
        if np.issubdtype(var_da.dtype, np.integer):
            masked_arr = var_da.astype(np.float32).where(mask_aligned, np.nan)
        else:
            masked_arr = var_da.where(mask_aligned, np.nan)

        # Preserve variable attributes
        masked_arr.attrs.update(var_da.attrs)
        masked_arr.attrs["scl_masked"] = True
        masked_arr.attrs["valid_scl_classes"] = list(valid_classes)
        ds_masked[var_name] = masked_arr

    return ds_masked
