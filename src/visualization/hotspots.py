"""Seasonal Hotspot and Spatial Persistence Heatmap Visualization.

Renders high-resolution spatial maps of actionable screening recurrence frequency
across the complete 15-bin canonical production season.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_spatial_persistence_heatmap(
    persistence_ds: xr.Dataset,
    aoi_name: str = "Moiben-Soy Agricultural Pilot Zone",
    season_name: str = "2023 Long Rains Season",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (9, 8),
    dpi: int = 300,
) -> plt.Figure:
    """Render a spatial heatmap of actionable diagnostic recurrence count across the season.

    Parameters
    ----------
    persistence_ds : xr.Dataset
        Output of calculate_spatial_persistence() containing 'actionable_recurrence_count'.
    aoi_name : str, default 'Moiben-Soy Agricultural Pilot Zone'
        Name of the agricultural study area.
    season_name : str, default '2023 Long Rains Season'
        Name of the production season.
    output_path : str or Path, optional
        Target file path for saving the figure.
    figsize : tuple of int, default (9, 8)
        Figure size in inches.
    dpi : int, default 300
        Output resolution.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_bins = int(persistence_ds.attrs.get("n_canonical_bins", 15))
    rec_count = persistence_ds["actionable_recurrence_count"].values

    fig.suptitle(
        f"Seasonal Diagnostic Hotspots & Spatial Persistence",
        fontsize=13, fontweight="bold", y=0.98
    )
    ax.set_title(
        f"{aoi_name} | {season_name} ({n_bins} Canonical 14-day Bins)",
        fontsize=9, style="italic", color="#444444", pad=14
    )

    cmap = plt.get_cmap("YlOrRd", n_bins + 1)
    im = ax.imshow(
        rec_count,
        cmap=cmap,
        vmin=0,
        vmax=n_bins,
        origin="upper",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Actionable Screening Recurrence (Number of 14-day Bins)", fontsize=9, fontweight="bold")
    cbar.set_ticks(np.arange(0, n_bins + 1, 2))

    ax.set_xlabel("Easting / Grid Columns (30 m Cells, EPSG:6933)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Northing / Grid Rows (30 m Cells, EPSG:6933)", fontsize=9, fontweight="bold")

    # Add statistics text box
    valid_px = rec_count[np.isfinite(rec_count)]
    total_px = len(valid_px)
    px_1plus = int(np.sum(valid_px >= 1))
    px_3plus = int(np.sum(valid_px >= 3))
    px_6plus = int(np.sum(valid_px >= 6))

    stats_text = (
        f"Persistence Statistics:\n"
        f"• Total Evaluation Pixels: {total_px:,}\n"
        f"• Recurrence >= 1 Bin: {px_1plus:,} ({px_1plus/total_px*100:.1f}%)\n"
        f"• Recurrence >= 3 Bins: {px_3plus:,} ({px_3plus/total_px*100:.1f}%)\n"
        f"• Recurrence >= 6 Bins: {px_6plus:,} ({px_6plus/total_px*100:.1f}%)"
    )
    ax.text(
        0.03, 0.04, stats_text,
        transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9)
    )

    footer = (
        "Notice: Hotspot recurrence reflects empirical frequency of screening flags (Case A, B, D, MULTI_SIGNAL) over time.\n"
        "Persistent signals indicate recurring anomalies under regional forcing, not confirmed chronic crop failure."
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=7.5, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig
