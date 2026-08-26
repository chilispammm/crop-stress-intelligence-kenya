"""Temporal Diagnostic Profiles and Seasonal Timeline Visualizations.

Generates publication-quality time-series profiles of hydro-meteorological,
vegetation, and root-zone soil moisture anomalies across canonical 14-day production bins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASE_COLORS = {
    "NORMAL": "#2b83ba",
    "CASE_A": "#d7191c",
    "CASE_B": "#fdae61",
    "CASE_C": "#abd9e9",
    "CASE_D": "#e66101",
    "MULTI_SIGNAL": "#7b3294",
    "INSUFFICIENT_EVIDENCE": "#cccccc",
}


def plot_temporal_diagnostic_profile(
    bin_dates: Sequence[Union[str, pd.Timestamp]],
    z_r_values: Sequence[float],
    z_ndvi_values: Sequence[float],
    swi_values: Sequence[float],
    delta_swi_values: Sequence[float],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "2023 Long Rains Multi-Modal Diagnostic Anomaly Profile",
    subtitle: str = "Moiben-Soy Agricultural Pilot Zone, Uasin Gishu County, Kenya",
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300,
) -> plt.Figure:
    """Generate a 4-panel seasonal time-series profile of all diagnostic screening modalities.

    Parameters
    ----------
    bin_dates : sequence of str or pd.Timestamp
        14-day canonical bin start dates.
    z_r_values : sequence of float
        Monthly Standardized Precipitation Anomalies (CHIRPS Z_R).
    z_ndvi_values : sequence of float
        14-day Standardized Vegetation Anomalies (Sentinel-2 Z_NDVI).
    swi_values : sequence of float
        14-day Root-zone Soil Water Index (SMAP L4 s1).
    delta_swi_values : sequence of float
        14-day change in SWI (delta_swi_14d; Bin 1 is NaN/undefined).
    output_path : str or Path, optional
        Target file path for saving the figure.
    title : str, default '2023 Long Rains Multi-Modal Diagnostic Anomaly Profile'
        Main figure title.
    subtitle : str, default 'Moiben-Soy Agricultural Pilot Zone...'
        Figure subtitle.
    figsize : tuple of int, default (12, 10)
        Figure dimensions in inches.
    dpi : int, default 300
        Resolution for saved raster outputs.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered figure object.
    """
    dates = pd.to_datetime(list(bin_dates))
    n_bins = len(dates)
    x_indices = np.arange(n_bins)

    TITLE_Y = 0.985
    SUBTITLE_Y = 0.945
    PLOT_TOP = 0.91

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=TITLE_Y)
    fig.text(0.5, SUBTITLE_Y, subtitle, ha="center", fontsize=10.5, style="italic", color="#555555")

    # Panel 1: CHIRPS Z_R
    ax1 = axes[0]
    z_r_arr = np.array(z_r_values, dtype=np.float32)
    bars1 = ax1.bar(
        x_indices, z_r_arr, color=np.where(z_r_arr >= 0, "#2b83ba", "#d7191c"),
        alpha=0.75, width=0.6, edgecolor="#333333", linewidth=0.8
    )
    ax1.axhline(0.0, color="#333333", linestyle="-", linewidth=0.8)
    ax1.axhline(0.5, color="#e66101", linestyle="--", linewidth=1.2, label="Case D Threshold ($Z_R \\geq +0.5$)")
    ax1.axhline(-0.8, color="#d7191c", linestyle="--", linewidth=1.2, label="Case A/C Threshold ($Z_R \\leq -0.8$)")
    ax1.set_ylabel("CHIRPS $Z_R$\n(Monthly)", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Sentinel-2 Z_NDVI
    ax2 = axes[1]
    z_ndvi_arr = np.array(z_ndvi_values, dtype=np.float32)
    ax2.plot(x_indices, z_ndvi_arr, marker="o", color="#1a9641", linewidth=2.0, markersize=6, label="Mean $Z_{\\text{NDVI}}$")
    ax2.axhline(0.0, color="#333333", linestyle="-", linewidth=0.8)
    ax2.axhline(-1.0, color="#d7191c", linestyle="--", linewidth=1.2, label="Case A ($Z_{\\text{NDVI}} \\leq -1.0$)")
    ax2.axhline(-1.2, color="#fdae61", linestyle="--", linewidth=1.2, label="Case B ($Z_{\\text{NDVI}} \\leq -1.2$)")
    ax2.set_ylabel("Sentinel-2 $Z_{\\text{NDVI}}$\n(14-day Composite)", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Panel 3: SMAP L4 SWI (s1)
    ax3 = axes[2]
    swi_arr = np.array(swi_values, dtype=np.float32)
    ax3.plot(x_indices, swi_arr, marker="s", color="#0571b0", linewidth=2.0, markersize=6, label="Root-Zone SWI ($s_1$)")
    ax3.axhline(0.30, color="#ca0020", linestyle="--", linewidth=1.2, label="Non-Dry Prior ($\\mathrm{SWI} \\geq 0.30$)")
    ax3.set_ylabel("SMAP L4 SWI\n($0\\text{–}100\\text{ cm}$ Fraction)", fontsize=10, fontweight="bold")
    ax3.set_ylim(0.0, 1.0)
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax3.grid(True, linestyle=":", alpha=0.5)

    # Panel 4: SMAP L4 Delta SWI (14-day change)
    ax4 = axes[3]
    dswi_arr = np.array(delta_swi_values, dtype=np.float32)
    
    # Plot valid delta_swi values (bins 2..N)
    valid_mask = np.isfinite(dswi_arr)
    if np.any(valid_mask):
        ax4.plot(
            x_indices[valid_mask], dswi_arr[valid_mask],
            marker="^", color="#7b3294", linewidth=2.0, markersize=6, label="$\\Delta\\mathrm{SWI}_{14d}$"
        )
    
    # Explicit marker for Bin 1 (t=0 undefined, NOT missing)
    if not valid_mask[0]:
        ax4.scatter(
            [0], [0.0], marker="o", facecolors="none", edgecolors="#d7191c", s=100, linewidth=2.0,
            label="Bin 1: Undefined $t=0$ (No Prior Bin)", zorder=5
        )
        ax4.annotate(
            "Undefined (t=0)\n[Contractual]",
            xy=(0, 0.0), xytext=(0.4, 0.02),
            arrowprops=dict(arrowstyle="->", color="#d7191c", lw=1.0),
            fontsize=8, color="#d7191c", fontweight="bold"
        )

    ax4.axhline(0.0, color="#333333", linestyle="--", linewidth=1.2, label="Zero Change Line ($\\Delta\\mathrm{SWI} \\leq 0.0$)")
    ax4.set_ylabel("$\\Delta\\mathrm{SWI}_{14d}$\n(14-day Difference)", fontsize=10, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax4.grid(True, linestyle=":", alpha=0.5)

    # Configure x-axis formatting
    ax4.set_xticks(x_indices)
    ax4.set_xticklabels([f"{d.strftime('%b %d')}\n(Bin {i + 1})" for i, d in enumerate(dates)], rotation=0, fontsize=8)
    ax4.set_xlabel("Canonical 14-day Production Bins (2023 Long Rains Season)", fontsize=10, fontweight="bold")

    # Add explanatory footer
    footer_text = (
        "Note: CHIRPS (5.5 km) and SMAP L4 (9 km) provide regional contextual hydro-meteorology assigned via nearest neighbor;\n"
        "Sentinel-2 (30 m) captures intra-field vegetation response. Screening signals represent priority hypotheses, not proven crop damage."
    )
    fig.text(0.5, 0.01, footer_text, ha="center", fontsize=8, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.04, 1, PLOT_TOP])

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig


def plot_diagnostic_area_timeline(
    bin_summaries: Sequence[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Candidate Scouting Area Progression by Diagnostic Case",
    subtitle: str = "Moiben-Soy Agricultural Pilot Zone — 2023 Long Rains Season",
    figsize: Tuple[int, int] = (11, 6),
    dpi: int = 300,
) -> plt.Figure:
    """Generate a stacked bar chart of candidate scouting area (in hectares) across 15 canonical bins.

    Parameters
    ----------
    bin_summaries : sequence of dict
        List of per-bin summary dictionaries containing area and case breakdowns.
    output_path : str or Path, optional
        Target file path for saving the figure.
    title : str, default 'Candidate Scouting Area Progression...'
        Figure title.
    subtitle : str, default 'Moiben-Soy Agricultural Pilot Zone...'
        Figure subtitle.
    figsize : tuple of int, default (11, 6)
        Figure dimensions.
    dpi : int, default 300
        Resolution for saved raster output.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered figure object.
    """
    n_bins = len(bin_summaries)
    bin_dates = [pd.to_datetime(b["bin_start"]) for b in bin_summaries]
    x_indices = np.arange(n_bins)

    case_a_areas = np.zeros(n_bins, dtype=np.float32)
    case_b_areas = np.zeros(n_bins, dtype=np.float32)
    case_d_areas = np.zeros(n_bins, dtype=np.float32)
    multi_areas  = np.zeros(n_bins, dtype=np.float32)

    for i, b in enumerate(bin_summaries):
        tot_area = float(b.get("total_area_ha", 0.0))
        dom_case = b.get("dominant_case", "")

        # Compute proportionate area breakdown from pixel counts if available
        px_a = b.get("case_a_pixels", 0)
        px_b = b.get("case_b_pixels", 0)
        px_d = b.get("case_d_pixels", 0)
        px_m = b.get("multi_signal_pixels", 0)
        actionable_px = px_a + px_b + px_d + px_m

        if actionable_px > 0 and tot_area > 0:
            case_a_areas[i] = tot_area * (px_a / actionable_px)
            case_b_areas[i] = tot_area * (px_b / actionable_px)
            case_d_areas[i] = tot_area * (px_d / actionable_px)
            multi_areas[i]  = tot_area * (px_m / actionable_px)
        elif tot_area > 0:
            if "CASE_B" in dom_case:
                case_b_areas[i] = tot_area
            elif "CASE_D" in dom_case:
                case_d_areas[i] = tot_area
            elif "MULTI" in dom_case:
                multi_areas[i] = tot_area

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    ax.set_title(subtitle, fontsize=10, style="italic", color="#444444", pad=15)

    p1 = ax.bar(x_indices, case_b_areas, width=0.6, label="Case B: Non-Dry Anomaly", color=CASE_COLORS["CASE_B"], edgecolor="#333", lw=0.7)
    p2 = ax.bar(x_indices, case_d_areas, bottom=case_b_areas, width=0.6, label="Case D: Hydrological Disconnect", color=CASE_COLORS["CASE_D"], edgecolor="#333", lw=0.7)
    p3 = ax.bar(x_indices, multi_areas, bottom=case_b_areas + case_d_areas, width=0.6, label="MULTI_SIGNAL: B+D Overlap", color=CASE_COLORS["MULTI_SIGNAL"], edgecolor="#333", lw=0.7)
    p4 = ax.bar(x_indices, case_a_areas, bottom=case_b_areas + case_d_areas + multi_areas, width=0.6, label="Case A: Drought Stress", color=CASE_COLORS["CASE_A"], edgecolor="#333", lw=0.7)

    # Set data-driven y-axis upper bound with sufficient headroom for labels
    total_seasonal_areas = case_b_areas + case_d_areas + multi_areas + case_a_areas
    max_tot = float(np.max(total_seasonal_areas)) if len(total_seasonal_areas) > 0 else 0.0
    y_top = max(max_tot * 1.25, 100.0)
    ax.set_ylim(0.0, y_top)

    ax.set_ylabel("Candidate Scouting Area (ha)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Canonical 14-day Production Bins (2023 Season)", fontsize=10, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{d.strftime('%b %d')}\n(Bin {i + 1})" for i, d in enumerate(bin_dates)], fontsize=8)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Annotate total area above actionable bars with headroom
    for i, tot in enumerate(total_seasonal_areas):
        if tot > 0:
            ax.text(i, tot + (y_top * 0.02), f"{tot:.1f} ha", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig
