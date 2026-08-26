"""Seasonal Executive Summary Generator for Crop Stress Intelligence Kenya.

Produces machine-readable JSON and publication-grade Markdown summaries
synthesizing multi-modal diagnostic screening results across all 15 canonical bins.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


def generate_seasonal_executive_summary(
    aoi_id: str,
    aoi_name: str,
    bbox: Sequence[float],
    season_name: str,
    bin_summaries: Sequence[Dict[str, Any]],
    soil_moisture_source: str = "NASA SMAP Level-4 (SPL4SMGP V008)",
    optical_source: str = "Digital Earth Africa Sentinel-2 L2A",
    rainfall_source: str = "CHIRPS Monthly Climatology (1991-2020 Baseline)",
) -> Dict[str, Any]:
    """Generate a comprehensive seasonal executive summary dictionary from per-bin outputs.

    Parameters
    ----------
    aoi_id : str
        Unique AOI identifier (e.g. 'ug_pilot_moiben_01').
    aoi_name : str
        Human-readable AOI name (e.g. 'Moiben-Soy Agricultural Pilot Zone').
    bbox : sequence of float
        Bounding box [min_lon, min_lat, max_lon, max_lat] in EPSG:4326.
    season_name : str
        Name of the seasonal production window (e.g. '2023 Long Rains').
    bin_summaries : sequence of dict
        Per-bin diagnostic summary dictionaries containing case counts, cluster counts, and areas.
    soil_moisture_source : str, default 'NASA SMAP Level-4 (SPL4SMGP V008)'
        Hydrology source description.
    optical_source : str, default 'Digital Earth Africa Sentinel-2 L2A'
        Vegetation source description.
    rainfall_source : str, default 'CHIRPS Monthly Climatology (1991-2020 Baseline)'
        Rainfall source description.

    Returns
    -------
    dict
        Machine-readable summary dictionary.
    """
    n_bins = len(bin_summaries)
    if n_bins == 0:
        raise ValueError("Cannot generate seasonal summary from empty bin_summaries list.")

    bin_dates = [b["bin_start"] for b in bin_summaries]
    season_start = str(bin_dates[0])
    season_end = str(bin_summaries[-1].get("bin_end", bin_dates[-1]))

    total_clusters = sum(b.get("cluster_count", 0) for b in bin_summaries)
    total_area_ha = sum(b.get("total_area_ha", 0.0) for b in bin_summaries)

    total_normal_px = sum(b.get("normal_pixels", 0) for b in bin_summaries)
    total_case_a_px = sum(b.get("case_a_pixels", 0) for b in bin_summaries)
    total_case_b_px = sum(b.get("case_b_pixels", 0) for b in bin_summaries)
    total_case_c_px = sum(b.get("case_c_pixels", 0) for b in bin_summaries)
    total_case_d_px = sum(b.get("case_d_pixels", 0) for b in bin_summaries)
    total_multi_px  = sum(b.get("multi_signal_pixels", 0) for b in bin_summaries)
    total_insufficient_px = sum(b.get("insufficient_evidence_pixels", 0) for b in bin_summaries)

    actionable_bins = []
    max_area_ha = 0.0
    max_area_bin = None

    for idx, b in enumerate(bin_summaries):
        area = b.get("total_area_ha", 0.0)
        clusters = b.get("cluster_count", 0)
        if clusters > 0 or area > 0.0:
            actionable_bins.append({
                "bin_index": idx + 1,
                "bin_start": str(b["bin_start"]),
                "cluster_count": clusters,
                "area_ha": round(float(area), 2),
                "dominant_case": b.get("dominant_case", "UNKNOWN"),
            })
            if area > max_area_ha:
                max_area_ha = float(area)
                max_area_bin = idx + 1

    first_actionable_date = actionable_bins[0]["bin_start"] if actionable_bins else None
    last_actionable_date = actionable_bins[-1]["bin_start"] if actionable_bins else None

    summary: Dict[str, Any] = {
        "metadata": {
            "title": "Crop Stress Intelligence Kenya — Seasonal Diagnostic Summary",
            "report_version": "1.0.0",
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "software_version": "0.4.0",
        },
        "spatial_scope": {
            "aoi_id": str(aoi_id),
            "aoi_name": str(aoi_name),
            "bbox_wgs84": list(bbox),
            "crs_optical": "EPSG:32736",
            "crs_diagnostic": "EPSG:6933",
            "mmu_threshold_ha": 2.0,
        },
        "temporal_scope": {
            "season_name": str(season_name),
            "season_start": season_start,
            "season_end": season_end,
            "n_canonical_bins": n_bins,
            "bin_interval_days": 14,
        },
        "diagnostic_totals": {
            "total_extracted_clusters": int(total_clusters),
            "total_candidate_area_ha_cumulative": round(float(total_area_ha), 2),
            "actionable_bin_count": len(actionable_bins),
            "first_actionable_date": first_actionable_date,
            "last_actionable_date": last_actionable_date,
            "max_single_bin_area_ha": round(float(max_area_ha), 2),
            "max_area_bin_index": max_area_bin,
        },
        "pixel_classification_totals": {
            "normal": int(total_normal_px),
            "case_a_drought_stress": int(total_case_a_px),
            "case_b_vegetation_anomaly_non_dry": int(total_case_b_px),
            "case_c_drought_meteorological": int(total_case_c_px),
            "case_d_hydrological_disconnect": int(total_case_d_px),
            "multi_signal_overlap": int(total_multi_px),
            "insufficient_evidence": int(total_insufficient_px),
        },
        "actionable_bins_detail": actionable_bins,
        "provenance_and_sources": {
            "soil_moisture_source": str(soil_moisture_source),
            "optical_vegetation_source": str(optical_source),
            "rainfall_climatology_source": str(rainfall_source),
            "diagnostic_matrix_version": "M5.0 Frozen Matrix (ADR-021)",
        },
        "scientific_caveats": [
            "Candidate scouting zones represent priority screening hypotheses to guide field verification, NOT confirmed yield loss or crop damage.",
            "Case B reflects vegetation anomalies under non-dry conditions; it does NOT prove insect pests, fungal pathogens, or specific disease presence.",
            "Case D reflects above-average rainfall coincident with non-increasing root-zone wetness; it does NOT prove soil crusting, runoff failure, or infiltration impedance.",
            "Spatial Scale Disparity Limitation: CHIRPS (5.5 km) and SMAP (9 km) data provide coarse regional forcing broadcast to the 30 m grid via nearest neighbor. Over the 503.10 ha focal block, regional hydro-meteorological conditions are spatially uniform.",
            "Evaluated Spatial Extent: The 503.10 ha (86x65, 30 m) grid represents the intentional frozen focal pilot evaluation block within the broader 490.5 km² regional pilot AOI.",
            "SWI >= 0.30 and Z_R >= 0.50 are operational screening priors, not empirically calibrated in-situ agronomic adequacy thresholds.",
        ],
    }
    return summary


def export_seasonal_summary_json(summary: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Export seasonal executive summary dictionary to formatted JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return p


def export_seasonal_summary_markdown(summary: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Export seasonal executive summary to publication-grade Markdown document."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    meta = summary["metadata"]
    spatial = summary["spatial_scope"]
    temporal = summary["temporal_scope"]
    diag = summary["diagnostic_totals"]
    px = summary["pixel_classification_totals"]
    prov = summary["provenance_and_sources"]
    caveats = summary["scientific_caveats"]

    lines = [
        f"# Seasonal Diagnostic Screening Executive Summary",
        f"",
        f"**Target Season**: {temporal['season_name']} (`{temporal['season_start']}` to `{temporal['season_end']}`)  ",
        f"**Agricultural Zone**: {spatial['aoi_name']} (`{spatial['aoi_id']}`)  ",
        f"**Bounding Box (WGS84)**: `{spatial['bbox_wgs84']}`  ",
        f"**Generated UTC**: `{meta['generated_utc']}`  ",
        f"**System Version**: `{meta['software_version']}` | **Matrix Version**: `{prov['diagnostic_matrix_version']}`  ",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary & Headline Indicators",
        f"",
        f"| Headline Indicator | Seasonal Production Value | Operational Interpretation |",
        f"| :--- | :---: | :--- |",
        f"| **Canonical Evaluation Bins** | **{temporal['n_canonical_bins']}** (14-day intervals) | Full contiguous seasonal coverage with zero gaps |",
        f"| **Actionable Bins** | **{diag['actionable_bin_count']}** of {temporal['n_canonical_bins']} bins | Bins exhibiting MMU-filtered candidate scouting clusters |",
        f"| **Total Extracted Clusters** | **{diag['total_extracted_clusters']}** | Distinct $\\ge 2.0\\text{{ ha}}$ connected candidate polygons |",
        f"| **Cumulative Candidate Area** | **{diag['total_candidate_area_ha_cumulative']:.2f} ha** | Sum of actionable polygon areas across all bins |",
        f"| **Maximum Single-Bin Area** | **{diag['max_single_bin_area_ha']:.2f} ha** (Bin {diag['max_area_bin_index']}) | Peak spatial extent of screening flags |",
        f"| **First Actionable Date** | `{diag['first_actionable_date']}` | Early-season negative vegetation anomaly screening |",
        f"| **Last Actionable Date** | `{diag['last_actionable_date']}` | Late-season precipitation-to-moisture rate disconnect |",
        f"",
        f"---",
        f"",
        f"## 2. Seasonal Pixel-Level Classification Distribution",
        f"",
        f"| Diagnostic Category | Pixel Count | % of Season | Screening Semantic |",
        f"| :--- | :---: | :---: | :--- |",
        f"| **NORMAL (0)** | {px['normal']:,} | {px['normal'] / sum(px.values()) * 100:.1f}% | Canopy greenness and moisture within expected climatological ranges |",
        f"| **CASE A (1) — Coincident Precipitation & Vegetation Deficit** | {px['case_a_drought_stress']:,} | {px['case_a_drought_stress'] / sum(px.values()) * 100:.1f}% | Coincident precipitation deficit ($Z_R \\le -0.8$) and vegetation anomaly ($Z_{{\\text{{NDVI}}}} \\le -1.0$) |",
        f"| **CASE B (2) — Vegetation Anomaly Under Non-Dry Priors** | {px['case_b_vegetation_anomaly_non_dry']:,} | {px['case_b_vegetation_anomaly_non_dry'] / sum(px.values()) * 100:.1f}% | Vegetation anomaly ($Z_{{\\text{{NDVI}}}} \\le -1.2$) under non-dry hydro-meteorology ($Z_R \\ge 0.0, \\text{{SWI}} \\ge 0.30$) |",
        f"| **CASE C (3) — Precipitation Deficit Without Vegetation Anomaly** | {px['case_c_drought_meteorological']:,} | {px['case_c_drought_meteorological'] / sum(px.values()) * 100:.1f}% | Precipitation deficit ($Z_R \\le -0.8$) without vegetation anomaly (non-actionable) |",
        f"| **CASE D (4) — Precipitation-to-Moisture Disconnect** | {px['case_d_hydrological_disconnect']:,} | {px['case_d_hydrological_disconnect'] / sum(px.values()) * 100:.1f}% | Above-normal rainfall ($Z_R \\ge 0.5$) with non-increasing root-zone wetness ($\\Delta\\text{{SWI}}_{{14d}} \\le 0.0$) |",
        f"| **MULTI_SIGNAL (5) — B+D Co-occurrence** | {px['multi_signal_overlap']:,} | {px['multi_signal_overlap'] / sum(px.values()) * 100:.1f}% | Simultaneous non-dry canopy anomaly and hydrological rate-of-change disconnect |",
        f"| **INSUFFICIENT_EVIDENCE (-1)** | {px['insufficient_evidence']:,} | {px['insufficient_evidence'] / sum(px.values()) * 100:.1f}% | Cloud-obscured or unobserved pixels (default safe state) |",
        f"",
        f"---",
        f"",
        f"## 3. Actionable Scouting Zones Breakdown",
        f"",
        f"| Bin # | Bin Start | Clusters | Candidate Area (ha) | Dominant Case | Operational Priority |",
        f"| :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for b in summary["actionable_bins_detail"]:
        lines.append(
            f"| **{b['bin_index']}** | `{b['bin_start']}` | {b['cluster_count']} | {b['area_ha']:.2f} | `{b['dominant_case']}` | High Priority Scouting |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 4. Multi-Modal Provenance & Data Sources",
        f"",
        f"- **Hydrology / Root-Zone Soil Moisture**: {prov['soil_moisture_source']}",
        f"- **Optical Vegetation Dynamics**: {prov['optical_vegetation_source']}",
        f"- **Rainfall Climatology**: {prov['rainfall_climatology_source']}",
        f"- **Diagnostic Grid Resolution**: 30 m equal-area projection (`EPSG:6933`)",
        f"- **Minimum Mapping Unit (MMU)**: $\\ge 2.0\\text{{ ha}}$ ($20,000\\text{{ m}}^2$)",
        f"",
        f"---",
        f"",
        f"## 5. Non-Overclaiming Scientific Caveats & Operating Limits",
        f"",
    ])

    for i, c in enumerate(caveats, 1):
        lines.append(f"{i}. {c}")

    lines.append("")

    content = "\n".join(lines)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return p
