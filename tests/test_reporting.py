"""Unit tests for Milestone 6 reporting and spatial persistence modules."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

from src.diagnostics.screening import DiagnosticCase
from src.reporting.persistence import calculate_spatial_persistence
from src.reporting.summary import (
    export_seasonal_summary_json,
    export_seasonal_summary_markdown,
    generate_seasonal_executive_summary,
)


def _sample_bin_summaries():
    return [
        {
            "bin_index": 1,
            "bin_start": "2023-03-01",
            "bin_end": "2023-03-15",
            "cluster_count": 22,
            "total_area_ha": 183.87,
            "dominant_case": "CASE_B",
            "normal_pixels": 1159,
            "case_a_pixels": 0,
            "case_b_pixels": 4431,
            "case_c_pixels": 0,
            "case_d_pixels": 0,
            "multi_signal_pixels": 0,
            "insufficient_evidence_pixels": 0,
        },
        {
            "bin_index": 2,
            "bin_start": "2023-03-15",
            "bin_end": "2023-03-29",
            "cluster_count": 0,
            "total_area_ha": 0.0,
            "dominant_case": "NORMAL",
            "normal_pixels": 5590,
            "case_a_pixels": 0,
            "case_b_pixels": 0,
            "case_c_pixels": 0,
            "case_d_pixels": 0,
            "multi_signal_pixels": 0,
            "insufficient_evidence_pixels": 0,
        },
        {
            "bin_index": 10,
            "bin_start": "2023-07-05",
            "bin_end": "2023-07-19",
            "cluster_count": 1,
            "total_area_ha": 503.10,
            "dominant_case": "CASE_D",
            "normal_pixels": 0,
            "case_a_pixels": 0,
            "case_b_pixels": 0,
            "case_c_pixels": 0,
            "case_d_pixels": 5590,
            "multi_signal_pixels": 0,
            "insufficient_evidence_pixels": 0,
        },
    ]


def test_generate_seasonal_executive_summary():
    """Verify seasonal executive summary compiles required structure and counts."""
    bin_summaries = _sample_bin_summaries()
    bbox = [35.15, 0.55, 35.35, 0.75]

    summary = generate_seasonal_executive_summary(
        aoi_id="ug_pilot_moiben_01",
        aoi_name="Moiben-Soy Pilot Zone",
        bbox=bbox,
        season_name="2023 Long Rains",
        bin_summaries=bin_summaries,
    )

    assert "metadata" in summary
    assert "spatial_scope" in summary
    assert "temporal_scope" in summary
    assert "diagnostic_totals" in summary
    assert "pixel_classification_totals" in summary
    assert "scientific_caveats" in summary

    totals = summary["diagnostic_totals"]
    assert totals["total_extracted_clusters"] == 23
    assert totals["total_candidate_area_ha_cumulative"] == round(183.87 + 503.10, 2)
    assert totals["actionable_bin_count"] == 2
    assert totals["first_actionable_date"] == "2023-03-01"
    assert totals["last_actionable_date"] == "2023-07-05"
    assert totals["max_single_bin_area_ha"] == 503.10


def test_export_summary_json_and_markdown(tmp_path: Path):
    """Verify summary exporter writes valid JSON and Markdown files."""
    bin_summaries = _sample_bin_summaries()
    summary = generate_seasonal_executive_summary(
        aoi_id="ug_test_01",
        aoi_name="Test Zone",
        bbox=[35.15, 0.55, 35.35, 0.75],
        season_name="2023 Test Season",
        bin_summaries=bin_summaries,
    )

    json_path = tmp_path / "summary.json"
    md_path = tmp_path / "summary.md"

    out_json = export_seasonal_summary_json(summary, json_path)
    out_md = export_seasonal_summary_markdown(summary, md_path)

    assert out_json.exists()
    assert out_md.exists()

    with open(out_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert loaded["spatial_scope"]["aoi_id"] == "ug_test_01"

    with open(out_md, "r", encoding="utf-8") as f:
        content = f.read()
        assert "# Seasonal Diagnostic Screening Executive Summary" in content
        assert "503.10 ha" in content


def test_calculate_spatial_persistence():
    """Verify spatial persistence calculation correctly aggregates recurrence counts."""
    y = np.arange(10)
    x = np.arange(10)
    shape = (len(y), len(x))

    # Grid 1: All NORMAL except center 2x2 is CASE_B
    g1 = np.zeros(shape, dtype=np.int32)
    g1[4:6, 4:6] = DiagnosticCase.CASE_B.value

    # Grid 2: All NORMAL
    g2 = np.zeros(shape, dtype=np.int32)

    # Grid 3: All CASE_D
    g3 = np.full(shape, DiagnosticCase.CASE_D.value, dtype=np.int32)

    grids = [
        xr.DataArray(g1, coords={"y": y, "x": x}, dims=["y", "x"]),
        xr.DataArray(g2, coords={"y": y, "x": x}, dims=["y", "x"]),
        xr.DataArray(g3, coords={"y": y, "x": x}, dims=["y", "x"]),
    ]

    persist_ds = calculate_spatial_persistence(grids)

    assert "actionable_recurrence_count" in persist_ds
    assert "actionable_recurrence_freq" in persist_ds
    assert "case_b_recurrence_count" in persist_ds
    assert "case_d_recurrence_count" in persist_ds

    rec_count = persist_ds["actionable_recurrence_count"].values
    # Center 2x2 had Case B in g1 and Case D in g3 -> count = 2
    assert (rec_count[4:6, 4:6] == 2).all()
    # Outer pixels had Case D only in g3 -> count = 1
    assert rec_count[0, 0] == 1

    freq = persist_ds["actionable_recurrence_freq"].values
    assert np.allclose(freq[4:6, 4:6], 2 / 3, atol=1e-4)
    assert np.isclose(freq[0, 0], 1 / 3, atol=1e-4)


def test_calculate_spatial_persistence_empty_raises():
    """Verify calculate_spatial_persistence raises ValueError on empty list."""
    with pytest.raises(ValueError, match="empty sequence"):
        calculate_spatial_persistence([])
