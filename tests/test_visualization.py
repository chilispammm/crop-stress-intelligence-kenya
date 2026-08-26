"""Unit tests for Milestone 6 visualization module."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.reporting.persistence import calculate_spatial_persistence
from src.visualization.hotspots import plot_spatial_persistence_heatmap
from src.visualization.profiles import (
    plot_diagnostic_area_timeline,
    plot_temporal_diagnostic_profile,
)
from src.visualization.spatial import plot_spatial_diagnostic_map


def test_plot_temporal_diagnostic_profile(tmp_path: Path):
    """Verify temporal diagnostic profile renders and saves correctly."""
    dates = pd.date_range("2023-03-01", periods=15, freq="14D")
    z_r = [0.97] * 3 + [1.40] * 2 + [0.97] * 2 + [1.00] * 2 + [1.47] * 2 + [1.33] * 3 + [1.30]
    z_ndvi = list(np.linspace(-1.5, 1.2, 8)) + list(np.linspace(1.0, -1.6, 7))
    swi = list(np.linspace(0.33, 0.55, 8)) + list(np.linspace(0.53, 0.34, 7))
    dswi = [float("nan")] + list(np.diff(swi))

    out_file = tmp_path / "profile.png"
    fig = plot_temporal_diagnostic_profile(
        bin_dates=dates,
        z_r_values=z_r,
        z_ndvi_values=z_ndvi,
        swi_values=swi,
        delta_swi_values=dswi,
        output_path=out_file,
    )

    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    assert out_file.stat().st_size > 5000
    plt.close(fig)


def test_plot_diagnostic_area_timeline(tmp_path: Path):
    """Verify diagnostic area timeline chart renders and saves correctly."""
    bin_summaries = [
        {"bin_start": "2023-03-01", "total_area_ha": 183.87, "dominant_case": "CASE_B", "case_b_pixels": 4431, "case_d_pixels": 0, "multi_signal_pixels": 0, "case_a_pixels": 0},
        {"bin_start": "2023-03-15", "total_area_ha": 0.0, "dominant_case": "NORMAL", "case_b_pixels": 0, "case_d_pixels": 0, "multi_signal_pixels": 0, "case_a_pixels": 0},
        {"bin_start": "2023-07-05", "total_area_ha": 503.10, "dominant_case": "CASE_D", "case_b_pixels": 0, "case_d_pixels": 5590, "multi_signal_pixels": 0, "case_a_pixels": 0},
    ]

    out_file = tmp_path / "timeline.png"
    fig = plot_diagnostic_area_timeline(
        bin_summaries=bin_summaries,
        output_path=out_file,
    )

    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    assert out_file.stat().st_size > 5000
    plt.close(fig)


def test_plot_spatial_diagnostic_map(tmp_path: Path):
    """Verify geospatial diagnostic map renders polygons and saves properly."""
    sample_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [35.18, 0.58],
                            [35.22, 0.58],
                            [35.22, 0.62],
                            [35.18, 0.62],
                            [35.18, 0.58],
                        ]
                    ],
                },
                "properties": {
                    "zone_id": "zone_001",
                    "case_type": "CASE_B",
                    "area_ha": 183.87,
                    "centroid_lat": 0.60,
                    "centroid_lon": 35.20,
                },
            }
        ],
        "pipeline_report": {
            "final_total_area_ha": 183.87,
        },
    }

    out_file = tmp_path / "spatial_map.png"
    fig = plot_spatial_diagnostic_map(
        geojson_data=sample_geojson,
        bin_date="2023-03-01",
        bin_index=1,
        aoi_bbox=[35.15, 0.55, 35.35, 0.75],
        grid_bbox=[35.24, 0.64, 35.26, 0.66],
        output_path=out_file,
    )

    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    assert out_file.stat().st_size > 5000
    plt.close(fig)


def test_plot_spatial_persistence_heatmap(tmp_path: Path):
    """Verify persistence heatmap renders and saves properly."""
    y = np.arange(10)
    x = np.arange(10)
    g = np.random.randint(0, 5, size=(10, 10))
    grid = xr.DataArray(g, coords={"y": y, "x": x}, dims=["y", "x"])
    persist_ds = calculate_spatial_persistence([grid, grid])

    out_file = tmp_path / "heatmap.png"
    fig = plot_spatial_persistence_heatmap(
        persistence_ds=persist_ds,
        output_path=out_file,
    )

    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    assert out_file.stat().st_size > 5000
    plt.close(fig)
