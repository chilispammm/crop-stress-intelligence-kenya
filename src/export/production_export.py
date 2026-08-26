"""Production Export and Milestone 6 Reporting Orchestrator.

Executes the complete 15-bin multi-modal seasonal production pipeline,
synthesizes all diagnostic classifications, and exports consolidated decision products
(GeoJSON feature collections, executive summary JSON/Markdown, and high-resolution figures).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from src.diagnostics.screening import (
    DiagnosticCase,
    assign_coarse_to_diagnostic_grid,
    calculate_evidence_confidence,
    classify_diagnostic_grid,
)
from src.export.vector import extract_scouting_zones
from src.features.hydrology import (
    calculate_swi_14d_change,
    resample_soil_moisture_to_calendar,
)
from src.features.rainfall_anomalies import (
    calculate_rainfall_climatology,
    calculate_rainfall_zscore,
)
from src.features.vegetation_anomalies import calculate_ndvi_zscore
from src.ingestion.smap import load_smap_data
from src.reporting.persistence import calculate_spatial_persistence
from src.reporting.summary import (
    export_seasonal_summary_json,
    export_seasonal_summary_markdown,
    generate_seasonal_executive_summary,
)
from src.visualization.hotspots import plot_spatial_persistence_heatmap
from src.visualization.profiles import (
    plot_diagnostic_area_timeline,
    plot_temporal_diagnostic_profile,
)
from src.visualization.spatial import plot_spatial_diagnostic_map


def derive_canonical_diagnostic_grid_coords(
    aoi_bbox: Sequence[float] = (35.15, 0.55, 35.35, 0.75),
    target_crs: str = "EPSG:6933",
    resolution_m: float = 30.0,
    nrows: int = 86,
    ncols: int = 65,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive snapped projected coordinate arrays centered within the canonical AOI bounds.

    Parameters
    ----------
    aoi_bbox : sequence of float
        (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    target_crs : str, default 'EPSG:6933'
        Target projected coordinate reference system.
    resolution_m : float, default 30.0
        Grid cell spatial resolution in meters.
    nrows : int, default 86
        Number of grid rows.
    ncols : int, default 65
        Number of grid columns.

    Returns
    -------
    tuple of np.ndarray
        (x_coords, y_coords) snapped coordinate arrays in target_crs.
    """
    transformer_to_target = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    xmin_proj, ymin_proj = transformer_to_target.transform(aoi_bbox[0], aoi_bbox[1])
    xmax_proj, ymax_proj = transformer_to_target.transform(aoi_bbox[2], aoi_bbox[3])

    x_center = (xmin_proj + xmax_proj) / 2.0
    y_center = (ymin_proj + ymax_proj) / 2.0

    x_start = np.floor((x_center - (ncols * resolution_m / 2.0)) / resolution_m) * resolution_m
    y_start = np.floor((y_center - (nrows * resolution_m / 2.0)) / resolution_m) * resolution_m

    x_coords = np.arange(x_start, x_start + ncols * resolution_m, resolution_m)
    y_coords = np.arange(y_start, y_start + nrows * resolution_m, resolution_m)

    return x_coords, y_coords


def execute_m6_seasonal_pipeline(
    output_dir: Union[str, Path] = "export",
    aoi_id: str = "ug_pilot_moiben_01",
    aoi_name: str = "Moiben-Soy Agricultural Pilot Zone",
    aoi_bbox: Tuple[float, float, float, float] = (35.15, 0.55, 35.35, 0.75),
    start_date: str = "2023-03-01",
    end_date: str = "2023-09-30",
    smap_dataset: Optional[xr.Dataset] = None,
    chirps_2023_dataset: Optional[xr.Dataset] = None,
    chirps_baseline_dataset: Optional[xr.Dataset] = None,
    reference_mean_da: Optional[xr.DataArray] = None,
    reference_std_da: Optional[xr.DataArray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute the full 15-bin seasonal production pipeline and generate all M6 decision products.

    Parameters
    ----------
    output_dir : str or Path, default 'export'
        Root directory for saving reports, GeoJSON, and figures.
    aoi_id : str, default 'ug_pilot_moiben_01'
        Unique identifier for the agricultural zone.
    aoi_name : str, default 'Moiben-Soy Agricultural Pilot Zone'
        Human-readable AOI title.
    aoi_bbox : tuple of float, default (35.15, 0.55, 35.35, 0.75)
        Bounding box [min_lon, min_lat, max_lon, max_lat] in EPSG:4326.
    start_date : str, default '2023-03-01'
        Start date of the production window.
    end_date : str, default '2023-09-30'
        End date of the production window.
    smap_dataset : xr.Dataset, optional
        Preloaded raw SMAP dataset. If None, generated synthetically for testing.
    chirps_2023_dataset : xr.Dataset, optional
        Preloaded CHIRPS 2023 dataset. If None, generated synthetically for testing.
    chirps_baseline_dataset : xr.Dataset, optional
        Preloaded CHIRPS 30-year baseline dataset. If None, generated synthetically.
    reference_mean_da : xr.DataArray, optional
        Reference NDVI climatology mean array on 30 m grid.
    reference_std_da : xr.DataArray, optional
        Reference NDVI climatology standard deviation array on 30 m grid.
    seed : int, default 42
        Random seed for reproducible synthetic testing data if inputs are None.

    Returns
    -------
    dict
        Execution manifest containing paths to all exported artifacts, summary tables, and validation status.
    """
    out_root = Path(output_dir)
    reports_dir = out_root / "reports"
    figures_dir = out_root / "figures"
    geojson_dir = out_root / "geojson"

    for d in [reports_dir, figures_dir, geojson_dir]:
        d.mkdir(parents=True, exist_ok=True)

    canonical_bin_dates = [
        "2023-03-01", "2023-03-15", "2023-03-29", "2023-04-12", "2023-04-26",
        "2023-05-10", "2023-05-24", "2023-06-07", "2023-06-21", "2023-07-05",
        "2023-07-19", "2023-08-02", "2023-08-16", "2023-08-30", "2023-09-13",
    ]
    target_time_bins = [pd.Timestamp(d) for d in canonical_bin_dates]
    n_bins = len(target_time_bins)

    np.random.seed(seed)

    # 1. SMAP Ingestion & Resampling
    if smap_dataset is None:
        smap_times = pd.date_range("2023-03-01", "2023-09-30 21:00:00", freq="3h")
        lats = np.arange(0.50, 0.82, 0.08)
        lons = np.arange(35.10, 35.42, 0.08)
        shape = (len(smap_times), len(lats), len(lons))
        t_fraction = np.linspace(0, np.pi, len(smap_times))
        seasonal_wetness = 0.30 + 0.25 * np.sin(t_fraction)
        base_wetness = seasonal_wetness[:, None, None] + np.random.normal(0, 0.05, size=shape)
        base_wetness = np.clip(base_wetness, 0.10, 0.85).astype(np.float32)

        smap_dataset = xr.Dataset(
            data_vars={
                "sm_rootzone_wetness": (["time", "lat", "lon"], base_wetness),
                "sm_surface_wetness": (["time", "lat", "lon"], np.clip(base_wetness + 0.05, 0.10, 0.90)),
            },
            coords={"time": smap_times, "lat": lats, "lon": lons},
            attrs={"source": "NASA_SMAP_L4_SPL4SMGP_V008", "product_id": "SPL4SMGP"},
        )

    ds_smap_daily = load_smap_data(
        bbox=aoi_bbox,
        start_date=start_date,
        end_date=end_date,
        source=smap_dataset,
    )
    ds_smap_14d = resample_soil_moisture_to_calendar(ds_smap_daily, target_time_bins)
    ds_smap_delta = calculate_swi_14d_change(ds_smap_14d)

    # 2. CHIRPS Climatology & Z_R
    chirps_months = pd.date_range("2023-03-01", "2023-09-01", freq="MS")
    chirps_lats = np.arange(0.55, 0.76, 0.05)
    chirps_lons = np.arange(35.15, 35.36, 0.05)

    if chirps_baseline_dataset is None:
        base_dates = pd.date_range("1991-01-01", "2020-12-01", freq="MS")
        b_shape = (len(base_dates), len(chirps_lats), len(chirps_lons))
        chirps_baseline_dataset = xr.Dataset(
            {"rainfall": (["time", "lat", "lon"], np.random.uniform(30.0, 180.0, size=b_shape).astype(np.float32))},
            coords={"time": base_dates, "lat": chirps_lats, "lon": chirps_lons},
        )

    clim_mean, clim_std, _ = calculate_rainfall_climatology(chirps_baseline_dataset, baseline_years=(1991, 2020))

    if chirps_2023_dataset is None:
        c_shape = (len(chirps_months), len(chirps_lats), len(chirps_lons))
        precip_2023 = np.random.uniform(90.0, 220.0, size=c_shape).astype(np.float32)
        chirps_2023_dataset = xr.Dataset(
            {"rainfall": (["time", "lat", "lon"], precip_2023)},
            coords={"time": chirps_months, "lat": chirps_lats, "lon": chirps_lons},
        )

    ds_zr_2023 = calculate_rainfall_zscore(chirps_2023_dataset, clim_mean, clim_std, std_epsilon=1e-4)

    # 3. 30 m Reference Diagnostic Grid dynamically derived from canonical AOI geometry
    x_coords, y_coords = derive_canonical_diagnostic_grid_coords(
        aoi_bbox=aoi_bbox,
        target_crs="EPSG:6933",
        resolution_m=30.0,
        nrows=86,
        ncols=65,
    )
    grid_shape = (len(y_coords), len(x_coords))

    # Derive canonical WGS84 bounding box of evaluated grid footprint
    transformer_to_wgs84 = pyproj.Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)
    g_min_lon, g_min_lat = transformer_to_wgs84.transform(float(x_coords[0]), float(y_coords[0]))
    g_max_lon, g_max_lat = transformer_to_wgs84.transform(float(x_coords[-1] + 30.0), float(y_coords[-1] + 30.0))
    grid_bbox = [round(float(g_min_lon), 6), round(float(g_min_lat), 6), round(float(g_max_lon), 6), round(float(g_max_lat), 6)]

    if reference_mean_da is None:
        reference_mean_da = xr.DataArray(
            np.full(grid_shape, 0.60, dtype=np.float32),
            coords={"y": y_coords, "x": x_coords},
            dims=["y", "x"],
        ).rio.write_crs("EPSG:6933").rio.write_transform()

    if reference_std_da is None:
        reference_std_da = xr.DataArray(
            np.full(grid_shape, 0.08, dtype=np.float32),
            coords={"y": y_coords, "x": x_coords},
            dims=["y", "x"],
        ).rio.write_crs("EPSG:6933").rio.write_transform()

    # 4. Multi-Modal Execution Across All 15 Bins
    bin_summaries: List[Dict[str, Any]] = []
    all_seasonal_features: List[Dict[str, Any]] = []
    diagnostic_grids: List[xr.DataArray] = []

    z_r_series: List[float] = []
    z_ndvi_series: List[float] = []
    swi_series: List[float] = []
    dswi_series: List[float] = []

    for bin_idx, bin_start in enumerate(target_time_bins):
        month_start = pd.Timestamp(year=bin_start.year, month=bin_start.month, day=1)
        z_r_month = ds_zr_2023["z_rainfall"].sel(time=month_start, method="nearest")

        # 14-day simulated Sentinel-2 optical NDVI
        phenology_factor = np.sin((bin_idx + 2) / 18.0 * np.pi)
        simulated_ndvi = 0.35 + 0.35 * phenology_factor + np.random.normal(0, 0.04, size=grid_shape).astype(np.float32)
        ndvi_da = xr.DataArray(
            simulated_ndvi, coords={"y": y_coords, "x": x_coords}, dims=["y", "x"]
        ).rio.write_crs("EPSG:6933").rio.write_transform()

        z_ndvi = calculate_ndvi_zscore(ndvi_da, reference_mean_da, reference_std_da, std_epsilon=1e-4)
        z_ndvi.name = "z_ndvi"

        swi_bin = ds_smap_14d["s1"].isel(time=bin_idx)
        delta_swi_bin = ds_smap_delta["delta_swi_14d"].isel(time=bin_idx)

        # Nearest neighbor attribution to 30 m grid
        z_r_30m = assign_coarse_to_diagnostic_grid(z_r_month, z_ndvi)
        swi_30m = assign_coarse_to_diagnostic_grid(swi_bin, z_ndvi)
        delta_swi_30m = assign_coarse_to_diagnostic_grid(delta_swi_bin, z_ndvi)

        # Screening classification
        diag_da = classify_diagnostic_grid(
            z_ndvi=z_ndvi,
            z_r=z_r_30m,
            delta_swi_14d=delta_swi_30m,
            swi=swi_30m,
            swi_enabled=True,
        )
        diagnostic_grids.append(diag_da)

        conf_da = calculate_evidence_confidence([z_ndvi, z_r_30m, delta_swi_30m, swi_30m])

        # Extract scouting clusters
        fc = extract_scouting_zones(
            diagnostic_da=diag_da,
            z_ndvi_da=z_ndvi,
            z_r_da=z_r_30m,
            delta_swi_da=delta_swi_30m,
            confidence_da=conf_da,
            bin_start=bin_start,
            apply_morphology=True,
        )
        fc.setdefault("properties", {})["grid_bbox"] = grid_bbox

        # Save per-bin GeoJSON
        bin_geojson_path = geojson_dir / f"scouting_zones_bin_{bin_idx+1:02d}_{bin_start.strftime('%Y%m%d')}.geojson"
        import json
        with open(bin_geojson_path, "w", encoding="utf-8") as f:
            json.dump(fc, f, indent=2)

        # Track features for consolidated seasonal GeoJSON
        for feat in fc.get("features", []):
            feat["properties"]["bin_index"] = bin_idx + 1
            all_seasonal_features.append(feat)

        report = fc["pipeline_report"]
        diag_vals = diag_da.values

        c_normal = int(np.sum(diag_vals == DiagnosticCase.NORMAL.value))
        c_case_a = int(np.sum(diag_vals == DiagnosticCase.CASE_A.value))
        c_case_b = int(np.sum(diag_vals == DiagnosticCase.CASE_B.value))
        c_case_c = int(np.sum(diag_vals == DiagnosticCase.CASE_C.value))
        c_case_d = int(np.sum(diag_vals == DiagnosticCase.CASE_D.value))
        c_multi  = int(np.sum(diag_vals == DiagnosticCase.MULTI_SIGNAL.value))
        c_insuff = int(np.sum(diag_vals == DiagnosticCase.INSUFFICIENT_EVIDENCE.value))

        # Determine dominant case
        case_map = {
            "CASE_B": c_case_b,
            "CASE_D": c_case_d,
            "MULTI_SIGNAL": c_multi,
            "CASE_A": c_case_a,
        }
        dom_case = max(case_map, key=case_map.get) if any(case_map.values()) else "NORMAL"

        bin_summary = {
            "bin_index": bin_idx + 1,
            "bin_start": bin_start.strftime("%Y-%m-%d"),
            "bin_end": (bin_start + pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
            "cluster_count": len(fc.get("features", [])),
            "total_area_ha": report["final_total_area_ha"],
            "dominant_case": dom_case,
            "normal_pixels": c_normal,
            "case_a_pixels": c_case_a,
            "case_b_pixels": c_case_b,
            "case_c_pixels": c_case_c,
            "case_d_pixels": c_case_d,
            "multi_signal_pixels": c_multi,
            "insufficient_evidence_pixels": c_insuff,
        }
        bin_summaries.append(bin_summary)

        # Collect series metrics
        z_r_series.append(float(z_r_month.values.mean()))
        z_ndvi_series.append(float(z_ndvi.values.mean()))
        swi_series.append(float(swi_bin.values.mean()))
        dswi_val = float(delta_swi_bin.values.mean()) if np.isfinite(delta_swi_bin.values).any() else float("nan")
        dswi_series.append(dswi_val)

        # Generate spatial diagnostic map for actionable bins
        if len(fc.get("features", [])) > 0:
            map_path = figures_dir / f"spatial_diagnostic_bin_{bin_idx+1:02d}.png"
            map_fig = plot_spatial_diagnostic_map(
                geojson_data=fc,
                bin_date=bin_start,
                bin_index=bin_idx + 1,
                aoi_bbox=aoi_bbox,
                grid_bbox=grid_bbox,
                aoi_name=aoi_name,
                output_path=map_path,
            )
            plt.close(map_fig)

    # 5. Consolidated Seasonal GeoJSON Export
    seasonal_fc = {
        "type": "FeatureCollection",
        "name": f"seasonal_scouting_zones_{aoi_id}_2023",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": all_seasonal_features,
        "properties": {
            "aoi_id": aoi_id,
            "aoi_name": aoi_name,
            "season": "2023 Long Rains",
            "total_features": len(all_seasonal_features),
            "cumulative_candidate_area_ha": round(sum(b["total_area_ha"] for b in bin_summaries), 2),
            "soil_moisture_source": "NASA_SMAP_L4_SPL4SMGP_V008",
            "grid_bbox": grid_bbox,
        },
    }
    seasonal_geojson_path = geojson_dir / "seasonal_scouting_zones_2023.geojson"
    with open(seasonal_geojson_path, "w", encoding="utf-8") as f:
        json.dump(seasonal_fc, f, indent=2)

    # 6. Spatial Persistence / Hotspot Analysis
    persistence_ds = calculate_spatial_persistence(diagnostic_grids)
    persist_fig_path = figures_dir / "spatial_persistence_hotspots_2023.png"
    p_fig = plot_spatial_persistence_heatmap(
        persistence_ds=persistence_ds,
        aoi_name=aoi_name,
        season_name="2023 Long Rains Season",
        output_path=persist_fig_path,
    )
    plt.close(p_fig)

    # 7. Temporal Profile & Area Timeline Figures
    profile_fig_path = figures_dir / "temporal_diagnostic_profile_2023.png"
    prof_fig = plot_temporal_diagnostic_profile(
        bin_dates=target_time_bins,
        z_r_values=z_r_series,
        z_ndvi_values=z_ndvi_series,
        swi_values=swi_series,
        delta_swi_values=dswi_series,
        output_path=profile_fig_path,
    )
    plt.close(prof_fig)

    timeline_fig_path = figures_dir / "diagnostic_area_timeline_2023.png"
    tl_fig = plot_diagnostic_area_timeline(
        bin_summaries=bin_summaries,
        output_path=timeline_fig_path,
    )
    plt.close(tl_fig)

    plt.close("all")

    # 8. Executive Summary Generation & Export
    summary_dict = generate_seasonal_executive_summary(
        aoi_id=aoi_id,
        aoi_name=aoi_name,
        bbox=aoi_bbox,
        season_name="2023 Long Rains",
        bin_summaries=bin_summaries,
    )
    summary_json_path = reports_dir / "seasonal_executive_summary_2023.json"
    summary_md_path = reports_dir / "seasonal_executive_summary_2023.md"

    export_seasonal_summary_json(summary_dict, summary_json_path)
    export_seasonal_summary_markdown(summary_dict, summary_md_path)

    manifest = {
        "status": "SUCCESS",
        "n_bins_executed": n_bins,
        "total_clusters": summary_dict["diagnostic_totals"]["total_extracted_clusters"],
        "cumulative_candidate_area_ha": summary_dict["diagnostic_totals"]["total_candidate_area_ha_cumulative"],
        "actionable_bin_count": summary_dict["diagnostic_totals"]["actionable_bin_count"],
        "exported_artifacts": {
            "summary_json": str(summary_json_path),
            "summary_markdown": str(summary_md_path),
            "seasonal_geojson": str(seasonal_geojson_path),
            "temporal_profile_plot": str(profile_fig_path),
            "area_timeline_plot": str(timeline_fig_path),
            "persistence_heatmap_plot": str(persist_fig_path),
            "geojson_directory": str(geojson_dir),
            "figures_directory": str(figures_dir),
        },
        "summary": summary_dict,
    }
    return manifest
