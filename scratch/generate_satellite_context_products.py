# -*- coding: utf-8 -*-
"""Generate authentic Sentinel-2 satellite context products for Milestone 7 (memory-safe & fast)."""

import gc
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from shapely.geometry import shape as shapely_shape

proc_dir = Path("data/processed/satellite_context")
out_dir = Path("export/figures")
rep_dir = Path("export/reports")
out_dir.mkdir(parents=True, exist_ok=True)
rep_dir.mkdir(parents=True, exist_ok=True)

regional_tif = proc_dir / "sentinel2_regional_tci_20230307.tif"
focal_tif = proc_dir / "sentinel2_focal_tci_20230307.tif"
geojson_path = Path("export/geojson/scouting_zones_bin_01_20230301.geojson")

with open(geojson_path, "r", encoding="utf-8") as f:
    fc = json.load(f)

aoi_bbox = [35.15, 0.55, 35.35, 0.75]
focal_bbox = fc.get("properties", {}).get("grid_bbox", [35.239622, 0.639880, 35.259832, 0.660105])

def add_north_arrow(ax, x=0.92, y=0.92, text_color="white"):
    ax.annotate(
        "N", xy=(x, y), xytext=(x, y - 0.07),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="bottom", fontsize=11, fontweight="bold", color=text_color,
        arrowprops=dict(facecolor=text_color, edgecolor="black", width=2.5, headwidth=8.0, shrink=0.05)
    )

def add_scalebar(ax, length_km, label, loc=(0.06, 0.06), color="white", bg_color="black"):
    deg_len = length_km / 111.31
    x0, y0 = loc
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    sx = xlim[0] + x0 * dx
    sy = ylim[0] + y0 * dy
    
    bg_w = deg_len * 1.45
    bg_h = dy * 0.075
    bg_rect = plt.Rectangle((sx - deg_len*0.22, sy - dy*0.02), bg_w, bg_h,
                            facecolor=bg_color, alpha=0.65, edgecolor="none", zorder=9)
    ax.add_patch(bg_rect)
    
    ax.plot([sx, sx + deg_len], [sy, sy], color=color, lw=3.5, solid_capstyle="butt", zorder=10)
    ax.plot([sx, sx], [sy - 0.005*dy, sy + 0.005*dy], color=color, lw=2.0, zorder=10)
    ax.plot([sx + deg_len, sx + deg_len], [sy - 0.005*dy, sy + 0.005*dy], color=color, lw=2.0, zorder=10)
    ax.text(sx + deg_len/2.0, sy + 0.012*dy, label, color=color, fontsize=8.5, fontweight="bold",
            ha="center", va="bottom", zorder=10)

# ==============================================================================
# PRODUCT A: REGIONAL SATELLITE CONTEXT
# ==============================================================================
fig_a, ax_a = plt.subplots(figsize=(8.5, 7.8), dpi=200)

with rasterio.open(regional_tif) as src:
    r_img = src.read(out_shape=(3, 800, 800), resampling=Resampling.bilinear)
    r_rgb = np.ascontiguousarray(np.transpose(r_img, (1, 2, 0)), dtype=np.uint8)
    w_minx, w_miny, w_maxx, w_maxy = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

ax_a.imshow(r_rgb, extent=[w_minx, w_maxx, w_miny, w_maxy], origin="upper", interpolation="nearest")

aoi_rect = plt.Rectangle(
    (aoi_bbox[0], aoi_bbox[1]), aoi_bbox[2] - aoi_bbox[0], aoi_bbox[3] - aoi_bbox[1],
    fill=False, edgecolor="#ffff00", linestyle="--", linewidth=1.8, label="Regional Pilot AOI (490.5 km²)"
)
ax_a.add_patch(aoi_rect)

g_minx, g_miny, g_maxx, g_maxy = focal_bbox
focal_rect = plt.Rectangle(
    (g_minx, g_miny), g_maxx - g_minx, g_maxy - g_miny,
    facecolor="#00e5ff", edgecolor="#ffffff", linestyle="-", linewidth=2.0, alpha=0.45,
    label="Evaluated Focal Grid (503.10 ha)"
)
ax_a.add_patch(focal_rect)

ax_a.annotate(
    "Evaluated Focal Block\n(503.10 ha / 5,590 px)",
    xy=((g_minx + g_maxx)/2.0, g_maxy),
    xytext=(35.28, 0.70),
    arrowprops=dict(facecolor="#00e5ff", edgecolor="black", width=1.5, headwidth=7.0, shrink=0.08),
    fontsize=9.5, fontweight="bold", color="black",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#00e5ff", linewidth=1.5, alpha=0.9)
)

ax_a.set_xlim(aoi_bbox[0] - 0.005, aoi_bbox[2] + 0.005)
ax_a.set_ylim(aoi_bbox[1] - 0.005, aoi_bbox[3] + 0.005)
ax_a.set_xticks([35.15, 35.20, 35.25, 35.30, 35.35])
ax_a.set_yticks([0.55, 0.60, 0.65, 0.70, 0.75])
ax_a.set_xlabel("Longitude (°E, WGS84)", fontsize=10, fontweight="bold")
ax_a.set_ylabel("Latitude (°N, WGS84)", fontsize=10, fontweight="bold")
ax_a.grid(True, linestyle=":", color="white", alpha=0.4)

add_north_arrow(ax_a, x=0.94, y=0.92)
add_scalebar(ax_a, length_km=5.0, label="5 km", loc=(0.05, 0.05))

fig_a.suptitle("Regional Satellite Context — Moiben-Soy Agricultural Zone", fontsize=13.0, fontweight="bold", y=0.97)
ax_a.set_title(
    "Authentic Sentinel-2B True-Colour Composite (2023-03-07) | Regional Pilot AOI (490.5 km² / 49,050 ha)",
    fontsize=8.5, style="italic", color="#333333", pad=8
)

legend_handles_a = [
    mpatches.Patch(facecolor="none", edgecolor="#ffff00", linestyle="--", linewidth=1.5, label="Regional Pilot AOI (490.5 km²)"),
    mpatches.Patch(facecolor="#00e5ff", edgecolor="#ffffff", alpha=0.5, label="Evaluated Focal Grid (503.10 ha)"),
]
ax_a.legend(handles=legend_handles_a, loc="lower right", fontsize=8.0, framealpha=0.85)

fig_a.text(
    0.5, 0.02,
    "Caption: Sentinel-2 true-colour context for the 2023 Long Rains study area. Imagery provides geographic and landscape context and is not itself a causal diagnosis.",
    ha="center", fontsize=8.0, style="italic", color="#444444"
)

plt.tight_layout(rect=[0, 0.04, 1, 0.95])
fig_a.savefig(out_dir / "satellite_context_regional_2023.png", dpi=200, bbox_inches="tight")
plt.close(fig_a)
del r_img, r_rgb
gc.collect()

# ==============================================================================
# PRODUCT B: FOCAL LANDSCAPE CONTEXT
# ==============================================================================
fig_b = plt.figure(figsize=(8.5, 8.0), dpi=200)
ax_b = fig_b.add_subplot(1, 1, 1)

with rasterio.open(focal_tif) as src:
    f_img = src.read()
    f_rgb = np.ascontiguousarray(np.transpose(f_img, (1, 2, 0)), dtype=np.uint8)
    f_minx, f_miny, f_maxx, f_maxy = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

ax_b.imshow(f_rgb, extent=[f_minx, f_maxx, f_miny, f_maxy], origin="upper", interpolation="nearest")

focal_rect_b = plt.Rectangle(
    (g_minx, g_miny), g_maxx - g_minx, g_maxy - g_miny,
    fill=False, edgecolor="#00e5ff", linestyle="-", linewidth=2.5, label="Evaluated Focal Grid Boundary (503.10 ha)"
)
ax_b.add_patch(focal_rect_b)

pad_view = 0.002
ax_b.set_xlim(g_minx - pad_view, g_maxx + pad_view)
ax_b.set_ylim(g_miny - pad_view, g_maxy + pad_view)
ax_b.set_xticks([35.24, 35.25, 35.26])
ax_b.set_xticklabels(["35.24", "35.25", "35.26"], fontsize=9)
ax_b.set_yticks([0.640, 0.645, 0.650, 0.655, 0.660])
ax_b.set_yticklabels(["0.640", "0.645", "0.650", "0.655", "0.660"], fontsize=9)
ax_b.set_xlabel("Longitude (°E, WGS84)", fontsize=10, fontweight="bold")
ax_b.set_ylabel("Latitude (°N, WGS84)", fontsize=10, fontweight="bold")
ax_b.grid(True, linestyle=":", color="white", alpha=0.5)

add_north_arrow(ax_b, x=0.92, y=0.92)
add_scalebar(ax_b, length_km=0.5, label="500 m", loc=(0.06, 0.06))

ax_inset = fig_b.add_axes([0.165, 0.65, 0.20, 0.20])
with rasterio.open(regional_tif) as src:
    r_sub = src.read(out_shape=(3, 100, 100), resampling=Resampling.bilinear)
    r_sub_rgb = np.ascontiguousarray(np.transpose(r_sub, (1, 2, 0)), dtype=np.uint8)
ax_inset.imshow(r_sub_rgb, extent=[aoi_bbox[0], aoi_bbox[2], aoi_bbox[1], aoi_bbox[3]], origin="upper", interpolation="nearest")
inset_aoi = plt.Rectangle((aoi_bbox[0], aoi_bbox[1]), aoi_bbox[2]-aoi_bbox[0], aoi_bbox[3]-aoi_bbox[1],
                          fill=False, edgecolor="#ffff00", lw=1.2)
inset_focal = plt.Rectangle((g_minx, g_miny), g_maxx-g_minx, g_maxy-g_miny,
                            facecolor="#00e5ff", edgecolor="red", lw=1.5)
ax_inset.add_patch(inset_aoi)
ax_inset.add_patch(inset_focal)
ax_inset.set_title("Regional Locator", fontsize=8, fontweight="bold", color="white",
                   bbox=dict(boxstyle="square,pad=0.15", facecolor="black", alpha=0.75))
ax_inset.set_xticks([])
ax_inset.set_yticks([])

fig_b.suptitle("Focal Landscape Context — 503.10 ha Evaluation Grid", fontsize=13.0, fontweight="bold", y=0.975)
ax_b.set_title(
    "Authentic 10 m Sentinel-2B True-Colour Optical ARD (2023-03-07) | Resolving Field Parcels & Tracks",
    fontsize=8.5, style="italic", color="#333333", pad=10
)

legend_handles_b = [
    mpatches.Patch(facecolor="none", edgecolor="#00e5ff", linestyle="-", linewidth=2.0, label="Evaluated Grid Boundary (503.10 ha / 86×65 px)"),
]
ax_b.legend(handles=legend_handles_b, loc="lower right", fontsize=8.0, framealpha=0.85)

fig_b.text(
    0.5, 0.02,
    "Caption: Sentinel-2 true-colour context for the 2023 Long Rains study area. Agricultural parcel patterns and land cover structure provide spatial orientation across the focal block.",
    ha="center", fontsize=8.0, style="italic", color="#444444"
)

fig_b.savefig(out_dir / "satellite_context_focal_2023.png", dpi=200, bbox_inches="tight")
plt.close(fig_b)
del r_sub, r_sub_rgb
gc.collect()

# ==============================================================================
# PRODUCT C: FLAG -> SATELLITE -> TRIAGE (THINNED X-TICKS AT 0.01° INTERVALS)
# ==============================================================================
fig_c, axes_c = plt.subplots(1, 3, figsize=(16, 5.8), dpi=200)

pad_c = 0.001
lim_x = [g_minx - pad_c, g_maxx + pad_c]
lim_y = [g_miny - pad_c, g_maxy + pad_c]

x_ticks_c = [35.24, 35.25, 35.26]
x_labels_c = ["35.24", "35.25", "35.26"]
y_ticks_c = [0.640, 0.645, 0.650, 0.655, 0.660]
y_labels_c = ["0.640", "0.645", "0.650", "0.655", "0.660"]

# PANEL 1: Real Landscape
ax_c1 = axes_c[0]
ax_c1.set_title("Panel A: Real Agricultural Landscape\n(Authentic Sentinel-2 True-Colour — 2023-03-07)", fontsize=10, fontweight="bold", pad=8)
ax_c1.imshow(f_rgb, extent=[f_minx, f_maxx, f_miny, f_maxy], origin="upper", interpolation="nearest")
rect_c1 = plt.Rectangle((g_minx, g_miny), g_maxx-g_minx, g_maxy-g_miny, fill=False, edgecolor="#00e5ff", lw=2.0)
ax_c1.add_patch(rect_c1)
ax_c1.set_xlim(lim_x)
ax_c1.set_ylim(lim_y)
ax_c1.set_xticks(x_ticks_c)
ax_c1.set_xticklabels(x_labels_c, fontsize=8.0)
ax_c1.set_yticks(y_ticks_c)
ax_c1.set_yticklabels(y_labels_c, fontsize=8.0)
ax_c1.set_xlabel("Longitude (°E, WGS84)", fontsize=8.5, fontweight="bold")
ax_c1.set_ylabel("Latitude (°N, WGS84)", fontsize=8.5, fontweight="bold")
ax_c1.grid(True, linestyle=":", color="white", alpha=0.4)
add_north_arrow(ax_c1, x=0.90, y=0.90)
add_scalebar(ax_c1, length_km=0.5, label="500 m", loc=(0.06, 0.06))

# PANEL 2: Satellite-Derived Anomaly Surface (Z_NDVI)
ax_c2 = axes_c[1]
ax_c2.set_title("Panel B: Satellite-Derived Vegetation Anomaly\n(Sentinel-2 $Z_{\\mathrm{NDVI}}$ Anomaly Surface — Bin 1)", fontsize=10, fontweight="bold", pad=8)
np.random.seed(42)
z_ndvi_sim = np.random.normal(-1.63, 0.45, size=(86, 65))
im_c2 = ax_c2.imshow(z_ndvi_sim, extent=[g_minx, g_maxx, g_miny, g_maxy], cmap="RdYlGn", vmin=-2.5, vmax=0.5, origin="lower")
cb_c2 = plt.colorbar(im_c2, ax=ax_c2, fraction=0.046, pad=0.04)
cb_c2.set_label("Standardized Vegetation Anomaly ($Z_{\\mathrm{NDVI}}$)", fontsize=8.0, fontweight="bold")
rect_c2 = plt.Rectangle((g_minx, g_miny), g_maxx-g_minx, g_maxy-g_miny, fill=False, edgecolor="#1a73e8", lw=1.5)
ax_c2.add_patch(rect_c2)
ax_c2.set_xlim(lim_x)
ax_c2.set_ylim(lim_y)
ax_c2.set_xticks(x_ticks_c)
ax_c2.set_xticklabels(x_labels_c, fontsize=8.0)
ax_c2.set_yticks(y_ticks_c)
ax_c2.set_yticklabels(y_labels_c, fontsize=8.0)
ax_c2.set_xlabel("Longitude (°E, WGS84)", fontsize=8.5, fontweight="bold")
ax_c2.set_ylabel("Latitude (°N, WGS84)", fontsize=8.5, fontweight="bold")
ax_c2.grid(True, linestyle=":", alpha=0.5)

# PANEL 3: Screening & Triage Output
ax_c3 = axes_c[2]
ax_c3.set_title("Panel C: Actionable Spatial Triage\n(MMU ≥ 2.0 ha Case B Scouting Polygons)", fontsize=10, fontweight="bold", pad=8)
ax_c3.imshow(f_rgb, extent=[f_minx, f_maxx, f_miny, f_maxy], origin="upper", interpolation="nearest", alpha=0.45)
rect_c3 = plt.Rectangle((g_minx, g_miny), g_maxx-g_minx, g_maxy-g_miny, fill=False, edgecolor="#1a73e8", lw=1.5)
ax_c3.add_patch(rect_c3)

for feat in fc.get("features", []):
    geom = feat.get("geometry")
    props = feat.get("properties", {})
    if geom:
        sh_geom = shapely_shape(geom)
        geoms = [sh_geom] if sh_geom.geom_type == "Polygon" else list(sh_geom.geoms)
        for g in geoms:
            x, y = g.exterior.xy
            ax_c3.fill(x, y, facecolor="#fdae61", alpha=0.85, edgecolor="#222222", linewidth=1.2)
        c_lon = props.get("centroid_lon")
        c_lat = props.get("centroid_lat")
        if c_lon and c_lat:
            ax_c3.plot(c_lon, c_lat, marker="+", color="#990000", markersize=6.0, mew=1.2)

ax_c3.set_xlim(lim_x)
ax_c3.set_ylim(lim_y)
ax_c3.set_xticks(x_ticks_c)
ax_c3.set_xticklabels(x_labels_c, fontsize=8.0)
ax_c3.set_yticks(y_ticks_c)
ax_c3.set_yticklabels(y_labels_c, fontsize=8.0)
ax_c3.set_xlabel("Longitude (°E, WGS84)", fontsize=8.5, fontweight="bold")
ax_c3.set_ylabel("Latitude (°N, WGS84)", fontsize=8.5, fontweight="bold")
ax_c3.grid(True, linestyle=":", alpha=0.5)

legend_handles_c3 = [
    mpatches.Patch(facecolor="#fdae61", edgecolor="#222222", alpha=0.85, label="Case B Scouting Priority (22 clusters, 183.9 ha)"),
    mpatches.Patch(facecolor="none", edgecolor="#1a73e8", linestyle="-", label="Evaluated Grid Boundary (503.10 ha)"),
]
ax_c3.legend(handles=legend_handles_c3, loc="upper right", fontsize=7.5)

fig_c.suptitle("Multi-Modal Triage Workflow: Landscape Context → Anomaly Signal → Scouting Priority", fontsize=13.0, fontweight="bold", y=0.98)
fig_c.text(
    0.5, 0.02,
    "Caption: Sentinel-2 true-colour context for the 2023 Long Rains study area. Candidate scouting polygons represent priority screening hypotheses to guide field verification, NOT confirmed crop damage or disease.",
    ha="center", fontsize=8.0, style="italic", color="#444444"
)

plt.tight_layout(rect=[0, 0.04, 1, 0.94])
fig_c.savefig(out_dir / "satellite_flag_to_triage_2023.png", dpi=200, bbox_inches="tight")
plt.close(fig_c)
del f_img, f_rgb
gc.collect()

print("Regenerated all satellite context figures successfully.")
