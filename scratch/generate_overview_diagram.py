# -*- coding: utf-8 -*-
"""Generate high-resolution Project Overview Architecture Diagram (Deliverable 01)."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out_dir = Path("export/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# PROJECT OVERVIEW DIAGRAM: project_overview_diagram.png
# -------------------------------------------------------------
fig_arch, ax_arch = plt.subplots(figsize=(15, 9.5), dpi=300)
ax_arch.axis("off")
fig_arch.patch.set_facecolor("#fcfcfc")

def draw_box(ax, x, y, w, h, title, items, header_bg="#1a73e8", border_color="#1a73e8", bg_color="#ffffff"):
    header_h = 0.22 * h
    rect_main = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
                               facecolor=bg_color, edgecolor=border_color, linewidth=1.5, zorder=2)
    ax.add_patch(rect_main)
    rect_head = FancyBboxPatch((x, y + h - header_h), w, header_h,
                               boxstyle="round,pad=0.015,rounding_size=0.02",
                               facecolor=header_bg, edgecolor=border_color, linewidth=1.5, zorder=3)
    ax.add_patch(rect_head)
    ax.text(x + w/2, y + h - header_h/2, title, color="white", fontsize=10, fontweight="bold",
            ha="center", va="center", zorder=4)
    item_y = y + h - header_h - 0.04
    for itm in items:
        ax.text(x + 0.02, item_y, itm, color="#222222", fontsize=8.2, va="top", zorder=4)
        item_y -= 0.042

draw_box(ax_arch, 0.04, 0.65, 0.27, 0.26,
         "1. MULTI-SENSOR INGESTION",
         ["• CHIRPS (5.5 km, Monthly Rain)",
          "• Sentinel-2 L2A (10-20 m Optical)",
          "• NASA SMAP L4 (9 km, Root-Zone)",
          "• Regional Pilot AOI (490.5 km²)",
          "• Dynamic Cropland Mask"],
         header_bg="#2b5b84", border_color="#2b5b84")

draw_box(ax_arch, 0.365, 0.65, 0.27, 0.26,
         "2. RESOLUTION INTEGRITY",
         ["• Canonical Grid: EPSG:6933 (30 m)",
          "• Scene Classification (SCL) Mask",
          "• No False Resampling / Interpolation",
          "• Nearest-Neighbor Broadcast",
          "• SHA-256 Provenance Manifest"],
         header_bg="#1e7e34", border_color="#1e7e34")

draw_box(ax_arch, 0.69, 0.65, 0.27, 0.26,
         "3. FEATURE ANOMALIES",
         ["• Rainfall Z_R (vs 30-yr baseline)",
          "• Vegetation Z_NDVI (vs baseline)",
          "• Root-Zone Soil Water Index (SWI)",
          "• 14-day Trend (ΔSWI_14d = SWI_t - SWI_t-1)",
          "• 15 Contiguous Canonical Bins"],
         header_bg="#d97706", border_color="#d97706")

draw_box(ax_arch, 0.12, 0.34, 0.76, 0.25,
         "4. DETERMINISTIC SCREENING ENGINE (M5 RULE MATRIX)",
         ["• CASE A (Drought Stress): Z_R < -0.8  ∧  Z_NDVI < -1.0  (Zero events in 2023)",
          "• CASE B (Vegetation Anomaly under Non-Dry Priors): Z_R ≥ 0.0  ∧  SWI ≥ 0.30  ∧  Z_NDVI < -1.2  (Bin 1: 183.9 ha)",
          "• CASE C (Meteorological Drought Buffering): Z_R < -1.0  ∧  SWI ≥ 0.30  ∧  Z_NDVI ≥ -0.5  (Non-actionable)",
          "• CASE D (Precipitation-to-Moisture Disconnect): Z_R ≥ 0.5  ∧  ΔSWI_14d ≤ 0.0  (Bins 10-14: 503.1 ha)",
          "• MULTI_SIGNAL: Simultaneous Case B + Case D co-occurrence  (Bin 15: 503.1 ha)",
          "• NORMAL (0): Safe baseline  |  INSUFFICIENT EVIDENCE (-1): Missing required modality"],
         header_bg="#b91c1c", border_color="#b91c1c")

draw_box(ax_arch, 0.04, 0.04, 0.43, 0.24,
         "5. SPATIAL TRIAGE & VECTORIZATION",
         ["• 3×3 Morphological Opening (Noise Removal)",
          "• 8-Connected Component Labeling",
          "• Minimum Mapping Unit (MMU) Filter: ≥ 2.0 ha",
          "• Polygon Vectorization & Topology Clean",
          "• Centroid Coordinates (+) for Dispatch",
          "• 28 Clusters Extracted (3,202.5 ha total)"],
         header_bg="#6b21a8", border_color="#6b21a8")

draw_box(ax_arch, 0.53, 0.04, 0.43, 0.24,
         "6. STAKEHOLDER DECISION PRODUCTS",
         ["• 10 High-Res Cartographic Products (300 DPI)",
          "• Seasonal Profile, Area Timeline, Persistence Map",
          "• RFC 7946 GeoJSON Candidate Scouting Zones",
          "• Markdown & JSON Executive Summaries",
          "• Strict Non-Overclaiming Scientific Terminology",
          "• Clear Separation: Detection vs Diagnosis"],
         header_bg="#0f766e", border_color="#0f766e")

# Connective arrows
arrow_props = dict(facecolor="#555555", edgecolor="none", width=2.0, headwidth=6.0, shrink=0.05)
ax_arch.annotate("", xy=(0.36, 0.78), xytext=(0.315, 0.78), arrowprops=arrow_props)
ax_arch.annotate("", xy=(0.685, 0.78), xytext=(0.64, 0.78), arrowprops=arrow_props)
ax_arch.annotate("", xy=(0.50, 0.60), xytext=(0.50, 0.64), arrowprops=arrow_props)
ax_arch.annotate("", xy=(0.25, 0.29), xytext=(0.35, 0.33), arrowprops=arrow_props)
ax_arch.annotate("", xy=(0.75, 0.29), xytext=(0.65, 0.33), arrowprops=arrow_props)

fig_arch.suptitle("End-to-End Multi-Modal Agricultural Screening Architecture", fontsize=15, fontweight="bold", y=0.97)
ax_arch.text(0.5, 0.93, "Deterministic Earth Observation Workflow: From Ingestion to Field Triage (Uasin Gishu, Kenya)",
             ha="center", fontsize=9.5, style="italic", color="#555555")

fig_arch.savefig(out_dir / "project_overview_diagram.png", dpi=300, bbox_inches="tight")
plt.close(fig_arch)
print("Generated project_overview_diagram.png")
