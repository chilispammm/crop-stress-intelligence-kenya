"""Generate official 2023 Long Rains Milestone 6 Production Artifacts."""

from pathlib import Path
from src.export.production_export import execute_m6_seasonal_pipeline

if __name__ == "__main__":
    export_dir = Path("export")
    manifest = execute_m6_seasonal_pipeline(
        output_dir=export_dir,
        aoi_id="ug_pilot_moiben_01",
        aoi_name="Moiben-Soy Agricultural Pilot Zone",
        aoi_bbox=(35.150, 0.550, 35.350, 0.750),
        seed=42,
    )
    print("=== M6 PRODUCTION PIPELINE EXECUTION MANIFEST ===")
    print(f"Status:                      {manifest['status']}")
    print(f"Canonical Bins Executed:     {manifest['n_bins_executed']}")
    print(f"Actionable Bins:             {manifest['actionable_bin_count']}")
    print(f"Total Extracted Clusters:    {manifest['total_clusters']}")
    print(f"Cumulative Candidate Area:   {manifest['cumulative_candidate_area_ha']:.2f} ha")
    print("\nExported Artifacts:")
    for k, v in manifest["exported_artifacts"].items():
        print(f"  * {k:<25}: {v}")
