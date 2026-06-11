# Dataset Handoff for B30 Experiments

This directory documents the dataset families used in the B30 work-edge ROI / polar SaikuPatchCore experiments.

## Dataset families

B30 uses three dataset families.

1. `original`: original KHI image datasets arranged in MVTec-compatible structure.
2. `roi_disk`: work-edge circular ROI cropped datasets with 15 percent margin.
3. `polar`: polar-transformed datasets generated from `roi_disk`.

Large image files are not committed here. This directory records the structure, generation logic, and local paths needed to reproduce or locate the datasets.

## Local source paths

| Dataset family | Local path |
|---|---|
| original | `C:\research\KHI_SaikuPatchCore_work\dataset_mvtec` |
| roi_disk | `C:\research\KHI_SaikuPatchCore_work\reports\priority_A_experiments\B30_work_edge_roi\B30e0d_margin_work_disk_extracted_datasets` |
| polar | `C:\research\KHI_SaikuPatchCore_work\reports\priority_A_experiments\B30_work_edge_roi\B30f1_polar_datasets` |

## Image counts

| dataset | train/good | test/good | test/chip |
|---|---:|---:|---:|
| KHI #6 | 1133 | 284 | 18 |
| KHI #7 | 2952 | 738 | 14 |

The same counts apply to original, ROI disk, and polar dataset families when generated correctly.

## Variants

| Variant | Description |
|---|---|
| `raw_disk_pad15` | Original image cropped around the manually estimated work-edge circle with 15 percent margin. |
| `mix_disk_pad15` | CLAHE / Retinex-like preprocessing applied before the same disk crop. |
| `raw_disk_pad15_polar` | Polar transform applied to `raw_disk_pad15`. |
| `mix_disk_pad15_polar` | Polar transform applied to `mix_disk_pad15`. |

## Why image files are not committed

The full datasets include thousands of camera images and derived anomaly maps. They should not be committed to a normal Git repository. Use Git LFS or external storage if binary sharing is required.
