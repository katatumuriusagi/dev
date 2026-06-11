# B30 Dataset Manifest

This manifest summarizes all dataset variants used in B30e1 and B30f3.

## Original MVTec-compatible datasets

| eval target | local dataset | category | train/good | test/good | test/chip |
|---|---|---|---:|---:|---:|
| KHI #6 | `khi_chip_6` | `chip` | 1133 | 284 | 18 |
| KHI #7 | `khi_chip_7` | `chip` | 2952 | 738 | 14 |

Representative local root:

```text
C:\research\KHI_SaikuPatchCore_work\dataset_mvtec
```

## ROI disk datasets generated in B30e0d

| variant | dataset | category | train/good | test/good | test/chip |
|---|---|---|---:|---:|---:|
| `raw_disk_pad15` | KHI #6 | `chip` | 1133 | 284 | 18 |
| `mix_disk_pad15` | KHI #6 | `chip` | 1133 | 284 | 18 |
| `raw_disk_pad15` | KHI #7 | `chip` | 2952 | 738 | 14 |
| `mix_disk_pad15` | KHI #7 | `chip` | 2952 | 738 | 14 |

Representative local root:

```text
C:\research\KHI_SaikuPatchCore_work\reports\priority_A_experiments\B30_work_edge_roi\B30e0d_margin_work_disk_extracted_datasets
```

B30e0d output summary:

```text
source_images: 5139
variants: raw_disk_pad15, mix_disk_pad15
output_images: 10278
elapsed_seconds: 1285.0
elapsed_minutes: 21.4
```

## Polar datasets generated in B30f1

| variant | dataset | category | train/good | test/good | test/chip | polar size |
|---|---|---|---:|---:|---:|---|
| `raw_disk_pad15_polar` | KHI #6 | `chip` | 1133 | 284 | 18 | 900 x 300 |
| `mix_disk_pad15_polar` | KHI #6 | `chip` | 1133 | 284 | 18 | 900 x 300 |
| `raw_disk_pad15_polar` | KHI #7 | `chip` | 2952 | 738 | 14 | 900 x 300 |
| `mix_disk_pad15_polar` | KHI #7 | `chip` | 2952 | 738 | 14 | 900 x 300 |

Representative local root:

```text
C:\research\KHI_SaikuPatchCore_work\reports\priority_A_experiments\B30_work_edge_roi\B30f1_polar_datasets
```

B30f1 output summary:

```text
source_rows: 10278
polar_width: 900
polar_height: 300
elapsed_seconds: 246.7
elapsed_minutes: 4.1
```

## SaikuPatchCore worksets

B30e1 disk worksets:

```text
C:\research\KHI_SaikuPatchCore_work\reports\priority_A_experiments\B30_work_edge_roi\B30e1_saiku_patchcore_disk_eval\work_datasets
```

B30f2/B30f3 polar worksets:

```text
C:\research\KHI_SaikuPatchCore_work\reports\priority_A_experiments\B30_work_edge_roi\B30f2_saiku_patchcore_polar_eval\work_datasets
```

KHI #7 short-path polar worksets:

```text
C:\research\b30f3_short\work_datasets\k7_raw_polar
C:\research\b30f3_short\work_datasets\k7_mix_polar
```

## Reproduction note

The files in this GitHub directory document the datasets but do not include the image binaries. To reproduce the experiment, place the image folders at the local paths above or update config files accordingly.
