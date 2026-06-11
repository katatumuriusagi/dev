# B30f3 Final Disk / Polar Comparison

## Final summary

| dataset | input_type | variant | AUROC | AUPRC | gap |
|---|---|---|---:|---:|---:|
| KHI #6 | disk | `raw_disk_pad15` | 0.982003 | 0.953535 | 2.017271 |
| KHI #6 | polar | `raw_disk_pad15_polar` | 0.987872 | 0.936162 | -0.108360 |
| KHI #6 | disk | `mix_disk_pad15` | 0.982394 | 0.953704 | 2.178359 |
| KHI #6 | polar | `mix_disk_pad15_polar` | 0.995110 | 0.962987 | 0.982605 |
| KHI #7 | disk | `raw_disk_pad15` | 0.984224 | 0.840054 | -0.168885 |
| KHI #7 | polar | `raw_disk_pad15_polar` | 0.984127 | 0.899778 | 0.602173 |
| KHI #7 | disk | `mix_disk_pad15` | 0.995161 | 0.871932 | 0.556542 |
| KHI #7 | polar | `mix_disk_pad15_polar` | 0.998355 | 0.947192 | 0.842705 |

## Main conclusion

For KHI #7, `mix_disk_pad15_polar` is the best condition. It improves AUROC, AUPRC, and gap over `mix_disk_pad15`.

- `mix_disk_pad15`: AUROC 0.995161, AUPRC 0.871932, gap 0.556542
- `mix_disk_pad15_polar`: AUROC 0.998355, AUPRC 0.947192, gap 0.842705

## Threshold result for KHI #7 mix polar

| threshold | TP | FP | TN | FN | precision | recall | specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| midpoint_p95_p05 | 14 | 13 | 725 | 0 | 0.518519 | 1.000000 | 0.982385 | 0.682927 |
| normal_max | 12 | 1 | 737 | 2 | 0.923077 | 0.857143 | 0.998645 | 0.888889 |
| normal_p95 | 14 | 37 | 701 | 0 | 0.274510 | 1.000000 | 0.949864 | 0.430769 |
| normal_p99 | 13 | 8 | 730 | 1 | 0.619048 | 0.928571 | 0.989160 | 0.742857 |

## Operational interpretation

The polar transform is especially useful for KHI #7. At the midpoint threshold, it keeps TP=14 and FN=0 while reducing FP from 25 in the disk mix condition to 13 in the polar mix condition. At the normal_max threshold, it keeps FP=1 and reduces FN from 4 to 2.
