# Known Errors and Fixes: B30 Work Edge ROI Experiment

## 1. generated_masks/good FileNotFoundError

### Symptom

KHI #7 polar evaluation failed while saving normal zero masks.

Example error:

```text
FileNotFoundError: [Errno 2] No such file or directory:
...\generated_masks\good\..._mask.png
```

### Cause

SaikuPatchCore attempted to save a zero mask for normal images under `generated_masks/good`, but the parent directory was not available at save time. Manual folder creation was not sufficient because output folders were reinitialized during execution.

### Fix

Patch `ensure_zero_mask()` in:

```text
C:/research/mvtec_ad2_exp/src/saiku_patchcore/main_SaikuPatchCore_multi_res_coreset_safe.py
```

Add parent-folder creation before saving:

```python
import os
os.makedirs(os.path.dirname(str(mask_path)), exist_ok=True)
Image.fromarray(zero).save(mask_path)
```

A backup was created in the local experiment:

```text
main_SaikuPatchCore_multi_res_coreset_safe.py.bak_B30_20260611_124044
```

## 2. Long Windows path issue

Even after patching `ensure_zero_mask()`, KHI #7 polar evaluation failed under the long output path:

```text
C:/research/KHI_SaikuPatchCore_work/reports/priority_A_experiments/B30_work_edge_roi/B30f2_saiku_patchcore_polar_eval/...
```

### Fix

Use a shorter output root:

```text
C:/research/b30f3_short
```

The short-path rerun succeeded and generated:

```text
predictions.csv
metrics_summary.csv
execution_times.csv
```

## 3. Pixel AUROC warnings

The following warning appeared:

```text
UndefinedMetricWarning: Only one class is present in y_true. ROC AUC score is not defined in that case.
```

This affects pixel-level AUROC only. Image-level AUROC/AUPRC and prediction CSV generation are valid for the current experiment.
