# Original MVTec-Compatible Dataset Structure

The original KHI datasets were arranged in an MVTec-compatible structure and used as the source for the B30 disk and polar datasets.

## Expected root

```text
C:\research\KHI_SaikuPatchCore_work\dataset_mvtec
```

## Expected structure

```text
dataset_mvtec/
  khi_chip_6/
    chip/
      train/
        good/
      test/
        good/
        chip/
      ground_truth/
        chip/       # optional; used when pixel masks are available
  khi_chip_7/
    chip/
      train/
        good/
      test/
        good/
        chip/
      ground_truth/
        chip/       # optional; used when pixel masks are available
```

## Counts used in B30

| dataset | train/good | test/good | test/chip |
|---|---:|---:|---:|
| `khi_chip_6` | 1133 | 284 | 18 |
| `khi_chip_7` | 2952 | 738 | 14 |

## Notes

- Category name is `chip` for both datasets.
- B30 disk datasets were derived from these original images.
- Original image binaries are not committed to this repository.
