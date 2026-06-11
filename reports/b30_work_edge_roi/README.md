# B30 Work Edge ROI / Polar Transform Experiment Handoff

このフォルダは，オートリベッタ穿孔後画像における切粉・異物検知を対象とした B30 系列実験の引き継ぎ用資料である。

## 目的

ワーク端部円内を検査対象とする入力画像を作成し，SaikuPatchCore による異常検知性能を disk 画像条件と polar 画像条件で比較した。

## 最重要結論

KHI #7 では `mix_disk_pad15_polar` が最良条件であった。

| dataset | input | AUROC | AUPRC | gap |
|---|---|---:|---:|---:|
| KHI #7 | `mix_disk_pad15` | 0.995161 | 0.871932 | 0.556542 |
| KHI #7 | `mix_disk_pad15_polar` | 0.998355 | 0.947192 | 0.842705 |

midpoint しきい値では，disk 条件の FP=25 に対し，polar 条件では FP=13 まで減少した。TP=14, FN=0 は維持された。

## 推奨される次工程

1. B30f4 として KHI #7 `mix_disk_pad15_polar` の異常マップ可視化レビューを行う。
2. FP 13 件の原因を分類する。
3. normal max しきい値で残る FN 2 件の切粉位置を確認する。
4. `mix_disk_pad15` と `mix_disk_pad15_polar` を通常 PatchCore / orgPatchCore / EfficientAD / SimpleNet / KIZUKI 比較に展開する。

## フォルダ構成

```text
reports/b30_work_edge_roi/
  README.md
  docs/
    B30_experiment_handoff_report.md
    B30_experiment_handoff_report.tex
  configs/
    config_B30f3_saiku_patchcore_polar_eval.yaml
    config_b30f3_short_khi7_polar.yaml
  scripts/
    make_b30f1_polar_datasets.py
    make_b30f2_saiku_polar_eval_worksets.py
    make_b30f3_final_collect_with_short_khi7.py
    patch_saiku_ensure_zero_mask.py
  results/
    B30e1_disk_summary.md
    B30f3_final_disk_polar_comparison.md
    B30f3_short_khi7_threshold_summary.md
  environment/
    anaconda_powershell_setup.md
    environment_note.md
  notes/
    directory_map.md
    known_errors_and_fixes.md
    next_experiments.md
```

## 注意

この共有物はチャット上で共有されたコマンド，ログ，出力結果から再構成した引き継ぎ用パッケージである。ローカル PC 内の `C:\research\...` の実ファイルを直接読み取ったものではないため，ローカル最新版と完全一致させたい場合は，該当 `.py` / `.yaml` をローカルから上書きすること。
