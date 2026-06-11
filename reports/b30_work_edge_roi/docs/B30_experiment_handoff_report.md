# B30追加実験引き継ぎレポート

## 1. 研究背景と目的

本実験は，オートリベッタ穿孔後画像における切粉・異物検知を対象として，ワーク端部円内を検査領域とする入力画像が SaikuPatchCore の検出性能に与える影響を評価したものである。

従来の検討では，カウンターシンク外側を中心とした ROI が高い AUROC を示した。しかし実検査範囲としては，中心孔，カウンターシンク，ワーク表面を含むワーク端部円内全体を扱う必要がある。そこで B30 系列では，手動推定したワーク端部円を基準として disk 画像を作成し，さらに円形構造を半径方向に整理するため polar 変換を適用した。

## 2. 実験系列

| ID | 内容 |
|---|---|
| B30d | 手動円アノテーションによりワーク端部円の中心・半径を推定 |
| B30e0d | 15%余白付きワーク円抽出 dataset 作成 |
| B30e1 | disk画像による SaikuPatchCore 評価 |
| B30f0/B30f0b | polar transform preview と mix確認 |
| B30f1 | 全画像 polar dataset 作成 |
| B30f2 | polar評価用 workset 作成 |
| B30f3 | polar画像による SaikuPatchCore 評価 |
| B30f3-short | KHI #7 polar を短い出力パスで再実行 |
| B30f3-final | disk / polar 最終比較 |

## 3. 入力条件

| データセット | 入力条件 | train/good | test/good | test/chip |
|---|---|---:|---:|---:|
| KHI #6 | `raw_disk_pad15` | 1133 | 284 | 18 |
| KHI #6 | `mix_disk_pad15` | 1133 | 284 | 18 |
| KHI #6 | `raw_disk_pad15_polar` | 1133 | 284 | 18 |
| KHI #6 | `mix_disk_pad15_polar` | 1133 | 284 | 18 |
| KHI #7 | `raw_disk_pad15` | 2952 | 738 | 14 |
| KHI #7 | `mix_disk_pad15` | 2952 | 738 | 14 |
| KHI #7 | `raw_disk_pad15_polar` | 2952 | 738 | 14 |
| KHI #7 | `mix_disk_pad15_polar` | 2952 | 738 | 14 |

## 4. SaikuPatchCore設定

| 項目 | 値 |
|---|---|
| `res_low` | 256 |
| `res_high` | 512 |
| `coreset_rate` | 0.05 |
| `device` | cuda |
| `save_anomaly_maps` | 1 |
| `write_predictions_csv` | 1 |

## 5. B30e1 disk評価結果

| dataset | input | AUROC | AUPRC | gap |
|---|---|---:|---:|---:|
| KHI #6 | `raw_disk_pad15` | 0.982003 | 0.953535 | 2.017271 |
| KHI #6 | `mix_disk_pad15` | 0.982394 | 0.953704 | 2.178359 |
| KHI #7 | `raw_disk_pad15` | 0.984224 | 0.840054 | -0.168885 |
| KHI #7 | `mix_disk_pad15` | 0.995161 | 0.871932 | 0.556542 |

B30e1では，特に KHI #7 において `mix_disk_pad15` が有効であった。raw条件では gap が負であったが，mix条件では gap が正となった。

## 6. B30f3 polar評価結果

| dataset | input | AUROC | AUPRC | gap |
|---|---|---:|---:|---:|
| KHI #6 | `raw_disk_pad15_polar` | 0.987872 | 0.936162 | -0.108360 |
| KHI #6 | `mix_disk_pad15_polar` | 0.995110 | 0.962987 | 0.982605 |
| KHI #7 | `raw_disk_pad15_polar` | 0.984127 | 0.899778 | 0.602173 |
| KHI #7 | `mix_disk_pad15_polar` | 0.998355 | 0.947192 | 0.842705 |

## 7. 最終比較

KHI #7では `mix_disk_pad15_polar` が最良条件となった。

| dataset | input | AUROC | AUPRC | gap |
|---|---|---:|---:|---:|
| KHI #7 | `mix_disk_pad15` | 0.995161 | 0.871932 | 0.556542 |
| KHI #7 | `mix_disk_pad15_polar` | 0.998355 | 0.947192 | 0.842705 |

midpointしきい値では，disk条件の FP=25 に対し，polar条件では FP=13 まで減少した。TP=14, FN=0 は維持された。normal maxしきい値では，disk条件の TP=10, FP=1, FN=4 に対し，polar条件では TP=12, FP=1, FN=2 となった。

## 8. エラーと修正

KHI #7 polar条件は，長い出力パスで `generated_masks/good` に正常画像用ゼロマスクを保存する際，`FileNotFoundError` で停止した。

対応:

1. `generated_masks/good` と `generated_masks/chip` を手動作成したが，再初期化により再発した。
2. SaikuPatchCore本体の `ensure_zero_mask()` に，保存前に親フォルダを作成するpatchを追加した。
3. 長いパス問題を避けるため，`C:/research/b30f3_short` で KHI #7 polar 2条件を再実行した。
4. short pathでは `predictions.csv`, `metrics_summary.csv`, `execution_times.csv` の生成を確認した。

## 9. 次に行うこと

次工程は B30f4 として，KHI #7 `mix_disk_pad15_polar` の異常マップ可視化レビューを行うことである。

確認項目:

- normal high score例
- midpointしきい値における FP 13件
- normal maxしきい値における FN 2件
- disk と polar の同一画像比較
- raw polar と mix polar の比較
- カウンターシンク，中心孔反射，ワーク色差への反応が低減しているか

