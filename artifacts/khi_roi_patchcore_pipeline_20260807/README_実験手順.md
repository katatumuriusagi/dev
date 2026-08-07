# KHI / MVTec AD 再実験パッケージ

## 1. 目的

本パッケージは、以下の研究方針を実験可能な形へ分解したものである。

```text
撮影環境固有の画像
        ↓
Dataset-specific ROI Prior Calibration
        ↓
幾何形状を維持した Canonical ROI Normalization
        ↓
CNN patch feature
        ↓
Normal-Mode-Preserving Stratified Coreset
        ↓
PatchCore nearest neighbor
        ↓
Suspicious patches
        ↓
Sparse Graph Context
        ↓
異常マップ・判定
```

ただし、研究上の寄与を混同しないため、最初の再実験では次の順序で進める。

1. ROI Prior Calibrationを確立する。
2. Canonical ROI Normalization後のSaikuPatchCore baselineを固定する。
3. train/goodだけで正常モードの存在を確認する。
4. その後、層化coreset単独を実装・評価する。
5. 層化coresetの寄与確認後にSparse Graph Contextへ進む。

したがって本パッケージは、**Phase 0～Phase 2を再現可能にするコード**である。層化coresetとGNNをいきなり同時に有効化しない。

---

# 2. ファイル一覧

```text
khi_roi_patchcore_pipeline_20260807/
├─ khi_roi_common.py
├─ 01_roi_calibration_app.py
├─ 02_build_canonical_mvtec_dataset.py
├─ 03_analyze_normal_modes.py
├─ 04_run_saiku_patchcore_baselines.py
├─ requirements.txt
└─ README_実験手順.md
```

## khi_roi_common.py

ROI mask、円fit、GrabCut、正方形crop、等方resize等の共通関数である。

## 01_roi_calibration_app.py

少数の代表正常画像から撮影環境固有のROI Priorを作る。

対応モード：

- `circle`
- `polygon`
- `smart`（粗い輪郭＋GrabCut）

## 02_build_canonical_mvtec_dataset.py

保存したROI profileをMVTec AD形式データセット全体へ適用する。

- ROIを含む正方形をcrop
- 正方形→正方形resize
- ROI maskとGTへ同一変換
- 円profileでは局所補正を任意で実施

## 03_analyze_normal_modes.py

`train/good`だけから明度・コントラスト・gradient等を算出し、探索的なnormal modeを作る。

異常画像・GTは使用しない。

## 04_run_saiku_patchcore_baselines.py

Canonical datasetに対するSaikuPatchCore系基準実験を行う。

- WideResNet50-2
- layer2[-1] / layer3[-1]
- low=256 / high=512
- k=3、異常スコアは第1近傍距離
- ROI外patchをmemory bankから除外
- memory bank特徴はchunk保存
- global k-means 1%、random 1%、解像度、fusion、FAISS方式比較

---

# 3. 推奨フォルダ配置

例：

```text
C:\research\KHI_SaikuPatchCore_work\
├─ tools\roi_patchcore_v1\
│  ├─ khi_roi_common.py
│  ├─ 01_roi_calibration_app.py
│  ├─ 02_build_canonical_mvtec_dataset.py
│  ├─ 03_analyze_normal_modes.py
│  └─ 04_run_saiku_patchcore_baselines.py
│
├─ dataset_mvtec\
│  ├─ khi_chip_6\
│  │  └─ chip\
│  └─ khi_chip_7\
│     └─ chip\
│
├─ roi_profiles\
├─ datasets_canonical\
└─ outputs\
```

元データは変更しないこと。

---

# 4. Phase 0: 環境確認

Anaconda PowerShell：

```powershell
conda activate <SaikuPatchCore環境名>

cd C:\research\KHI_SaikuPatchCore_work

python -c "import cv2, torch, torchvision, faiss, sklearn, numpy, pandas; print('OK')"
```

推定実行時間：数秒。

CUDA確認：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

# 5. Phase 1A: ROI Prior Calibration

## 5.1 KHIのような円形領域

### #6例

```powershell
cd C:\research\KHI_SaikuPatchCore_work\tools\roi_patchcore_v1

python .\01_roi_calibration_app.py `
  --image-dir "C:\research\KHI_SaikuPatchCore_work\dataset_mvtec\khi_chip_6\chip\train\good" `
  --mode circle `
  --max-images 8 `
  --circle-residual-px 3.0 `
  --output-dir "C:\research\KHI_SaikuPatchCore_work\roi_profiles\khi_chip_6\chip"
```

### #7例

```powershell
python .\01_roi_calibration_app.py `
  --image-dir "C:\research\KHI_SaikuPatchCore_work\dataset_mvtec\khi_chip_7\chip\train\good" `
  --mode circle `
  --max-images 8 `
  --circle-residual-px 3.0 `
  --output-dir "C:\research\KHI_SaikuPatchCore_work\roi_profiles\khi_chip_7\chip"
```

### 操作

- 左クリック：円周点を追加
- 5～20点程度を推奨
- 右クリック：最後の点を削除
- Enter：RANSAC円fitして確定
- N：次画像
- B：前画像
- R：リセット
- S：profile保存
- Q：終了

### 出力

```text
roi_profiles\khi_chip_6\chip\
├─ roi_profile.json
├─ calibration_raw.json
├─ per_image_masks\
└─ previews\
```

### 推定時間

8枚校正の場合、人手操作を含め約3～10分。

---

## 5.2 非円形：Polygon

```powershell
python .\01_roi_calibration_app.py `
  --image-dir "<train_good_folder>" `
  --mode polygon `
  --max-images 5 `
  --output-dir "<roi_profile_output>"
```

頂点を順にクリックし、Enterで確定する。

複数画像のpolygon maskを共通解像度へそろえ、ROI存在確率を作る。

---

## 5.3 非円形：Smart Brush / GrabCut

```powershell
python .\01_roi_calibration_app.py `
  --image-dir "<train_good_folder>" `
  --mode smart `
  --max-images 5 `
  --probability-threshold 0.6 `
  --output-dir "<roi_profile_output>"
```

左ドラッグで対象領域の外周を大まかに閉じるようになぞる。
EnterでGrabCutを実行する。

初回実装ではGrabCut結果の詳細brush修正は含まない。結果が不安定な画像はRでやり直す。

---

# 6. Phase 1B: Canonical dataset生成

## #6

```powershell
python .\02_build_canonical_mvtec_dataset.py `
  --dataset-root "C:\research\KHI_SaikuPatchCore_work\dataset_mvtec\khi_chip_6" `
  --profile-root "C:\research\KHI_SaikuPatchCore_work\roi_profiles\khi_chip_6\chip" `
  --output-root "C:\research\KHI_SaikuPatchCore_work\datasets_canonical\khi_chip_6_roi512" `
  --output-size 512 `
  --margin-ratio 0.10
```

## #7

```powershell
python .\02_build_canonical_mvtec_dataset.py `
  --dataset-root "C:\research\KHI_SaikuPatchCore_work\dataset_mvtec\khi_chip_7" `
  --profile-root "C:\research\KHI_SaikuPatchCore_work\roi_profiles\khi_chip_7\chip" `
  --output-root "C:\research\KHI_SaikuPatchCore_work\datasets_canonical\khi_chip_7_roi512" `
  --output-size 512 `
  --margin-ratio 0.10
```

### 円ROIを軽量局所補正する場合

```powershell
python .\02_build_canonical_mvtec_dataset.py `
  --dataset-root "C:\research\KHI_SaikuPatchCore_work\dataset_mvtec\khi_chip_6" `
  --profile-root "C:\research\KHI_SaikuPatchCore_work\roi_profiles\khi_chip_6\chip" `
  --output-root "C:\research\KHI_SaikuPatchCore_work\datasets_canonical\khi_chip_6_roi512_refined" `
  --output-size 512 `
  --margin-ratio 0.10 `
  --refine-circle `
  --center-search-px 8 `
  --radius-search-px 5
```

### 比較するべき条件

最初は以下を分ける。

```text
P0: 元画像stretch resize（旧baseline）
P1: ROI Prior固定 + canonical crop
P2: ROI Prior + constrained circle refinement
```

P1とP2の差が小さければ、工場運用では高速なP1を優先できる。

### 出力構造

```text
datasets_canonical\khi_chip_6_roi512\
└─ chip\
   ├─ train\good\
   ├─ test\good\
   ├─ test\chip\
   ├─ ground_truth\chip\
   ├─ roi_masks\train\good\
   ├─ roi_masks\test\good\
   ├─ roi_masks\test\chip\
   └─ preprocessing_manifest.csv
```

### 必ず確認すること

1. 円が楕円になっていない。
2. 切粉がcropで切れていない。
3. GTと画像が一致している。
4. ROI maskが検査対象を覆っている。
5. paddingが異常に大きくない。

### 推定時間

単純crop/resizeは数分～十数分程度。
`--refine-circle`は各画像で局所探索するため数倍長くなる可能性がある。
実測値は`preprocessing_manifest.csv`の`preprocess_sec`で記録する。

---

# 7. Phase 1C: SaikuPatchCore baseline

まずk-means 1%を使う。

```powershell
python .\04_run_saiku_patchcore_baselines.py `
  --dataset-root "C:\research\KHI_SaikuPatchCore_work\datasets_canonical\khi_chip_6_roi512" `
  --output-root "C:\research\KHI_SaikuPatchCore_work\outputs\roi_patchcore_v1" `
  --preset-name "E01_global_kmeans_1percent" `
  --device cuda
```

### 重要

このコードではROI外patchをmemory bankへ入れない。
したがってpadding・背景が正常prototypeを占有しにくい。

### 実験プリセット

```text
E00_baseline_no_compression
E01_global_kmeans_1percent
E02_random_1percent
E03_low256_only
E04_high512_only
E05_dual_max
E06_flatl2_diagnostic
E07_hnsw
```

`E00`はmemory bankが巨大になるため、全量KHIでは使用しない。
小規模互換確認だけに使う。

### 最初に行う推奨順

```text
E01 global k-means 1%
↓
E02 random 1%
↓
E03 low256 only
↓
E04 high512 only
↓
E06 FlatL2 diagnostic
```

### 出力

```text
outputs\roi_patchcore_v1\E01_global_kmeans_1percent\chip\
├─ memory_bank.npy
├─ memory_index.faiss
├─ predictions.csv
├─ metrics.json
├─ metrics.csv
├─ resolved_config.json
└─ anomaly_maps\
```

### 記録される時間

- feature extraction
- sampler / k-means
- FAISS index build
- test total
- mean inference / image
- p95 inference / image

### 推定実行時間

初回は実測が必要。
256+512、全train画像、CPU MiniBatchKMeansを含むため、KHI全量では数十分～数時間になる可能性がある。

---

# 8. Phase 2: 正常モード探索

baselineと並行してtrain/goodの正常構造を確認する。

```powershell
python .\03_analyze_normal_modes.py `
  --dataset-root "C:\research\KHI_SaikuPatchCore_work\datasets_canonical\khi_chip_6_roi512" `
  --output-root "C:\research\KHI_SaikuPatchCore_work\outputs\normal_modes_v1\khi_chip_6" `
  --n-modes 4
```

最初は`n_modes=3,4,5,6`程度を比較してもよいが、異常画像の成績を見てcluster数を選んではならない。

出力：

```text
normal_descriptors.csv
normal_modes.csv
normal_mode_summary.csv
normal_mode_config.json
```

見る項目：

- modeごとの画像数
- 最少modeのfrequency
- mean intensity
- contrast
- gradient
- 代表画像
- clusterが撮影時系列だけを分離していないか

---

# 9. Phase 3へ進む条件

次の3点が確認できてから正常モード保存型層化coresetを実装する。

1. canonical化によって幾何歪みが解消している。
2. baselineが再現可能に実行でき、異常マップが切粉部分へ反応している。
3. 正常モード分析で、少数正常群と多数正常群の不均衡が確認できる。

その後、`normal_modes.csv`を使い、同一memory bank予算の下で以下を比較する。

```text
Global random
Global k-means
Uniform per-mode
Proportional per-mode
Minimum-guaranteed proportional
Minimum-guaranteed sqrt allocation
```

層化coresetの主指標は単なるImage AUROCだけではなく、

- FP/1000 good
- mode別FPR
- worst-mode FPR
- chip Recall
- Pixel AP
- AUPRO
- memory bank size
- sec/image

とする。

---

# 10. Sparse Graph Contextはその後

GNNは層化coreset単独の効果を確認後に追加する。

最初の候補：

```text
G0: graphなし
G1: training-free sparse graph baseline
G2: 1-layer GraphSAGE
G3: 1～2 layer GAT
```

全patchへGNNをかけず、PatchCore score上位M patchだけを再評価する。

これにより、精度改善と追加計算時間を分離して評価する。

---

# 11. 今回の実験で必ず残す成果物

各実験ごとに以下を保存する。

```text
experiment_id
目的
変更した要因
固定条件
dataset
split
normal/anomaly枚数
ROI profile
canonical size
margin
backbone
low/high resolution
sampler
memory bank before/after
index
metrics
feature extraction time
sampler time
index build time
inference mean/p95
total time
異常マップ
代表TP/FP/FN/TN
```

---

# 12. 実験結果を貼るとき

ChatGPTへ以下を共有する。

1. PowerShellの実行コマンド
2. コンソールログ全文または最後100～200行
3. `metrics.json`
4. `predictions.csv`
5. `normal_mode_summary.csv`（Phase 2以降）
6. 異常マップの代表例

その時点で、成功/失敗、エラー原因、結果解釈、次実験、ゼミ資料化可能性を判断する。
