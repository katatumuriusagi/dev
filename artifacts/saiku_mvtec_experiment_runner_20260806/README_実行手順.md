# MVTec AD形式対応 SaikuPatchCore実験ランナー

## 1. 目的

`saiku_patchcore_mvtec_experiment.py`は、元のSaikuPatchCoreコードを基に、次の機能を追加した実験用コードである。

- MVTec AD形式の任意データセットに対応
- データセット直下からカテゴリを自動検出
- KHI #6、KHI #7、MVTec AD等を同じフォルダ規則で実行
- 冒頭の実験プリセットをコメントアウトで切替
- 元条件、k-means 1%、256/512解像度アブレーション、random 1%、IPCA、FlatL2、HNSWを比較
- 特徴をチャンク保存し、巨大なPython listによるCPU RAM不足を軽減
- Image AUROC、Image AP、画素指標の近似値、工程別時間を保存
- 日本語コメント付き

正常モード保存型層化coresetとGNNは、本研究の主提案候補である。ただし、baseline、層化coreset単独、graph単独、統合の順で検証する必要があるため、この基礎ランナーにはまだ実装していない。

---

## 2. 対応データセット構造

### 複数カテゴリを含む場合

```text
DATASET_ROOT
├─ bottle
│  ├─ train
│  │  └─ good
│  ├─ test
│  │  ├─ good
│  │  └─ broken_large
│  └─ ground_truth
│     └─ broken_large
├─ leather
│  └─ ...
└─ metal_nut
   └─ ...
```

### KHI #6だけを実行する場合

```text
DATASET_ROOT
└─ chip
   ├─ train
   │  └─ good
   ├─ test
   │  ├─ good
   │  └─ chip
   └─ ground_truth
      └─ chip
```

異常画像が次の場合、

```text
test/chip/000.png
```

GTは次のいずれかで対応付ける。

```text
ground_truth/chip/000_mask.png
ground_truth/chip/000.png
```

対応拡張子は以下である。

```text
.png .jpg .jpeg .bmp .tif .tiff
```

---

## 3. 実験条件の切替

コード冒頭の次の部分だけを編集する。

```python
ACTIVE_EXPERIMENT = EXPERIMENT_KMEANS_1_PERCENT

# ACTIVE_EXPERIMENT = EXPERIMENT_BASELINE_ORIGINAL
# ACTIVE_EXPERIMENT = EXPERIMENT_LOW_256_ONLY
# ACTIVE_EXPERIMENT = EXPERIMENT_HIGH_512_ONLY
# ACTIVE_EXPERIMENT = EXPERIMENT_DUAL_MAX
# ACTIVE_EXPERIMENT = EXPERIMENT_RANDOM_1_PERCENT
# ACTIVE_EXPERIMENT = EXPERIMENT_IPCA256_KMEANS1
# ACTIVE_EXPERIMENT = EXPERIMENT_FLAT_DIAGNOSTIC
# ACTIVE_EXPERIMENT = EXPERIMENT_HNSW
```

実行したい1行だけを有効にする。同時に複数行を有効化してはならない。

### 実験プリセット

| プリセット | 内容 | 研究上の用途 |
|---|---|---|
| E00 | 256+512、圧縮なし、IVF、mean融合 | 元SaikuPatchCoreに近い基準。ただし大規模KHIではメモリ負荷が高い |
| E01 | 256+512、MiniBatchKMeans 1% | KHI全量向けの主要初期条件 |
| E02 | 256のみ、k-means 1% | 解像度アブレーション |
| E03 | 512のみ、k-means 1% | 微小局所異常と速度の比較 |
| E04 | 256+512、max融合 | mean融合との比較 |
| E05 | random 1% | k-meansの寄与確認 |
| E06 | IPCA 256次元+k-means 1% | 次元削減・省メモリ評価 |
| E07 | FlatL2 | IVF近似誤差の診断基準 |
| E08 | HNSW | 高速近似検索比較 |

---

## 4. カテゴリ選択

### 自動検出

```python
CATEGORY_SELECTION_MODE = "auto"
```

`DATASET_ROOT`直下にあるMVTec AD形式カテゴリをすべて実行する。

### 手動指定

```python
# CATEGORY_SELECTION_MODE = "auto"
CATEGORY_SELECTION_MODE = "manual"
MANUAL_CATEGORIES = ["chip"]
```

コマンドラインの`--categories`を指定した場合は、コード内設定より優先される。

---

## 5. 導入

Anaconda PowerShellで実行する。

```powershell
conda activate <SaikuPatchCore環境名>

cd C:\research\KHI_SaikuPatchCore_work

pip install numpy pandas pillow scipy scikit-learn opencv-python
```

PyTorchはCUDA環境に合った既存版を使用する。FAISSも現在のSaikuPatchCore環境に導入済みのものを使用する。

### 構文確認

```powershell
python -m py_compile .\saiku_patchcore_mvtec_experiment.py
```

推定時間は数秒である。

---

## 6. 最初に行うデータセット検証

モデル処理をせず、フォルダ構造、画像数、GT対応だけを確認する。

```powershell
python .\saiku_patchcore_mvtec_experiment.py `
  --dataset-root "C:\research\KHI_SaikuPatchCore_work\dataset_mvtec\khi_chip_6" `
  --output-root "C:\research\KHI_SaikuPatchCore_work\outputs\saiku_mvtec_experiments" `
  --validate-only
```

推定時間は数秒から1分程度である。

成功例：

```text
検出カテゴリ: ['chip']
実行カテゴリ: ['chip']
[検証成功] chip: train/good=1133, test=302
validate-onlyのため終了する。
```

---

## 7. k-means 1%実験

### コメントアウトで選択する場合

```python
ACTIVE_EXPERIMENT = EXPERIMENT_KMEANS_1_PERCENT
```

### コマンドラインで選択する場合

```powershell
python .\saiku_patchcore_mvtec_experiment.py `
  --dataset-root "C:\research\KHI_SaikuPatchCore_work\dataset_mvtec\khi_chip_6" `
  --output-root "C:\research\KHI_SaikuPatchCore_work\outputs\saiku_mvtec_experiments" `
  --preset-name "E01_kmeans_1percent" `
  --categories chip
```

### 推定時間

初回は環境と画像構成によるため断定できない。

- KHI #6、256+512、train約1100枚：数十分から数時間程度
- KHI #7、256+512、train約2900枚：#6より長く、数時間に達する可能性がある
- CPU MiniBatchKMeans：prototype数と特徴数に依存
- 2回目以降も現コードは特徴を再生成するため、同程度の時間が必要

実測後は`metrics.json`の以下を次回推定へ使用する。

```text
feature_extraction_sec
reducer_fit_transform_sec
bank_and_index_sec
test_total_sec
inference_mean_sec
inference_p95_sec
```

---

## 8. 出力構成

```text
outputs/saiku_mvtec_experiments
└─ E01_kmeans_1percent
   ├─ all_categories_summary.csv
   ├─ run_info.json
   └─ chip
      ├─ resolved_config.json
      ├─ compressed_memory_bank.npy
      ├─ memory_index.faiss
      ├─ predictions.csv
      ├─ metrics.json
      ├─ metrics.csv
      └─ anomaly_maps
         ├─ good
         └─ chip
```

`save_intermediate_chunks=False`では特徴チャンクを実験終了後に削除する。

---

## 9. 重要な注意

### 9.1 画素指標

`pixel_auroc_sampled`と`pixel_ap_sampled`は、正常画素と異常画素をそれぞれ最大100万件抽出した近似値である。巨大な画素配列によるRAM不足を防ぐためである。

論文・最終ゼミ資料では、別途保存した異常マップとGTから厳密なPixel AP、Pixel AUROC、AUPROを再計算する評価スクリプトを用意する。

### 9.2 k-means 1%の上限

```python
max_prototypes=50000
```

を設定している。元パッチ数の1%が5万を超える場合、実際のprototype数は5万で打ち切られる。`metrics.json`の次を必ず確認する。

```text
memory_bank_vectors_before
memory_bank_vectors_after
compression_ratio_actual
```

論文比較では、同じmemory bankサイズでglobal、random、層化coresetを比較する。

### 9.3 元コードとの完全同一性

本版は研究実験を安定して行うため、以下を変更している。

- カテゴリ自動検出
- PNG以外への対応
- GTをstemで厳密対応
- layer3の空間整合をnearest補間でベクトル化
- 全特徴のPython list蓄積をチャンク保存へ変更
- FAISS GPU KMeansではなくCPU MiniBatchKMeansを使用

したがって、Original SaikuPatchCoreとの再現性確認では、まず小規模データでスコア順位、異常マップ、AUROC差を比較する。

受入基準案：

```text
Spearman順位相関 >= 0.999
Image AUROC差 <= 0.001
Pixel AUROC差 <= 0.001
```

### 9.4 正常モード層化coresetとGNN

以下の順序を崩さない。

```text
baseline
↓
正常モードの存在確認
↓
層化coreset単独
↓
graph単独
↓
層化coreset+疎graph
↓
軽量化
```

本コードはbaselineから検索方式比較までの基礎実験用である。

---

## 10. 次に行う操作

1. ファイルを`C:\research\KHI_SaikuPatchCore_work`へ配置する。
2. `--validate-only`を実行する。
3. KHI #6の一部データでE01を実行する。
4. `metrics.json`、`predictions.csv`、異常マップを確認する。
5. 元コードとの互換比較を行う。
6. 問題がなければKHI #6全量、次に#7全量へ進む。
