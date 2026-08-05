# -*- coding: utf-8 -*-
"""
MVTec AD形式の任意データセットで実行できるSaikuPatchCore系実験ランナー
=====================================================================

目的
----
1. MVTec AD形式のデータセット直下からカテゴリを自動検出する。
2. 冒頭の「実験プリセット」だけをコメント解除して実験条件を切り替える。
3. 元SaikuPatchCore互換条件、k-means 1%、解像度アブレーション、
   random sampling、FAISS検索方式比較を同一コードで実施する。
4. 特徴を一度に巨大なPython listへ保持せず、チャンクファイルへ保存して
   CPU RAM不足を起こしにくくする。
5. 処理時間、設定、画像スコア、異常マップを実験別フォルダへ保存する。

重要
----
- 本ファイルは、アップロードされた元コード
  main_SaikuPatchCore_256_512_0788.py を基礎として整理した実験用コードである。
- 元コードと同様にWideResNet50-2のlayer2/layer3特徴を用いる。
- n_neighbors=3を維持するが、互換モードでは最近傍第1位の距離だけを
  異常スコアとして使用する。
- 正常モード保存型層化coresetとGNNは研究上の主提案候補であるが、
  基礎比較前に混ぜると寄与が分からなくなるため、本版では未実装である。
- 最初は小規模データで動作確認し、その後KHI全量へ進むこと。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# =============================================================================
# 0. ユーザーが主に変更する場所
# =============================================================================

# -----------------------------------------------------------------------------
# データセットルート
# -----------------------------------------------------------------------------
# 例1：複数カテゴリを含むMVTec AD形式
# DATASET_ROOT/
#   bottle/train/good/...
#   bottle/test/good/...
#   bottle/test/broken_large/...
#   bottle/ground_truth/broken_large/*_mask.png
#   leather/train/good/...
#   ...
#
# 例2：KHI #6だけを処理する場合
# DATASET_ROOT/
#   chip/train/good/...
#   chip/test/good/...
#   chip/test/chip/...
#   chip/ground_truth/chip/*_mask.png
#
# コマンドラインの --dataset-root で上書き可能である。
DEFAULT_DATASET_ROOT = Path(r"C:\research\KHI_SaikuPatchCore_work\dataset_mvtec\khi_chip_6")

# 出力先。--output-rootで上書き可能である。
DEFAULT_OUTPUT_ROOT = Path(r"C:\research\KHI_SaikuPatchCore_work\outputs\saiku_mvtec_experiments")

# -----------------------------------------------------------------------------
# カテゴリ選択
# -----------------------------------------------------------------------------
# "auto"：DATASET_ROOT直下にあるMVTec AD形式カテゴリをすべて自動検出する。
# "manual"：MANUAL_CATEGORIESに指定したカテゴリだけを実行する。
CATEGORY_SELECTION_MODE = "auto"
# CATEGORY_SELECTION_MODE = "manual"

# manualを使う場合だけ編集する。
MANUAL_CATEGORIES = ["chip"]
# MANUAL_CATEGORIES = ["bottle", "leather"]

# -----------------------------------------------------------------------------
# 実験プリセット
# -----------------------------------------------------------------------------
# 実行する実験の行だけコメント解除する。
# 同時に複数行を有効化してはいけない。
# コマンドラインの --preset-name で上書きすることもできる。
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """1回の実験条件を保持する。"""

    # 実験名。出力フォルダ名にも使う。
    name: str

    # 入力解像度
    use_low: bool = True
    use_high: bool = True
    low_size: int = 256
    high_size: int = 512

    # 前処理
    # resize：画像全体を正方形へリサイズする。KHIのROI画像・極座標画像向け。
    # center_crop：一度load_sizeへ拡大後に中央を切り出す。MVTec物体カテゴリ向け。
    transform_mode: str = "resize"
    load_size: int = 286

    # backbone・特徴
    backbone: str = "wide_resnet50_2"
    pool_kernel: int = 3
    pool_stride: int = 1
    pool_padding: int = 1

    # 次元削減
    # none / ipca
    reducer: str = "none"
    reduced_dim: int = 256
    ipca_batch_size: int = 4096

    # memory bank圧縮
    # none / random / minibatch_kmeans
    sampler: str = "none"
    sampler_ratio: float = 1.0
    max_prototypes: int = 50000
    kmeans_batch_size: int = 4096
    kmeans_max_iter: int = 100

    # 最近傍探索
    # flat / ivf / hnsw
    index_type: str = "ivf"
    n_neighbors: int = 3
    ivf_nlist_factor: float = 4.0
    ivf_nprobe_ratio: float = 0.01
    hnsw_m: int = 32
    hnsw_ef_search: int = 64

    # 2解像度融合
    # max / mean / low / high
    fusion: str = "mean"

    # 実行設定
    train_batch_size: int = 4
    num_workers: int = 0
    device: str = "auto"  # auto / cuda / cpu
    seed: int = 319
    gaussian_sigma: float = 4.0
    save_anomaly_maps: bool = True
    save_intermediate_chunks: bool = False

    # 画素評価用の層化サンプリング数。
    # 正常画素と異常画素をそれぞれ最大この件数まで保持する。
    # Image AUROC/APは全画像で厳密計算する。
    pixel_metric_sample_per_class: int = 1_000_000

    # 互換確認用
    # Trueならk=3で検索し、第1近傍距離だけを使う元コード互換動作である。
    compatibility_first_neighbor_only: bool = True


# --- E00：元SaikuPatchCoreに近い基準条件 -------------------------------
EXPERIMENT_BASELINE_ORIGINAL = ExperimentConfig(
    name="E00_baseline_original_like",
    use_low=True,
    use_high=True,
    low_size=256,
    high_size=512,
    transform_mode="resize",
    reducer="none",
    sampler="none",
    sampler_ratio=1.0,
    index_type="ivf",
    n_neighbors=3,
    fusion="mean",
    train_batch_size=4,
)

# --- E01：メモリ不足対策のk-means 1% -------------------------------
EXPERIMENT_KMEANS_1_PERCENT = replace(
    EXPERIMENT_BASELINE_ORIGINAL,
    name="E01_kmeans_1percent",
    sampler="minibatch_kmeans",
    sampler_ratio=0.01,
    max_prototypes=50000,
)

# --- E02：256入力だけを使う解像度アブレーション --------------------
EXPERIMENT_LOW_256_ONLY = replace(
    EXPERIMENT_KMEANS_1_PERCENT,
    name="E02_low256_only_kmeans1",
    use_low=True,
    use_high=False,
    fusion="low",
)

# --- E03：512入力だけを使う解像度アブレーション --------------------
EXPERIMENT_HIGH_512_ONLY = replace(
    EXPERIMENT_KMEANS_1_PERCENT,
    name="E03_high512_only_kmeans1",
    use_low=False,
    use_high=True,
    fusion="high",
)

# --- E04：256+512、max融合 -----------------------------------------
EXPERIMENT_DUAL_MAX = replace(
    EXPERIMENT_KMEANS_1_PERCENT,
    name="E04_dual_max_kmeans1",
    fusion="max",
)

# --- E05：random sampling 1% ---------------------------------------
EXPERIMENT_RANDOM_1_PERCENT = replace(
    EXPERIMENT_BASELINE_ORIGINAL,
    name="E05_random_1percent",
    sampler="random",
    sampler_ratio=0.01,
    max_prototypes=50000,
)

# --- E06：IPCA 256次元 + k-means 1% -------------------------------
EXPERIMENT_IPCA256_KMEANS1 = replace(
    EXPERIMENT_KMEANS_1_PERCENT,
    name="E06_ipca256_kmeans1",
    reducer="ipca",
    reduced_dim=256,
)

# --- E07：FlatL2診断 -----------------------------------------------
# 近似検索IVFによる順位変化の影響を確認する基準である。
EXPERIMENT_FLAT_DIAGNOSTIC = replace(
    EXPERIMENT_KMEANS_1_PERCENT,
    name="E07_flatl2_diagnostic",
    index_type="flat",
)

# --- E08：HNSW高速検索 ---------------------------------------------
EXPERIMENT_HNSW = replace(
    EXPERIMENT_KMEANS_1_PERCENT,
    name="E08_hnsw_kmeans1",
    index_type="hnsw",
)

# -----------------------------------------------------------------------------
# 実行プリセット選択
# -----------------------------------------------------------------------------
# 現在はk-means 1%を有効にしている。
ACTIVE_EXPERIMENT = EXPERIMENT_KMEANS_1_PERCENT

# 別実験を行う場合は、上の行をコメントアウトし、下から1行だけ有効にする。
# ACTIVE_EXPERIMENT = EXPERIMENT_BASELINE_ORIGINAL
# ACTIVE_EXPERIMENT = EXPERIMENT_LOW_256_ONLY
# ACTIVE_EXPERIMENT = EXPERIMENT_HIGH_512_ONLY
# ACTIVE_EXPERIMENT = EXPERIMENT_DUAL_MAX
# ACTIVE_EXPERIMENT = EXPERIMENT_RANDOM_1_PERCENT
# ACTIVE_EXPERIMENT = EXPERIMENT_IPCA256_KMEANS1
# ACTIVE_EXPERIMENT = EXPERIMENT_FLAT_DIAGNOSTIC
# ACTIVE_EXPERIMENT = EXPERIMENT_HNSW

PRESETS: Dict[str, ExperimentConfig] = {
    cfg.name: cfg
    for cfg in [
        EXPERIMENT_BASELINE_ORIGINAL,
        EXPERIMENT_KMEANS_1_PERCENT,
        EXPERIMENT_LOW_256_ONLY,
        EXPERIMENT_HIGH_512_ONLY,
        EXPERIMENT_DUAL_MAX,
        EXPERIMENT_RANDOM_1_PERCENT,
        EXPERIMENT_IPCA256_KMEANS1,
        EXPERIMENT_FLAT_DIAGNOSTIC,
        EXPERIMENT_HNSW,
    ]
}

# 対応拡張子。元コードはPNG限定であったが、本版は以下へ対応する。
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# =============================================================================
# 1. データセット検証・読込み
# =============================================================================


@dataclass(frozen=True)
class TestRecord:
    image_path: Path
    gt_path: Optional[Path]
    label: int
    defect_type: str


@dataclass(frozen=True)
class CategoryInfo:
    name: str
    path: Path
    train_good: Tuple[Path, ...]
    test_records: Tuple[TestRecord, ...]


def list_images(folder: Path) -> List[Path]:
    """指定フォルダ直下の対応画像をファイル名順で返す。"""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def is_mvtec_category_dir(path: Path) -> bool:
    """MVTec AD形式カテゴリとして最低限必要な構造を持つか確認する。"""
    return (
        path.is_dir()
        and (path / "train" / "good").is_dir()
        and (path / "test").is_dir()
    )


def discover_category_paths(dataset_root: Path) -> Dict[str, Path]:
    """
    データセットルートからカテゴリを検出する。

    dataset_root自体がカテゴリフォルダの場合にも対応する。
    """
    dataset_root = dataset_root.resolve()

    if is_mvtec_category_dir(dataset_root):
        return {dataset_root.name: dataset_root}

    categories = {
        p.name: p
        for p in sorted(dataset_root.iterdir())
        if is_mvtec_category_dir(p)
    }

    if not categories:
        raise FileNotFoundError(
            "MVTec AD形式カテゴリが見つからない。\n"
            f"確認したルート: {dataset_root}\n"
            "必要構造: <category>/train/good と <category>/test"
        )

    return categories


def find_gt_path(category_path: Path, defect_type: str, image_path: Path) -> Optional[Path]:
    """
    異常画像に対応するGTマスクを探す。

    MVTec AD標準の <stem>_mask.png を優先し、同名stemも候補とする。
    """
    gt_dir = category_path / "ground_truth" / defect_type
    if not gt_dir.exists():
        return None

    stems = [f"{image_path.stem}_mask", image_path.stem]
    for stem in stems:
        for ext in sorted(IMAGE_EXTENSIONS):
            candidate = gt_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def load_category_info(category_name: str, category_path: Path) -> CategoryInfo:
    """カテゴリ内のtrain/test/GTを検証して一覧化する。"""
    train_good = tuple(list_images(category_path / "train" / "good"))
    if not train_good:
        raise ValueError(f"{category_name}: train/goodに画像がない。")

    test_root = category_path / "test"
    defect_dirs = sorted(p for p in test_root.iterdir() if p.is_dir())
    if not defect_dirs:
        raise ValueError(f"{category_name}: test配下にgood/異常フォルダがない。")

    test_records: List[TestRecord] = []
    missing_gt: List[Path] = []

    for defect_dir in defect_dirs:
        defect_type = defect_dir.name
        images = list_images(defect_dir)
        label = 0 if defect_type == "good" else 1

        for image_path in images:
            gt_path = None
            if label == 1:
                gt_path = find_gt_path(category_path, defect_type, image_path)
                if gt_path is None:
                    missing_gt.append(image_path)

            test_records.append(
                TestRecord(
                    image_path=image_path,
                    gt_path=gt_path,
                    label=label,
                    defect_type=defect_type,
                )
            )

    if not test_records:
        raise ValueError(f"{category_name}: test画像がない。")

    labels = {record.label for record in test_records}
    if labels != {0, 1}:
        print(
            f"[警告] {category_name}: testに正常・異常の両方が存在しないため、"
            "Image AUROC/APを計算できない。"
        )

    if missing_gt:
        preview = "\n".join(str(p) for p in missing_gt[:10])
        raise FileNotFoundError(
            f"{category_name}: 異常画像に対応するGTが{len(missing_gt)}件見つからない。\n"
            f"先頭例:\n{preview}\n"
            "GT名は通常 <画像stem>_mask.png とする。"
        )

    return CategoryInfo(
        name=category_name,
        path=category_path,
        train_good=train_good,
        test_records=tuple(test_records),
    )


class ImagePathDataset(Dataset):
    """正常学習画像のパスをtransformして返す。"""

    def __init__(self, image_paths: Sequence[Path], transform):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path)


# =============================================================================
# 2. 前処理・特徴抽出
# =============================================================================


def build_transform(config: ExperimentConfig, size: int, is_gt: bool = False):
    """実験設定から画像またはGT用transformを構築する。"""
    interpolation = (
        transforms.InterpolationMode.NEAREST
        if is_gt
        else transforms.InterpolationMode.BILINEAR
    )

    operations = []
    if config.transform_mode == "resize":
        operations.append(transforms.Resize((size, size), interpolation=interpolation))
    elif config.transform_mode == "center_crop":
        load_size = max(config.load_size, size)
        operations.extend(
            [
                transforms.Resize((load_size, load_size), interpolation=interpolation),
                transforms.CenterCrop(size),
            ]
        )
    else:
        raise ValueError(f"未対応transform_mode: {config.transform_mode}")

    operations.append(transforms.ToTensor())

    if not is_gt:
        operations.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    return transforms.Compose(operations)


def resolve_device(config: ExperimentConfig) -> torch.device:
    """device設定を実環境へ解決する。"""
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cudaだがCUDAが使用できない。")
    return torch.device(config.device)


def build_backbone(config: ExperimentConfig, device: torch.device):
    """事前学習backboneを構築し、重みを固定する。"""
    if config.backbone != "wide_resnet50_2":
        raise ValueError(
            "本版はwide_resnet50_2のみ対応。"
            "軽量backbone比較は基礎手法の効果確認後に追加する。"
        )

    try:
        weights = torchvision.models.Wide_ResNet50_2_Weights.DEFAULT
        model = torchvision.models.wide_resnet50_2(weights=weights)
    except AttributeError:
        # 古いtorchvisionとの互換用
        model = torchvision.models.wide_resnet50_2(pretrained=True)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.to(device)
    model.eval()
    return model


class FeatureExtractor:
    """WideResNet50-2のlayer2/layer3特徴を結合してパッチ特徴を返す。"""

    def __init__(self, model, config: ExperimentConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.outputs: Dict[str, torch.Tensor] = {}

        self.handles = [
            model.layer2[-1].register_forward_hook(self._hook("layer2")),
            model.layer3[-1].register_forward_hook(self._hook("layer3")),
        ]

    def _hook(self, name: str):
        def callback(_module, _inputs, output):
            self.outputs[name] = output
        return callback

    def close(self):
        for handle in self.handles:
            handle.remove()

    @torch.no_grad()
    def extract(self, batch: torch.Tensor) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        入力batchから[全パッチ数, 特徴次元]を返す。

        元コードのembedding_concatはlayer3特徴をlayer2の空間サイズへ反復して
        結合している。本版ではnearest補間で同等の空間整合を行い、
        Pythonループを減らす。
        """
        self.outputs.clear()
        batch = batch.to(self.device, non_blocking=True)
        _ = self.model(batch)

        layer2 = self.outputs["layer2"]
        layer3 = self.outputs["layer3"]

        pool = torch.nn.AvgPool2d(
            kernel_size=self.config.pool_kernel,
            stride=self.config.pool_stride,
            padding=self.config.pool_padding,
        )
        layer2 = pool(layer2)
        layer3 = pool(layer3)

        layer3_up = F.interpolate(
            layer3,
            size=layer2.shape[-2:],
            mode="nearest",
        )
        merged = torch.cat([layer2, layer3_up], dim=1)

        batch_size, channels, height, width = merged.shape
        patches = (
            merged.permute(0, 2, 3, 1)
            .reshape(batch_size * height * width, channels)
            .contiguous()
        )

        result = patches.detach().cpu().numpy().astype(np.float32, copy=False)
        return result, (height, width)


# =============================================================================
# 3. 特徴チャンク保存・次元削減・memory bank圧縮
# =============================================================================


@dataclass
class ChunkSet:
    paths: List[Path]
    total_vectors: int
    feature_dim: int


def write_feature_chunks(
    image_paths: Sequence[Path],
    transform,
    extractor: FeatureExtractor,
    config: ExperimentConfig,
    chunk_dir: Path,
    scale_name: str,
) -> ChunkSet:
    """学習画像特徴をbatchごとの.npyへ保存し、巨大list保持を避ける。"""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    dataset = ImagePathDataset(image_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    paths: List[Path] = []
    total_vectors = 0
    feature_dim = 0

    for batch_index, (images, _paths) in enumerate(loader):
        features, _grid = extractor.extract(images)
        feature_dim = features.shape[1]
        total_vectors += features.shape[0]

        output_path = chunk_dir / f"{scale_name}_{batch_index:06d}.npy"
        np.save(output_path, features, allow_pickle=False)
        paths.append(output_path)

        if batch_index % 10 == 0:
            print(
                f"[{scale_name}] batch={batch_index}, "
                f"vectors={total_vectors:,}, dim={feature_dim}"
            )

        del images, features
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not paths:
        raise RuntimeError(f"{scale_name}: 特徴チャンクが生成されなかった。")

    return ChunkSet(paths=paths, total_vectors=total_vectors, feature_dim=feature_dim)


def iter_chunk_arrays(chunk_paths: Sequence[Path]) -> Iterable[np.ndarray]:
    """チャンクを1つずつmmapで開く。"""
    for path in chunk_paths:
        yield np.load(path, mmap_mode="r")


def fit_ipca(
    chunk_paths: Sequence[Path],
    feature_dim: int,
    config: ExperimentConfig,
) -> IncrementalPCA:
    """全特徴をRAMへ載せずIncrementalPCAを学習する。"""
    n_components = min(config.reduced_dim, feature_dim)
    ipca = IncrementalPCA(
        n_components=n_components,
        batch_size=max(config.ipca_batch_size, n_components),
    )

    for chunk in iter_chunk_arrays(chunk_paths):
        # partial_fitにはn_components以上の行数が必要である。
        if chunk.shape[0] >= n_components:
            ipca.partial_fit(np.asarray(chunk, dtype=np.float32))

    return ipca


def transform_chunks_with_ipca(
    chunk_paths: Sequence[Path],
    ipca: IncrementalPCA,
    output_dir: Path,
) -> List[Path]:
    """IPCA変換済み特徴を新しいチャンクとして保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []

    for index, chunk in enumerate(iter_chunk_arrays(chunk_paths)):
        transformed = ipca.transform(np.asarray(chunk, dtype=np.float32)).astype(np.float32)
        output_path = output_dir / f"reduced_{index:06d}.npy"
        np.save(output_path, transformed, allow_pickle=False)
        output_paths.append(output_path)

    return output_paths


def calculate_prototype_count(total_vectors: int, config: ExperimentConfig) -> int:
    """圧縮率と上限から保存prototype数を決める。"""
    requested = max(1, int(round(total_vectors * config.sampler_ratio)))
    return min(requested, config.max_prototypes, total_vectors)


def random_sample_chunks(
    chunk_paths: Sequence[Path],
    total_vectors: int,
    config: ExperimentConfig,
) -> np.ndarray:
    """全特徴からreservoir samplingで指定数を抽出する。"""
    sample_count = calculate_prototype_count(total_vectors, config)
    rng = np.random.default_rng(config.seed)
    reservoir: Optional[np.ndarray] = None
    seen = 0

    for chunk in iter_chunk_arrays(chunk_paths):
        array = np.asarray(chunk, dtype=np.float32)
        if reservoir is None:
            reservoir = np.empty((sample_count, array.shape[1]), dtype=np.float32)

        for row in array:
            if seen < sample_count:
                reservoir[seen] = row
            else:
                replacement = int(rng.integers(0, seen + 1))
                if replacement < sample_count:
                    reservoir[replacement] = row
            seen += 1

    if reservoir is None:
        raise RuntimeError("random sampling対象特徴がない。")

    return np.ascontiguousarray(reservoir[: min(seen, sample_count)], dtype=np.float32)


def minibatch_kmeans_chunks(
    chunk_paths: Sequence[Path],
    total_vectors: int,
    config: ExperimentConfig,
) -> np.ndarray:
    """
    チャンクを順次読み込み、MiniBatchKMeansでprototypeを作る。

    元コードのFAISS KMeansはlow/high全特徴をvstackしてから学習するため、
    KHI全量ではCPU RAMまたはGPUメモリ不足になりやすい。本関数は
    全特徴を同時保持せず、CPU上で順次partial_fitする。
    """
    n_clusters = calculate_prototype_count(total_vectors, config)
    print(
        f"MiniBatchKMeans: total={total_vectors:,}, "
        f"ratio={config.sampler_ratio}, clusters={n_clusters:,}"
    )

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=config.seed,
        batch_size=max(config.kmeans_batch_size, n_clusters),
        max_iter=config.kmeans_max_iter,
        n_init=1,
        reassignment_ratio=0.01,
        verbose=0,
    )

    # 初回partial_fit用にn_clusters行以上を集める。
    initial_parts: List[np.ndarray] = []
    initial_rows = 0
    for chunk in iter_chunk_arrays(chunk_paths):
        array = np.asarray(chunk, dtype=np.float32)
        initial_parts.append(array)
        initial_rows += array.shape[0]
        if initial_rows >= n_clusters:
            break

    initial_data = np.concatenate(initial_parts, axis=0)
    if initial_data.shape[0] > max(n_clusters * 2, config.kmeans_batch_size):
        rng = np.random.default_rng(config.seed)
        indices = rng.choice(
            initial_data.shape[0],
            size=max(n_clusters, config.kmeans_batch_size),
            replace=False,
        )
        initial_data = initial_data[indices]

    model.partial_fit(initial_data)
    del initial_data, initial_parts
    gc.collect()

    for index, chunk in enumerate(iter_chunk_arrays(chunk_paths)):
        array = np.asarray(chunk, dtype=np.float32)
        model.partial_fit(array)
        if index % 20 == 0:
            print(f"MiniBatchKMeans partial_fit: chunk={index}")

    return np.ascontiguousarray(model.cluster_centers_, dtype=np.float32)


# =============================================================================
# 4. FAISS index
# =============================================================================


def sample_index_training_vectors(
    chunk_paths: Sequence[Path],
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """IVF学習用にチャンクからreservoir samplingする。"""
    dummy_config = ExperimentConfig(
        name="index_training_sample",
        sampler_ratio=1.0,
        max_prototypes=max_samples,
        seed=seed,
    )
    total = sum(np.load(p, mmap_mode="r").shape[0] for p in chunk_paths)
    return random_sample_chunks(chunk_paths, total, dummy_config)


def build_faiss_index_from_bank(
    bank: np.ndarray,
    config: ExperimentConfig,
):
    """圧縮済みmemory bankからFAISS indexを作る。"""
    bank = np.ascontiguousarray(bank.astype(np.float32, copy=False))
    dim = bank.shape[1]

    if config.index_type == "flat":
        index = faiss.IndexFlatL2(dim)
        index.add(bank)

    elif config.index_type == "ivf":
        nlist = max(1, int(config.ivf_nlist_factor * math.sqrt(bank.shape[0])))
        nlist = min(nlist, bank.shape[0])
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        if hasattr(index, "cp"):
            index.cp.min_points_per_centroid = 1
        index.train(bank)
        index.add(bank)
        index.nprobe = max(1, int(round(nlist * config.ivf_nprobe_ratio)))

    elif config.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, config.hnsw_m, faiss.METRIC_L2)
        index.hnsw.efSearch = config.hnsw_ef_search
        index.add(bank)

    else:
        raise ValueError(f"未対応index_type: {config.index_type}")

    return index


def build_faiss_index_from_chunks(
    chunk_paths: Sequence[Path],
    feature_dim: int,
    total_vectors: int,
    config: ExperimentConfig,
):
    """圧縮なしの特徴チャンクをFAISS indexへ順次追加する。"""
    if config.index_type == "flat":
        index = faiss.IndexFlatL2(feature_dim)

    elif config.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(feature_dim, config.hnsw_m, faiss.METRIC_L2)
        index.hnsw.efSearch = config.hnsw_ef_search

    elif config.index_type == "ivf":
        nlist = max(1, int(config.ivf_nlist_factor * math.sqrt(total_vectors)))
        nlist = min(nlist, total_vectors)
        quantizer = faiss.IndexFlatL2(feature_dim)
        index = faiss.IndexIVFFlat(quantizer, feature_dim, nlist, faiss.METRIC_L2)
        if hasattr(index, "cp"):
            index.cp.min_points_per_centroid = 1
        training = sample_index_training_vectors(
            chunk_paths,
            max_samples=min(max(nlist * 40, 10000), 200000),
            seed=config.seed,
        )
        index.train(training)
        index.nprobe = max(1, int(round(nlist * config.ivf_nprobe_ratio)))
        del training

    else:
        raise ValueError(f"未対応index_type: {config.index_type}")

    for chunk in iter_chunk_arrays(chunk_paths):
        index.add(np.ascontiguousarray(chunk, dtype=np.float32))

    return index


# =============================================================================
# 5. 評価
# =============================================================================


class StratifiedPixelSampler:
    """正常画素・異常画素を別々にreservoir samplingする。"""

    def __init__(self, max_per_class: int, seed: int):
        self.max_per_class = max_per_class
        self.rng = np.random.default_rng(seed)
        self.scores = {
            0: np.empty(max_per_class, dtype=np.float32),
            1: np.empty(max_per_class, dtype=np.float32),
        }
        self.seen = {0: 0, 1: 0}
        self.stored = {0: 0, 1: 0}

    def update(self, scores: np.ndarray, labels: np.ndarray):
        scores = scores.ravel().astype(np.float32, copy=False)
        labels = labels.ravel().astype(np.uint8, copy=False)

        for class_id in (0, 1):
            class_scores = scores[labels == class_id]
            for score in class_scores:
                seen = self.seen[class_id]
                if seen < self.max_per_class:
                    self.scores[class_id][seen] = score
                    self.stored[class_id] += 1
                else:
                    replacement = int(self.rng.integers(0, seen + 1))
                    if replacement < self.max_per_class:
                        self.scores[class_id][replacement] = score
                self.seen[class_id] += 1

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        normal = self.scores[0][: self.stored[0]]
        anomaly = self.scores[1][: self.stored[1]]
        scores = np.concatenate([normal, anomaly])
        labels = np.concatenate(
            [
                np.zeros(normal.shape[0], dtype=np.uint8),
                np.ones(anomaly.shape[0], dtype=np.uint8),
            ]
        )
        return scores, labels


def safe_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")


def safe_ap(labels: Sequence[int], scores: Sequence[float]) -> float:
    return float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else float("nan")


def normalize_for_heatmap(array: np.ndarray) -> np.ndarray:
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def save_anomaly_map(output_path: Path, image_rgb: np.ndarray, anomaly_map: np.ndarray):
    """異常マップを入力画像へ重畳して保存する。"""
    normalized = normalize_for_heatmap(anomaly_map)
    heatmap = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_JET)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if heatmap.shape[:2] != image_bgr.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    overlay = cv2.addWeighted(image_bgr, 0.55, heatmap, 0.45, 0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def load_gt_mask(record: TestRecord, transform) -> np.ndarray:
    if record.gt_path is None:
        # transform出力サイズを得るためダミー画像を使用する。
        dummy = Image.new("L", (8, 8), color=0)
        return transform(dummy).numpy()[0].astype(np.uint8)

    mask = Image.open(record.gt_path).convert("L")
    tensor = transform(mask)
    return (tensor.numpy()[0] > 0.5).astype(np.uint8)


def infer_single_scale(
    image: Image.Image,
    transform,
    extractor: FeatureExtractor,
    index,
    config: ExperimentConfig,
    output_size: int,
) -> Tuple[float, np.ndarray, float]:
    """1画像・1解像度を推論する。"""
    start = time.perf_counter()
    tensor = transform(image).unsqueeze(0)
    features, (grid_h, grid_w) = extractor.extract(tensor)
    distances, _indices = index.search(
        np.ascontiguousarray(features, dtype=np.float32),
        config.n_neighbors,
    )

    if config.compatibility_first_neighbor_only:
        patch_scores = distances[:, 0]
    else:
        patch_scores = distances.mean(axis=1)

    image_score = float(patch_scores.max())
    anomaly_map = patch_scores.reshape(grid_h, grid_w)
    anomaly_map = cv2.resize(
        anomaly_map,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR,
    )
    anomaly_map = gaussian_filter(anomaly_map, sigma=config.gaussian_sigma)
    elapsed = time.perf_counter() - start
    return image_score, anomaly_map.astype(np.float32), elapsed


def fuse_results(
    low_result: Optional[Tuple[float, np.ndarray, float]],
    high_result: Optional[Tuple[float, np.ndarray, float]],
    config: ExperimentConfig,
) -> Tuple[float, np.ndarray, float]:
    """低解像度・高解像度のスコアと異常マップを融合する。"""
    if low_result is None and high_result is None:
        raise RuntimeError("low/highの両方が無効である。")
    if low_result is None:
        return high_result  # type: ignore[return-value]
    if high_result is None:
        return low_result

    low_score, low_map, low_time = low_result
    high_score, high_map, high_time = high_result
    low_map_up = cv2.resize(low_map, (high_map.shape[1], high_map.shape[0]))

    if config.fusion == "max":
        score = max(low_score, high_score)
        anomaly_map = np.maximum(low_map_up, high_map)
    elif config.fusion == "mean":
        score = (low_score + high_score) / 2.0
        anomaly_map = (low_map_up + high_map) / 2.0
    elif config.fusion == "low":
        return low_result
    elif config.fusion == "high":
        return high_result
    else:
        raise ValueError(f"未対応fusion: {config.fusion}")

    return score, anomaly_map.astype(np.float32), low_time + high_time


# =============================================================================
# 6. カテゴリ単位の実験
# =============================================================================


def run_category(
    category: CategoryInfo,
    config: ExperimentConfig,
    output_root: Path,
) -> Dict[str, object]:
    """1カテゴリについてmemory bank構築・推論・評価を行う。"""
    category_output = output_root / config.name / category.name
    chunk_root = category_output / "feature_chunks"
    category_output.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"実験名       : {config.name}")
    print(f"カテゴリ     : {category.name}")
    print(f"train/good   : {len(category.train_good)}")
    print(f"test         : {len(category.test_records)}")
    print(f"出力先       : {category_output}")
    print("=" * 78)

    with open(category_output / "resolved_config.json", "w", encoding="utf-8") as file:
        json.dump(asdict(config), file, ensure_ascii=False, indent=2)

    device = resolve_device(config)
    model = build_backbone(config, device)
    extractor = FeatureExtractor(model, config, device)

    low_transform = build_transform(config, config.low_size, is_gt=False)
    high_transform = build_transform(config, config.high_size, is_gt=False)
    gt_size = config.high_size if config.use_high else config.low_size
    gt_transform = build_transform(config, gt_size, is_gt=True)

    stage_times: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    # S01: memory bank用特徴抽出
    # -------------------------------------------------------------------------
    start = time.perf_counter()
    chunk_sets: List[ChunkSet] = []

    if config.use_low:
        chunk_sets.append(
            write_feature_chunks(
                category.train_good,
                low_transform,
                extractor,
                config,
                chunk_root / "low",
                "low",
            )
        )

    if config.use_high:
        chunk_sets.append(
            write_feature_chunks(
                category.train_good,
                high_transform,
                extractor,
                config,
                chunk_root / "high",
                "high",
            )
        )

    all_chunk_paths = [path for chunk_set in chunk_sets for path in chunk_set.paths]
    total_vectors = sum(chunk_set.total_vectors for chunk_set in chunk_sets)
    feature_dims = {chunk_set.feature_dim for chunk_set in chunk_sets}
    if len(feature_dims) != 1:
        raise RuntimeError(f"low/highの特徴次元が一致しない: {feature_dims}")
    feature_dim = next(iter(feature_dims))
    stage_times["feature_extraction_sec"] = time.perf_counter() - start

    # -------------------------------------------------------------------------
    # S02: 次元削減
    # -------------------------------------------------------------------------
    reducer = None
    if config.reducer == "ipca":
        start = time.perf_counter()
        reducer = fit_ipca(all_chunk_paths, feature_dim, config)
        reduced_paths = transform_chunks_with_ipca(
            all_chunk_paths,
            reducer,
            chunk_root / "reduced",
        )
        all_chunk_paths = reduced_paths
        feature_dim = min(config.reduced_dim, feature_dim)
        stage_times["reducer_fit_transform_sec"] = time.perf_counter() - start

        # sklearnオブジェクトの保存は環境差があるため、成分をnpzで保存する。
        np.savez(
            category_output / "ipca_parameters.npz",
            components=reducer.components_,
            mean=reducer.mean_,
            explained_variance=reducer.explained_variance_,
        )
    elif config.reducer != "none":
        raise ValueError(f"未対応reducer: {config.reducer}")

    # -------------------------------------------------------------------------
    # S03: memory bank圧縮・FAISS index
    # -------------------------------------------------------------------------
    start = time.perf_counter()
    bank = None

    if config.sampler == "none":
        index = build_faiss_index_from_chunks(
            all_chunk_paths,
            feature_dim,
            total_vectors,
            config,
        )
        bank_size = total_vectors

    elif config.sampler == "random":
        bank = random_sample_chunks(all_chunk_paths, total_vectors, config)
        index = build_faiss_index_from_bank(bank, config)
        bank_size = bank.shape[0]

    elif config.sampler == "minibatch_kmeans":
        bank = minibatch_kmeans_chunks(all_chunk_paths, total_vectors, config)
        index = build_faiss_index_from_bank(bank, config)
        bank_size = bank.shape[0]

    else:
        raise ValueError(f"未対応sampler: {config.sampler}")

    stage_times["bank_and_index_sec"] = time.perf_counter() - start
    faiss.write_index(index, str(category_output / "memory_index.faiss"))
    if bank is not None:
        np.save(category_output / "compressed_memory_bank.npy", bank, allow_pickle=False)

    # -------------------------------------------------------------------------
    # S04: test推論
    # -------------------------------------------------------------------------
    image_labels: List[int] = []
    image_scores: List[float] = []
    rows: List[Dict[str, object]] = []
    pixel_sampler = StratifiedPixelSampler(
        max_per_class=config.pixel_metric_sample_per_class,
        seed=config.seed,
    )

    inference_start = time.perf_counter()

    for record_index, record in enumerate(category.test_records):
        image = Image.open(record.image_path).convert("RGB")

        low_result = None
        high_result = None

        if config.use_low:
            low_result = infer_single_scale(
                image,
                low_transform,
                extractor,
                index,
                config,
                config.low_size,
            )

        if config.use_high:
            high_result = infer_single_scale(
                image,
                high_transform,
                extractor,
                index,
                config,
                config.high_size,
            )

        fused_score, fused_map, inference_time = fuse_results(
            low_result,
            high_result,
            config,
        )

        gt = load_gt_mask(record, gt_transform)
        if fused_map.shape != gt.shape:
            fused_map = cv2.resize(
                fused_map,
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        image_labels.append(record.label)
        image_scores.append(fused_score)
        pixel_sampler.update(fused_map, gt)

        rows.append(
            {
                "category": category.name,
                "image_path": str(record.image_path),
                "file_name": record.image_path.name,
                "defect_type": record.defect_type,
                "label": record.label,
                "score": fused_score,
                "inference_sec": inference_time,
            }
        )

        if config.save_anomaly_maps:
            image_rgb = np.asarray(image)
            output_path = (
                category_output
                / "anomaly_maps"
                / record.defect_type
                / f"{record.image_path.stem}_score_{fused_score:.6f}.png"
            )
            save_anomaly_map(output_path, image_rgb, fused_map)

        if record_index % 20 == 0:
            print(
                f"test {record_index + 1}/{len(category.test_records)}: "
                f"{record.image_path.name}, score={fused_score:.6f}"
            )

    stage_times["test_total_sec"] = time.perf_counter() - inference_start

    # -------------------------------------------------------------------------
    # S05: 評価・保存
    # -------------------------------------------------------------------------
    prediction_df = pd.DataFrame(rows)
    prediction_df.to_csv(
        category_output / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pixel_scores, pixel_labels = pixel_sampler.arrays()
    inference_times = prediction_df["inference_sec"].to_numpy(dtype=float)

    metrics: Dict[str, object] = {
        "category": category.name,
        "experiment": config.name,
        "train_good_count": len(category.train_good),
        "test_count": len(category.test_records),
        "test_good_count": int(sum(r.label == 0 for r in category.test_records)),
        "test_anomaly_count": int(sum(r.label == 1 for r in category.test_records)),
        "feature_dim": feature_dim,
        "memory_bank_vectors_before": total_vectors,
        "memory_bank_vectors_after": bank_size,
        "compression_ratio_actual": bank_size / total_vectors,
        "image_auroc": safe_auroc(image_labels, image_scores),
        "image_ap": safe_ap(image_labels, image_scores),
        # 画素指標は正常・異常画素を層化reservoir samplingした近似値である。
        "pixel_auroc_sampled": safe_auroc(pixel_labels.tolist(), pixel_scores.tolist()),
        "pixel_ap_sampled": safe_ap(pixel_labels.tolist(), pixel_scores.tolist()),
        "pixel_sample_normal": int((pixel_labels == 0).sum()),
        "pixel_sample_anomaly": int((pixel_labels == 1).sum()),
        "inference_mean_sec": float(inference_times.mean()),
        "inference_p50_sec": float(np.percentile(inference_times, 50)),
        "inference_p95_sec": float(np.percentile(inference_times, 95)),
        "inference_p99_sec": float(np.percentile(inference_times, 99)),
        **stage_times,
    }

    with open(category_output / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    pd.DataFrame([metrics]).to_csv(
        category_output / "metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    extractor.close()
    del model, index, bank
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not config.save_intermediate_chunks:
        shutil.rmtree(chunk_root, ignore_errors=True)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


# =============================================================================
# 7. 実行制御
# =============================================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="MVTec AD形式対応 SaikuPatchCore実験ランナー"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="MVTec AD形式データセットのルート",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="実験出力ルート",
    )
    parser.add_argument(
        "--preset-name",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="冒頭のACTIVE_EXPERIMENTをコマンドラインから上書きする",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="実行カテゴリを明示指定。未指定時は冒頭設定を使用する",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="データセット構造だけ検証し、モデル処理を行わない",
    )
    return parser.parse_args()


def select_categories(
    detected: Dict[str, Path],
    cli_categories: Optional[Sequence[str]],
) -> List[str]:
    """カテゴリ選択設定を解決する。"""
    if cli_categories:
        selected = list(cli_categories)
    elif CATEGORY_SELECTION_MODE == "manual":
        selected = list(MANUAL_CATEGORIES)
    elif CATEGORY_SELECTION_MODE == "auto":
        selected = sorted(detected.keys())
    else:
        raise ValueError(
            f"CATEGORY_SELECTION_MODEはauto/manualのみ: {CATEGORY_SELECTION_MODE}"
        )

    unknown = sorted(set(selected) - set(detected))
    if unknown:
        raise KeyError(
            f"指定カテゴリがデータセット内にない: {unknown}\n"
            f"検出カテゴリ: {sorted(detected)}"
        )
    return selected


def main():
    args = parse_args()
    config = PRESETS[args.preset_name] if args.preset_name else ACTIVE_EXPERIMENT
    set_seed(config.seed)

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    detected = discover_category_paths(dataset_root)
    selected = select_categories(detected, args.categories)

    print("検出カテゴリ:", sorted(detected.keys()))
    print("実行カテゴリ:", selected)
    print("実験プリセット:", config.name)

    category_infos: List[CategoryInfo] = []
    for category_name in selected:
        info = load_category_info(category_name, detected[category_name])
        category_infos.append(info)
        print(
            f"[検証成功] {category_name}: "
            f"train/good={len(info.train_good)}, test={len(info.test_records)}"
        )

    if args.validate_only:
        print("validate-onlyのため終了する。モデル処理は未実行である。")
        return

    run_started = datetime.now().isoformat(timespec="seconds")
    all_metrics: List[Dict[str, object]] = []

    for category_info in category_infos:
        metrics = run_category(category_info, config, output_root)
        all_metrics.append(metrics)

    summary_dir = output_root / config.name
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(
        summary_dir / "all_categories_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    run_info = {
        "started_at": run_started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "categories": selected,
        "config": asdict(config),
    }
    with open(summary_dir / "run_info.json", "w", encoding="utf-8") as file:
        json.dump(run_info, file, ensure_ascii=False, indent=2)

    print("全カテゴリの実験が完了した。")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
