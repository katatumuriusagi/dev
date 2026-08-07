# -*- coding: utf-8 -*-
"""
04_run_saiku_patchcore_baselines.py
===================================

Canonical ROI datasetに対するSaikuPatchCore系基準実験ランナー。

今回の役割
----------
- ROI校正・canonical化を入れた後の新しいbaselineを固定する。
- global k-means 1%、random 1%、解像度、fusion、FAISS方式を比較する。
- ROI外patchはmemory bankへ登録せず、test異常スコアからも除外する。
- 全train特徴を巨大Python listへ保持せず、一旦chunkへ保存する。

まだ入れないもの
----------------
- 正常モード保存型層化coreset
- GNN / sparse graph context

これらはbaselineと正常モード存在確認が終わった後に別実験として追加する。
寄与を混ぜないためである。

元SaikuPatchCoreとの互換点
-------------------------
- WideResNet50-2 pretrained
- layer2[-1], layer3[-1]の特徴
- 256 / 512の2解像度
- AvgPool2d(kernel=3, stride=1, padding=1)
- n_neighbors=3
- 検索した3近傍のうち第1近傍距離をpatch anomaly scoreに使用

注意
----
本版はメモリ安全化・ROI mask対応のため実装詳細を整理している。
元コードとの完全同一値を保証するものではない。
小規模subsetで順位相関・AUROC差・異常マップを確認してから本実験へ進むこと。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from khi_roi_common import IMAGE_EXTENSIONS, imread_unicode, imwrite_unicode, save_json


# =============================================================================
# 0. 実験プリセット
# =============================================================================


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    use_low: bool = True
    use_high: bool = True
    low_size: int = 256
    high_size: int = 512
    fusion: str = "mean"            # mean / max / low / high

    # memory bank
    sampler: str = "minibatch_kmeans"  # none / random / minibatch_kmeans
    sampler_ratio: float = 0.01
    max_prototypes: int = 10000
    kmeans_batch_size: int = 4096
    kmeans_max_iter: int = 100

    # FAISS
    index_type: str = "ivf"         # flat / ivf / hnsw
    n_neighbors: int = 3
    ivf_nprobe_ratio: float = 0.01
    hnsw_m: int = 32
    hnsw_ef_search: int = 64

    # backbone
    pool_kernel: int = 3
    pool_stride: int = 1
    pool_padding: int = 1
    train_batch_size: int = 2
    num_workers: int = 0
    seed: int = 319
    gaussian_sigma: float = 4.0
    save_anomaly_maps: bool = True
    keep_feature_chunks: bool = False

    # 画素評価時、各画像から最大何画素を使うか。
    # ROI内だけから均等ランダム抽出する。
    pixel_sample_per_image: int = 20000


E00_BASELINE_NO_COMPRESSION = ExperimentConfig(
    name="E00_baseline_no_compression",
    sampler="none",
    sampler_ratio=1.0,
    max_prototypes=250000,
)

E01_KMEANS_1PERCENT = replace(
    E00_BASELINE_NO_COMPRESSION,
    name="E01_global_kmeans_1percent",
    sampler="minibatch_kmeans",
    sampler_ratio=0.01,
    max_prototypes=10000,
)

E02_RANDOM_1PERCENT = replace(
    E01_KMEANS_1PERCENT,
    name="E02_random_1percent",
    sampler="random",
)

E03_LOW256_ONLY = replace(
    E01_KMEANS_1PERCENT,
    name="E03_low256_only",
    use_low=True,
    use_high=False,
    fusion="low",
)

E04_HIGH512_ONLY = replace(
    E01_KMEANS_1PERCENT,
    name="E04_high512_only",
    use_low=False,
    use_high=True,
    fusion="high",
)

E05_DUAL_MAX = replace(
    E01_KMEANS_1PERCENT,
    name="E05_dual_max",
    fusion="max",
)

E06_FLATL2 = replace(
    E01_KMEANS_1PERCENT,
    name="E06_flatl2_diagnostic",
    index_type="flat",
)

E07_HNSW = replace(
    E01_KMEANS_1PERCENT,
    name="E07_hnsw",
    index_type="hnsw",
)


# -----------------------------------------------------------------------------
# 実験切替
# 行う実験だけ有効にする。コマンドライン--preset-nameでも上書きできる。
# -----------------------------------------------------------------------------
ACTIVE_EXPERIMENT = E01_KMEANS_1PERCENT
# ACTIVE_EXPERIMENT = E00_BASELINE_NO_COMPRESSION
# ACTIVE_EXPERIMENT = E02_RANDOM_1PERCENT
# ACTIVE_EXPERIMENT = E03_LOW256_ONLY
# ACTIVE_EXPERIMENT = E04_HIGH512_ONLY
# ACTIVE_EXPERIMENT = E05_DUAL_MAX
# ACTIVE_EXPERIMENT = E06_FLATL2
# ACTIVE_EXPERIMENT = E07_HNSW

PRESETS = {
    cfg.name: cfg
    for cfg in [
        E00_BASELINE_NO_COMPRESSION,
        E01_KMEANS_1PERCENT,
        E02_RANDOM_1PERCENT,
        E03_LOW256_ONLY,
        E04_HIGH512_ONLY,
        E05_DUAL_MAX,
        E06_FLATL2,
        E07_HNSW,
    ]
}


# =============================================================================
# 1. MVTec AD形式読込み
# =============================================================================


@dataclass(frozen=True)
class TestRecord:
    image_path: Path
    roi_path: Path
    gt_path: Optional[Path]
    label: int
    defect_type: str


@dataclass(frozen=True)
class CategoryInfo:
    name: str
    path: Path
    train_images: Tuple[Path, ...]
    train_roi_paths: Tuple[Path, ...]
    test_records: Tuple[TestRecord, ...]


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def is_category_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "train" / "good").is_dir()
        and (path / "test").is_dir()
    )


def discover_categories(root: Path) -> Dict[str, Path]:
    root = Path(root).resolve()
    if is_category_dir(root):
        return {root.name: root}
    result = {
        p.name: p
        for p in sorted(root.iterdir())
        if is_category_dir(p)
    }
    if not result:
        raise FileNotFoundError(f"MVTec AD形式カテゴリがない: {root}")
    return result


def find_gt(category: Path, defect: str, image: Path) -> Optional[Path]:
    if defect == "good":
        return None
    folder = category / "ground_truth" / defect
    for stem in (f"{image.stem}_mask", image.stem):
        for ext in IMAGE_EXTENSIONS:
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def load_category(name: str, path: Path) -> CategoryInfo:
    train = tuple(list_images(path / "train" / "good"))
    if not train:
        raise ValueError(f"{name}: train/goodが空")

    train_roi = tuple(
        path / "roi_masks" / "train" / "good" / f"{p.stem}_roi.png"
        for p in train
    )
    for p in train_roi:
        if not p.exists():
            raise FileNotFoundError(f"ROI maskがない: {p}")

    tests: List[TestRecord] = []
    for defect_dir in sorted(p for p in (path / "test").iterdir() if p.is_dir()):
        defect = defect_dir.name
        label = 0 if defect == "good" else 1
        for image in list_images(defect_dir):
            roi = path / "roi_masks" / "test" / defect / f"{image.stem}_roi.png"
            if not roi.exists():
                raise FileNotFoundError(f"ROI maskがない: {roi}")
            gt = find_gt(path, defect, image)
            if label == 1 and gt is None:
                raise FileNotFoundError(f"GTがない: {image}")
            tests.append(TestRecord(image, roi, gt, label, defect))

    return CategoryInfo(name, path, train, train_roi, tuple(tests))


class TrainDataset(Dataset):
    def __init__(self, images: Sequence[Path], roi_paths: Sequence[Path], size: int):
        self.images = list(images)
        self.roi_paths = list(roi_paths)
        self.size = int(size)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        tensor = self.image_transform(image)

        roi = imread_unicode(self.roi_paths[index], cv2.IMREAD_GRAYSCALE)
        roi = cv2.resize(roi, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        roi = torch.from_numpy((roi > 127).astype(np.float32))[None, ...]
        return tensor, roi, str(self.images[index])


# =============================================================================
# 2. Backbone / SaikuPatchCore特徴
# =============================================================================


class FeatureExtractor:
    def __init__(self, device: torch.device, config: ExperimentConfig):
        try:
            weights = torchvision.models.Wide_ResNet50_2_Weights.DEFAULT
            self.model = torchvision.models.wide_resnet50_2(weights=weights)
        except AttributeError:
            self.model = torchvision.models.wide_resnet50_2(pretrained=True)

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval().to(device)

        self.device = device
        self.config = config
        self.outputs: List[torch.Tensor] = []
        self.model.layer2[-1].register_forward_hook(self._hook)
        self.model.layer3[-1].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.outputs.append(output)

    def embedding_concat_legacy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """元SaikuPatchCoreのembedding_concatと同じ考え方。"""
        b, c1, h1, w1 = x.size()
        _, c2, h2, w2 = y.size()
        s = int(h1 / h2)
        if s < 1 or h1 % h2 != 0 or w1 % w2 != 0:
            # 想定外backboneでは安全側へinterpolate。
            y_up = F.interpolate(y, size=(h1, w1), mode="nearest")
            return torch.cat([x, y_up], dim=1)

        unfolded = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        unfolded = unfolded.view(b, c1, -1, h2, w2)
        z = torch.zeros(
            b,
            c1 + c2,
            unfolded.size(2),
            h2,
            w2,
            device=x.device,
            dtype=x.dtype,
        )
        for i in range(unfolded.size(2)):
            z[:, :, i, :, :] = torch.cat((unfolded[:, :, i, :, :], y), 1)
        z = z.view(b, -1, h2 * w2)
        return F.fold(z, kernel_size=s, output_size=(h1, w1), stride=s)

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        self.outputs = []
        _ = self.model(x.to(self.device, non_blocking=True))
        if len(self.outputs) != 2:
            raise RuntimeError(f"hook出力数が不正: {len(self.outputs)}")

        pool = torch.nn.AvgPool2d(
            kernel_size=self.config.pool_kernel,
            stride=self.config.pool_stride,
            padding=self.config.pool_padding,
        )
        low = pool(self.outputs[0])
        high = pool(self.outputs[1])
        embedding = self.embedding_concat_legacy(low, high)
        return embedding


def flatten_valid_features(
    embedding: torch.Tensor,
    roi_mask: torch.Tensor,
) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
    """
    embedding: B,C,H,W
    roi_mask: B,1,inputH,inputW

    ROI maskをfeature mapサイズへnearest縮小し、ROI内patchだけ返す。
    """
    b, c, h, w = embedding.shape
    valid = F.interpolate(roi_mask.float(), size=(h, w), mode="nearest") > 0.5
    features = embedding.permute(0, 2, 3, 1).contiguous()

    rows = features[valid[:, 0]].detach().cpu().numpy().astype(np.float32)
    valid_np = valid[:, 0].detach().cpu().numpy().astype(bool)
    return rows, (h, w), valid_np


# =============================================================================
# 3. train特徴をchunk保存
# =============================================================================


def extract_train_chunks(
    info: CategoryInfo,
    extractor: FeatureExtractor,
    config: ExperimentConfig,
    chunk_root: Path,
    size: int,
    resolution_name: str,
) -> Tuple[List[Path], int, int]:
    folder = chunk_root / resolution_name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

    dataset = TrainDataset(info.train_images, info.train_roi_paths, size=size)
    loader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    chunk_paths: List[Path] = []
    total = 0
    dim = 0
    for batch_index, (images, rois, paths) in enumerate(loader):
        embedding = extractor.extract(images)
        features, _, _ = flatten_valid_features(embedding, rois.to(extractor.device))
        if features.size == 0:
            continue
        dim = int(features.shape[1])
        total += int(features.shape[0])
        chunk_path = folder / f"chunk_{batch_index:06d}.npy"
        np.save(chunk_path, features)
        chunk_paths.append(chunk_path)

        del embedding, features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (batch_index + 1) % 50 == 0:
            print(f"  [{resolution_name}] batch={batch_index+1}, vectors={total}")

    return chunk_paths, total, dim


def iter_chunks(paths: Sequence[Path]):
    for path in paths:
        yield np.load(path, mmap_mode="r")


# =============================================================================
# 4. Memory bank sampler
# =============================================================================


def target_prototypes(total: int, config: ExperimentConfig) -> int:
    if config.sampler == "none":
        return total
    target = max(1, int(round(total * config.sampler_ratio)))
    return min(target, config.max_prototypes, total)


def collect_all_with_safety(paths: Sequence[Path], total: int, dim: int, limit: int) -> np.ndarray:
    if total > limit:
        raise MemoryError(
            f"圧縮なしmemory bankは{total} vectorsあり安全上限{limit}を超える。"
            "小規模subsetでのみE00を使うか、E01 k-means 1%を使用すること。"
        )
    result = np.empty((total, dim), dtype=np.float32)
    cursor = 0
    for arr in iter_chunks(paths):
        n = len(arr)
        result[cursor:cursor+n] = arr
        cursor += n
    return result


def random_sample_stream(
    paths: Sequence[Path],
    total: int,
    target: int,
    dim: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(total, size=target, replace=False))
    result = np.empty((target, dim), dtype=np.float32)

    global_start = 0
    out_cursor = 0
    for arr in iter_chunks(paths):
        global_end = global_start + len(arr)
        left = np.searchsorted(chosen, global_start, side="left")
        right = np.searchsorted(chosen, global_end, side="left")
        if right > left:
            local = chosen[left:right] - global_start
            n = right - left
            result[out_cursor:out_cursor+n] = np.asarray(arr[local], dtype=np.float32)
            out_cursor += n
        global_start = global_end

    return result


def kmeans_stream(
    paths: Sequence[Path],
    total: int,
    target: int,
    dim: int,
    config: ExperimentConfig,
) -> np.ndarray:
    if target >= total:
        return collect_all_with_safety(paths, total, dim, limit=config.max_prototypes)

    print(f"MiniBatchKMeans: total={total}, centroids={target}")
    model = MiniBatchKMeans(
        n_clusters=target,
        random_state=config.seed,
        batch_size=max(config.kmeans_batch_size, target),
        max_iter=config.kmeans_max_iter,
        n_init=1,
        reassignment_ratio=0.01,
    )

    # 初回partial_fitには少なくともn_clusters行必要なのでbufferする。
    first_parts: List[np.ndarray] = []
    first_count = 0
    initialized = False

    for arr_mm in iter_chunks(paths):
        arr = np.asarray(arr_mm, dtype=np.float32)
        if not initialized:
            first_parts.append(arr)
            first_count += len(arr)
            if first_count >= target:
                first = np.concatenate(first_parts, axis=0)
                model.partial_fit(first)
                initialized = True
                first_parts = []
        else:
            model.partial_fit(arr)

    if not initialized:
        first = np.concatenate(first_parts, axis=0)
        if len(first) < target:
            raise RuntimeError("k-means初期化に必要なvector数が不足。")
        model.partial_fit(first)

    return np.asarray(model.cluster_centers_, dtype=np.float32)


def build_memory_bank(
    chunk_paths: Sequence[Path],
    total: int,
    dim: int,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, Dict]:
    target = target_prototypes(total, config)
    t0 = time.perf_counter()

    if config.sampler == "none":
        bank = collect_all_with_safety(
            chunk_paths,
            total,
            dim,
            limit=config.max_prototypes,
        )
    elif config.sampler == "random":
        bank = random_sample_stream(
            chunk_paths,
            total,
            target,
            dim,
            seed=config.seed,
        )
    elif config.sampler == "minibatch_kmeans":
        bank = kmeans_stream(
            chunk_paths,
            total,
            target,
            dim,
            config,
        )
    else:
        raise ValueError(f"未対応sampler: {config.sampler}")

    elapsed = time.perf_counter() - t0
    bank = np.ascontiguousarray(bank, dtype=np.float32)
    return bank, {
        "vectors_before": total,
        "vectors_after": int(len(bank)),
        "compression_ratio_actual": float(len(bank) / total),
        "sampler_sec": elapsed,
    }


# =============================================================================
# 5. FAISS index
# =============================================================================


def build_index(bank: np.ndarray, config: ExperimentConfig):
    dim = int(bank.shape[1])
    t0 = time.perf_counter()

    if config.index_type == "flat":
        index = faiss.IndexFlatL2(dim)
        index.add(bank)
        detail = {"index_type": "flat"}

    elif config.index_type == "ivf":
        nlist = max(1, int(4 * math.sqrt(len(bank))))
        nlist = min(nlist, len(bank))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        if hasattr(index, "cp"):
            index.cp.min_points_per_centroid = 1
        index.train(bank)
        index.add(bank)
        index.nprobe = max(1, min(nlist, int(round(nlist * config.ivf_nprobe_ratio))))
        detail = {
            "index_type": "ivf",
            "nlist": nlist,
            "nprobe": int(index.nprobe),
        }

    elif config.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, config.hnsw_m, faiss.METRIC_L2)
        index.hnsw.efSearch = config.hnsw_ef_search
        index.add(bank)
        detail = {
            "index_type": "hnsw",
            "M": config.hnsw_m,
            "efSearch": config.hnsw_ef_search,
        }

    else:
        raise ValueError(f"未対応index_type: {config.index_type}")

    detail["index_build_sec"] = time.perf_counter() - t0
    return index, detail


# =============================================================================
# 6. test推論
# =============================================================================


def image_tensor(image_bgr: np.ndarray, size: int) -> torch.Tensor:
    # canonical画像はすでに正方形。正方形→正方形なので形状は歪まない。
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tfm = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return tfm(pil).unsqueeze(0)


def infer_resolution(
    image_bgr: np.ndarray,
    roi_mask: np.ndarray,
    size: int,
    extractor: FeatureExtractor,
    index,
    config: ExperimentConfig,
) -> Tuple[float, np.ndarray, float]:
    t0 = time.perf_counter()
    tensor = image_tensor(image_bgr, size)
    roi_small = cv2.resize(roi_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    roi_tensor = torch.from_numpy((roi_small > 127).astype(np.float32))[None, None, ...]

    embedding = extractor.extract(tensor)
    _, _, h, w = embedding.shape
    valid = F.interpolate(roi_tensor.to(extractor.device), size=(h, w), mode="nearest")[:, 0] > 0.5
    features = embedding.permute(0, 2, 3, 1).contiguous()[valid]
    features_np = np.ascontiguousarray(
        features.detach().cpu().numpy().astype(np.float32)
    )

    if len(features_np) == 0:
        raise RuntimeError("test画像でvalid ROI patchが0件。")

    distances, _ = index.search(features_np, config.n_neighbors)
    patch_score = distances[:, 0]
    score_img = float(np.max(patch_score))

    grid = np.zeros((h, w), dtype=np.float32)
    valid_np = valid[0].detach().cpu().numpy().astype(bool)
    grid[valid_np] = patch_score

    amap = cv2.resize(grid, (size, size), interpolation=cv2.INTER_LINEAR)
    amap = gaussian_filter(amap, sigma=config.gaussian_sigma)
    amap[roi_small <= 127] = 0.0

    elapsed = time.perf_counter() - t0
    return score_img, amap.astype(np.float32), elapsed


def fuse_scores(
    low_score: Optional[float],
    high_score: Optional[float],
    low_map: Optional[np.ndarray],
    high_map: Optional[np.ndarray],
    config: ExperimentConfig,
) -> Tuple[float, np.ndarray]:
    target_size = config.high_size if config.use_high else config.low_size

    if config.fusion == "low":
        return float(low_score), low_map
    if config.fusion == "high":
        return float(high_score), high_map

    low_up = cv2.resize(low_map, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    high = high_map

    if config.fusion == "mean":
        return float((low_score + high_score) / 2.0), (low_up + high) / 2.0
    if config.fusion == "max":
        return float(max(low_score, high_score)), np.maximum(low_up, high)
    raise ValueError(f"未対応fusion: {config.fusion}")


def save_map(path: Path, image_bgr: np.ndarray, amap: np.ndarray, roi: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if amap.max() > amap.min():
        norm = (amap - amap.min()) / (amap.max() - amap.min())
    else:
        norm = np.zeros_like(amap)
    heat = cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_JET)
    image = cv2.resize(image_bgr, (amap.shape[1], amap.shape[0]), interpolation=cv2.INTER_AREA)
    overlay = cv2.addWeighted(image, 0.55, heat, 0.45, 0)
    roi2 = cv2.resize(roi, (amap.shape[1], amap.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay[roi2 <= 127] = image[roi2 <= 127]
    imwrite_unicode(path, overlay)


def sample_pixels(
    amap: np.ndarray,
    gt: np.ndarray,
    roi: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    valid = roi > 127
    scores = amap[valid].reshape(-1)
    labels = (gt[valid] > 0).astype(np.uint8).reshape(-1)
    if len(scores) > max_samples:
        ids = rng.choice(len(scores), size=max_samples, replace=False)
        scores = scores[ids]
        labels = labels[ids]
    return scores.astype(np.float32), labels


# =============================================================================
# 7. Category experiment
# =============================================================================


def run_category(
    info: CategoryInfo,
    config: ExperimentConfig,
    output_dir: Path,
    device: torch.device,
):
    rng = np.random.default_rng(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_root = output_dir / "_feature_chunks"
    if chunk_root.exists():
        shutil.rmtree(chunk_root)
    chunk_root.mkdir(parents=True)

    extractor = FeatureExtractor(device, config)

    # ----- train feature extraction -----
    t_feature = time.perf_counter()
    all_chunk_paths: List[Path] = []
    total_vectors = 0
    dim = 0

    if config.use_low:
        paths, n, d = extract_train_chunks(
            info, extractor, config, chunk_root, config.low_size, "low"
        )
        all_chunk_paths.extend(paths)
        total_vectors += n
        dim = d

    if config.use_high:
        paths, n, d = extract_train_chunks(
            info, extractor, config, chunk_root, config.high_size, "high"
        )
        all_chunk_paths.extend(paths)
        total_vectors += n
        dim = d

    feature_extraction_sec = time.perf_counter() - t_feature
    print("train valid patch vectors:", total_vectors)

    # ----- sampler -----
    bank, bank_info = build_memory_bank(
        all_chunk_paths,
        total_vectors,
        dim,
        config,
    )
    np.save(output_dir / "memory_bank.npy", bank)

    # ----- index -----
    index, index_info = build_index(bank, config)
    faiss.write_index(index, str(output_dir / "memory_index.faiss"))

    # ----- test -----
    image_labels: List[int] = []
    image_scores: List[float] = []
    pixel_scores: List[np.ndarray] = []
    pixel_labels: List[np.ndarray] = []
    prediction_rows: List[Dict] = []
    inference_times: List[float] = []

    t_test_all = time.perf_counter()
    for i, record in enumerate(info.test_records, start=1):
        image = imread_unicode(record.image_path, cv2.IMREAD_COLOR)
        roi = imread_unicode(record.roi_path, cv2.IMREAD_GRAYSCALE)
        if roi.shape != image.shape[:2]:
            roi = cv2.resize(roi, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        low_score = low_map = low_sec = None
        high_score = high_map = high_sec = None

        if config.use_low:
            low_score, low_map, low_sec = infer_resolution(
                image, roi, config.low_size, extractor, index, config
            )
        if config.use_high:
            high_score, high_map, high_sec = infer_resolution(
                image, roi, config.high_size, extractor, index, config
            )

        score, amap = fuse_scores(
            low_score, high_score, low_map, high_map, config
        )
        inference_sec = float((low_sec or 0.0) + (high_sec or 0.0))
        inference_times.append(inference_sec)

        target_size = amap.shape[0]
        roi_eval = cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        amap[roi_eval <= 127] = 0.0

        if record.gt_path is None:
            gt = np.zeros((target_size, target_size), dtype=np.uint8)
        else:
            gt = imread_unicode(record.gt_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            gt = np.where(gt > 0, 255, 0).astype(np.uint8)

        ps, pl = sample_pixels(
            amap, gt, roi_eval,
            max_samples=config.pixel_sample_per_image,
            rng=rng,
        )
        pixel_scores.append(ps)
        pixel_labels.append(pl)

        image_labels.append(record.label)
        image_scores.append(score)
        prediction_rows.append(
            {
                "category": info.name,
                "file_name": record.image_path.name,
                "defect_type": record.defect_type,
                "label": record.label,
                "score": score,
                "low_score": low_score,
                "high_score": high_score,
                "inference_sec": inference_sec,
            }
        )

        if config.save_anomaly_maps:
            save_map(
                output_dir / "anomaly_maps" / record.defect_type / f"{record.image_path.stem}_amap.jpg",
                image,
                amap,
                roi_eval,
            )

        if i % 50 == 0 or i == len(info.test_records):
            print(f"test {i}/{len(info.test_records)}")

    test_total_sec = time.perf_counter() - t_test_all

    # ----- metrics -----
    labels_np = np.asarray(image_labels, dtype=np.uint8)
    scores_np = np.asarray(image_scores, dtype=np.float64)
    if len(np.unique(labels_np)) >= 2:
        image_auroc = float(roc_auc_score(labels_np, scores_np))
        image_ap = float(average_precision_score(labels_np, scores_np))
    else:
        image_auroc = None
        image_ap = None

    px_label = np.concatenate(pixel_labels) if pixel_labels else np.array([], dtype=np.uint8)
    px_score = np.concatenate(pixel_scores) if pixel_scores else np.array([], dtype=np.float32)
    if len(px_label) and len(np.unique(px_label)) >= 2:
        pixel_auroc = float(roc_auc_score(px_label, px_score))
        pixel_ap = float(average_precision_score(px_label, px_score))
    else:
        pixel_auroc = None
        pixel_ap = None

    metrics = {
        "category": info.name,
        "preset": config.name,
        "device": str(device),
        "train_good_images": len(info.train_images),
        "test_images": len(info.test_records),
        "test_normal_images": int(sum(r.label == 0 for r in info.test_records)),
        "test_anomaly_images": int(sum(r.label == 1 for r in info.test_records)),
        "image_auroc": image_auroc,
        "image_ap": image_ap,
        "pixel_auroc_sampled": pixel_auroc,
        "pixel_ap_sampled": pixel_ap,
        "pixel_samples": int(len(px_label)),
        "feature_extraction_sec": feature_extraction_sec,
        "test_total_sec": test_total_sec,
        "inference_mean_sec": float(np.mean(inference_times)) if inference_times else None,
        "inference_p95_sec": float(np.percentile(inference_times, 95)) if inference_times else None,
        "memory_bank_mb": float(bank.nbytes / (1024 ** 2)),
    }
    metrics.update(bank_info)
    metrics.update(index_info)

    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_json(output_dir / "metrics.json", metrics)
    pd.DataFrame([metrics]).to_csv(
        output_dir / "metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_json(output_dir / "resolved_config.json", asdict(config))

    if not config.keep_feature_chunks:
        shutil.rmtree(chunk_root, ignore_errors=True)

    print("----------- Results -----------")
    for key in [
        "image_auroc",
        "image_ap",
        "pixel_auroc_sampled",
        "pixel_ap_sampled",
        "vectors_before",
        "vectors_after",
        "memory_bank_mb",
        "feature_extraction_sec",
        "sampler_sec",
        "index_build_sec",
        "inference_mean_sec",
        "inference_p95_sec",
        "test_total_sec",
    ]:
        print(f"{key}: {metrics.get(key)}")
    return metrics


# =============================================================================
# 8. CLI
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="ROI-aware SaikuPatchCore baseline experiments")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--preset-name", choices=sorted(PRESETS.keys()), default=None)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    config = PRESETS[args.preset_name] if args.preset_name else ACTIVE_EXPERIMENT

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAが使用できない。")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    categories = discover_categories(args.dataset_root)
    if args.categories:
        missing = [c for c in args.categories if c not in categories]
        if missing:
            raise KeyError(f"カテゴリがない: {missing}")
        categories = {c: categories[c] for c in args.categories}

    print("Preset:", config.name)
    print("Device:", device)
    print("Categories:", list(categories.keys()))

    all_metrics = []
    for category, path in categories.items():
        info = load_category(category, path)
        metrics = run_category(
            info,
            config,
            args.output_root / config.name / category,
            device,
        )
        all_metrics.append(metrics)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_metrics).to_csv(
        args.output_root / config.name / "all_categories_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
