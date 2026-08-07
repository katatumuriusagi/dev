# -*- coding: utf-8 -*-
"""
03_analyze_normal_modes.py
==========================

canonical化したtrain/goodだけから正常外観のモード構造を探索する。
異常画像・GTは一切使用しない。

目的
----
- 明るい/暗い、低/高コントラスト等の正常変動を定量化する。
- global coresetが少数正常モードを失うか検証する前段階として、
  normal_mode_idを作る。
- 本ファイルのk-means結果を最終的な真の「正常モード」と決め打ちしない。
  cluster数・安定性・可視化・mode別FPRを確認してから仕様を固定する。

出力
----
normal_descriptors.csv
normal_modes.csv
normal_mode_summary.csv
normal_mode_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from khi_roi_common import imread_unicode, list_images, save_json


def descriptor(image_bgr: np.ndarray, roi_mask: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    valid = roi_mask > 127
    if not np.any(valid):
        raise ValueError("ROI maskが空。")

    values = gray[valid].astype(np.float32)
    p05, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])

    # gradient量。反射・エッジの強さの粗い代理。
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    grad = mag[valid]

    return {
        "mean_intensity": float(values.mean()),
        "std_intensity": float(values.std()),
        "p05": float(p05),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "dynamic_p95_p05": float(p95 - p05),
        "iqr": float(p75 - p25),
        "mean_gradient": float(grad.mean()),
        "p95_gradient": float(np.percentile(grad, 95)),
        "roi_ratio": float(valid.mean()),
    }


def find_categories(dataset_root: Path) -> Dict[str, Path]:
    if (dataset_root / "train" / "good").is_dir():
        return {dataset_root.name: dataset_root}
    result = {
        p.name: p
        for p in sorted(dataset_root.iterdir())
        if p.is_dir() and (p / "train" / "good").is_dir()
    }
    if not result:
        raise FileNotFoundError("canonical MVTecカテゴリが見つからない。")
    return result


def analyze_category(category: str, path: Path, output_dir: Path, n_modes: int, seed: int):
    images = list_images(path / "train" / "good")
    rows: List[Dict] = []

    for i, image_path in enumerate(images, start=1):
        roi_path = path / "roi_masks" / "train" / "good" / f"{image_path.stem}_roi.png"
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI maskがない: {roi_path}")

        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        mask = imread_unicode(roi_path, cv2.IMREAD_GRAYSCALE)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        row = {
            "category": category,
            "file_name": image_path.name,
            "image_path": str(image_path),
        }
        row.update(descriptor(image, mask))
        rows.append(row)

        if i % 200 == 0 or i == len(images):
            print(f"[{category}] descriptor {i}/{len(images)}")

    df = pd.DataFrame(rows)
    feature_columns = [
        "mean_intensity",
        "std_intensity",
        "p05",
        "p25",
        "p50",
        "p75",
        "p95",
        "dynamic_p95_p05",
        "iqr",
        "mean_gradient",
        "p95_gradient",
        "roi_ratio",
    ]

    if len(df) < n_modes:
        raise ValueError(
            f"train/good={len(df)}枚に対してn_modes={n_modes}は大きすぎる。"
        )

    x = df[feature_columns].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    z = scaler.fit_transform(x)

    model = KMeans(n_clusters=n_modes, random_state=seed, n_init=20)
    labels = model.fit_predict(z)
    df["normal_mode_id"] = [f"M{int(v):02d}" for v in labels]
    df["distance_to_mode_center"] = np.linalg.norm(z - model.cluster_centers_[labels], axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "normal_descriptors.csv", index=False, encoding="utf-8-sig")
    df[["category", "file_name", "image_path", "normal_mode_id"]].to_csv(
        output_dir / "normal_modes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        df.groupby("normal_mode_id")
        .agg(
            n=("file_name", "count"),
            mean_intensity=("mean_intensity", "mean"),
            std_intensity=("std_intensity", "mean"),
            dynamic_p95_p05=("dynamic_p95_p05", "mean"),
            mean_gradient=("mean_gradient", "mean"),
            max_center_distance=("distance_to_mode_center", "max"),
        )
        .reset_index()
        .sort_values("n")
    )
    summary["frequency"] = summary["n"] / len(df)
    summary.to_csv(output_dir / "normal_mode_summary.csv", index=False, encoding="utf-8-sig")

    save_json(
        output_dir / "normal_mode_config.json",
        {
            "category": category,
            "n_train_good": len(df),
            "n_modes": n_modes,
            "seed": seed,
            "feature_columns": feature_columns,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "cluster_centers_standardized": model.cluster_centers_.tolist(),
            "note": "train/goodのみから生成した探索的正常モード。異常画像・GTは使用していない。",
        },
    )

    print("\nNormal mode summary")
    print(summary.to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(description="Analyze normal modes using train/good only")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--n-modes", type=int, default=4)
    p.add_argument("--seed", type=int, default=319)
    p.add_argument("--categories", nargs="*", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    categories = find_categories(args.dataset_root)
    if args.categories:
        categories = {c: categories[c] for c in args.categories}

    for category, path in categories.items():
        analyze_category(
            category,
            path,
            args.output_root / category,
            n_modes=args.n_modes,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
