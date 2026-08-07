# -*- coding: utf-8 -*-
"""
02_build_canonical_mvtec_dataset.py
===================================

ROI校正profileをMVTec AD形式データセットへ適用し、
幾何形状を維持したcanonical datasetを生成する。

処理
----
1. category/train/good, category/test/*, category/ground_truth/*を探索
2. roi_profile.jsonから評価領域を復元
3. 円ROIの場合は必要ならprofile近傍だけ局所補正
4. ROIを包含するmargin付き正方形をcrop
5. 正方形→正方形で等方resize
6. 画像・GT・ROI maskへ同一幾何変換を適用
7. MVTec AD形式を維持して保存
8. preprocessing_manifest.csvへ変換情報を記録

重要
----
元画像1600x1200を直接256x256へ押し潰さない。
ROIを正方形cropした後にresizeするため、円は円のまま保持される。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from khi_roi_common import (
    Circle,
    build_mask_from_profile,
    canonicalize_image_and_mask,
    imread_unicode,
    imwrite_unicode,
    list_images,
    load_json,
    save_json,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_mvtec_category(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "train" / "good").is_dir()
        and (path / "test").is_dir()
    )


def discover_categories(dataset_root: Path) -> Dict[str, Path]:
    dataset_root = Path(dataset_root)
    if is_mvtec_category(dataset_root):
        return {dataset_root.name: dataset_root}

    categories = {
        p.name: p
        for p in sorted(dataset_root.iterdir())
        if is_mvtec_category(p)
    }
    if not categories:
        raise FileNotFoundError(
            f"MVTec AD形式カテゴリが見つからない: {dataset_root}"
        )
    return categories


def find_gt(category_path: Path, defect_type: str, image_path: Path) -> Optional[Path]:
    if defect_type == "good":
        return None
    gt_dir = category_path / "ground_truth" / defect_type
    if not gt_dir.exists():
        return None
    for stem in (f"{image_path.stem}_mask", image_path.stem):
        for ext in IMAGE_EXTENSIONS:
            candidate = gt_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def circle_from_profile(profile: Dict, h: int, w: int) -> Circle:
    p = profile["parameters"]
    return Circle(
        cx=float(p["cx_norm"]) * w,
        cy=float(p["cy_norm"]) * h,
        r=float(p["radius_norm"]) * min(w, h),
    )


def circle_mask(shape_hw: Tuple[int, int], circle: Circle) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(
        mask,
        (int(round(circle.cx)), int(round(circle.cy))),
        max(1, int(round(circle.r))),
        255,
        -1,
    )
    return mask


def circle_edge_score(gray: np.ndarray, circle: Circle, n_samples: int = 720) -> float:
    """
    候補円周上のgradient強度平均。

    保存ROIの近傍だけを探索するための軽量スコアであり、
    画像全体からHoughで円を探す用途ではない。
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    xs = np.rint(circle.cx + circle.r * np.cos(theta)).astype(np.int32)
    ys = np.rint(circle.cy + circle.r * np.sin(theta)).astype(np.int32)

    valid = (
        (xs >= 0) & (xs < gray.shape[1])
        & (ys >= 0) & (ys < gray.shape[0])
    )
    if int(valid.sum()) < n_samples * 0.8:
        return -np.inf
    return float(mag[ys[valid], xs[valid]].mean())


def refine_circle_local(
    image_bgr: np.ndarray,
    initial: Circle,
    center_search_px: int,
    radius_search_px: int,
    center_step: int = 2,
    radius_step: int = 1,
) -> Tuple[Circle, float]:
    """
    ROI prior周囲だけで円中心・半径を微調整する。

    完全自動検出ではなく、校正済みpriorを初期値とする制約付き探索である。
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    best = initial
    best_score = circle_edge_score(gray, initial)

    dx_values = range(-center_search_px, center_search_px + 1, max(1, center_step))
    dy_values = range(-center_search_px, center_search_px + 1, max(1, center_step))
    dr_values = range(-radius_search_px, radius_search_px + 1, max(1, radius_step))

    for dy in dy_values:
        for dx in dx_values:
            for dr in dr_values:
                candidate = Circle(
                    cx=initial.cx + dx,
                    cy=initial.cy + dy,
                    r=max(3.0, initial.r + dr),
                )
                score = circle_edge_score(gray, candidate)
                if score > best_score:
                    best = candidate
                    best_score = score

    return best, best_score


def profile_for_category(profile_root: Path, category: str) -> Tuple[Dict, Path]:
    """
    profile-rootが直接profileフォルダでも、カテゴリ別フォルダ群でも対応する。
    """
    direct = profile_root / "roi_profile.json"
    if direct.exists():
        return load_json(direct), profile_root

    category_dir = profile_root / category
    candidate = category_dir / "roi_profile.json"
    if candidate.exists():
        return load_json(candidate), category_dir

    raise FileNotFoundError(
        f"ROI profileが見つからない。category={category}, root={profile_root}"
    )


def process_one(
    image_path: Path,
    gt_path: Optional[Path],
    profile: Dict,
    profile_dir: Path,
    output_size: int,
    margin_ratio: float,
    refine_circle: bool,
    center_search_px: int,
    radius_search_px: int,
):
    image = imread_unicode(image_path, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]

    refine_info = {
        "refined": False,
        "profile_cx": "",
        "profile_cy": "",
        "profile_r": "",
        "used_cx": "",
        "used_cy": "",
        "used_r": "",
        "edge_score": "",
    }

    if profile["roi_type"] == "circle":
        initial = circle_from_profile(profile, h, w)
        used = initial
        score = ""
        if refine_circle:
            used, score_value = refine_circle_local(
                image,
                initial,
                center_search_px=center_search_px,
                radius_search_px=radius_search_px,
            )
            score = score_value
        roi_mask = circle_mask((h, w), used)
        refine_info.update(
            {
                "refined": bool(refine_circle),
                "profile_cx": initial.cx,
                "profile_cy": initial.cy,
                "profile_r": initial.r,
                "used_cx": used.cx,
                "used_cy": used.cy,
                "used_r": used.r,
                "edge_score": score,
            }
        )
    else:
        roi_mask = build_mask_from_profile(
            profile,
            image_shape_hw=(h, w),
            profile_dir=profile_dir,
        )

    gt = None
    if gt_path is not None:
        gt = imread_unicode(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt.shape != (h, w):
            # 元データでサイズが異なる場合のみ画像座標へ合わせる。
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        gt = np.where(gt > 0, 255, 0).astype(np.uint8)

    image_out, roi_out, gt_out, transform = canonicalize_image_and_mask(
        image_bgr=image,
        roi_mask=roi_mask,
        output_size=output_size,
        margin_ratio=margin_ratio,
        gt_mask=gt,
    )

    info = {
        "source_width": w,
        "source_height": h,
        "crop_x0": transform.x0,
        "crop_y0": transform.y0,
        "crop_side": transform.side,
        "pad_left": transform.pad_left,
        "pad_top": transform.pad_top,
        "pad_right": transform.pad_right,
        "pad_bottom": transform.pad_bottom,
        "roi_pixels_output": int(np.count_nonzero(roi_out)),
    }
    info.update(refine_info)
    return image_out, roi_out, gt_out, info


def process_category(
    category: str,
    category_path: Path,
    output_category: Path,
    profile: Dict,
    profile_dir: Path,
    output_size: int,
    margin_ratio: float,
    refine_circle: bool,
    center_search_px: int,
    radius_search_px: int,
) -> List[Dict]:
    rows: List[Dict] = []

    # train/good
    train_good = list_images(category_path / "train" / "good")
    jobs = [("train", "good", p, None) for p in train_good]

    # test/* と対応GT
    test_root = category_path / "test"
    for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        defect_type = defect_dir.name
        for image_path in list_images(defect_dir):
            gt_path = find_gt(category_path, defect_type, image_path)
            if defect_type != "good" and gt_path is None:
                raise FileNotFoundError(
                    f"GTが見つからない: category={category}, defect={defect_type}, image={image_path.name}"
                )
            jobs.append(("test", defect_type, image_path, gt_path))

    start = time.perf_counter()
    for index, (split, defect_type, image_path, gt_path) in enumerate(jobs, start=1):
        t0 = time.perf_counter()
        image_out, roi_out, gt_out, info = process_one(
            image_path=image_path,
            gt_path=gt_path,
            profile=profile,
            profile_dir=profile_dir,
            output_size=output_size,
            margin_ratio=margin_ratio,
            refine_circle=refine_circle,
            center_search_px=center_search_px,
            radius_search_px=radius_search_px,
        )

        image_dst = output_category / split / defect_type / image_path.name
        imwrite_unicode(image_dst, image_out)

        roi_dst = output_category / "roi_masks" / split / defect_type / f"{image_path.stem}_roi.png"
        imwrite_unicode(roi_dst, roi_out)

        gt_dst = ""
        if gt_out is not None:
            gt_dst_path = output_category / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
            imwrite_unicode(gt_dst_path, gt_out)
            gt_dst = str(gt_dst_path)

        elapsed = time.perf_counter() - t0
        row = {
            "category": category,
            "split": split,
            "defect_type": defect_type,
            "source_image": str(image_path),
            "output_image": str(image_dst),
            "output_roi_mask": str(roi_dst),
            "output_gt": gt_dst,
            "output_size": output_size,
            "margin_ratio": margin_ratio,
            "preprocess_sec": elapsed,
        }
        row.update(info)
        rows.append(row)

        if index % 100 == 0 or index == len(jobs):
            print(f"[{category}] {index}/{len(jobs)}")

    print(
        f"[{category}] 完了: {len(jobs)}枚, total={time.perf_counter()-start:.1f}s"
    )
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Build canonical MVTec-format dataset")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--profile-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--output-size", type=int, default=512)
    p.add_argument("--margin-ratio", type=float, default=0.10)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--refine-circle", action="store_true")
    p.add_argument("--center-search-px", type=int, default=8)
    p.add_argument("--radius-search-px", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    categories = discover_categories(args.dataset_root)

    if args.categories:
        missing = [c for c in args.categories if c not in categories]
        if missing:
            raise KeyError(f"指定カテゴリが見つからない: {missing}")
        categories = {c: categories[c] for c in args.categories}

    args.output_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    for category, category_path in categories.items():
        profile, profile_dir = profile_for_category(args.profile_root, category)
        output_category = args.output_root / category
        rows = process_category(
            category=category,
            category_path=category_path,
            output_category=output_category,
            profile=profile,
            profile_dir=profile_dir,
            output_size=args.output_size,
            margin_ratio=args.margin_ratio,
            refine_circle=args.refine_circle,
            center_search_px=args.center_search_px,
            radius_search_px=args.radius_search_px,
        )
        write_csv(output_category / "preprocessing_manifest.csv", rows)
        all_rows.extend(rows)

    write_csv(args.output_root / "preprocessing_manifest_all.csv", all_rows)
    save_json(
        args.output_root / "preprocessing_config.json",
        {
            "dataset_root": str(args.dataset_root),
            "profile_root": str(args.profile_root),
            "output_size": args.output_size,
            "margin_ratio": args.margin_ratio,
            "refine_circle": bool(args.refine_circle),
            "center_search_px": args.center_search_px,
            "radius_search_px": args.radius_search_px,
            "categories": list(categories.keys()),
        },
    )

    times = [float(r["preprocess_sec"]) for r in all_rows]
    if times:
        print("----------- Preprocessing Time -----------")
        print("images      :", len(times))
        print("mean [sec]  :", round(float(np.mean(times)), 5))
        print("p95  [sec]  :", round(float(np.percentile(times, 95)), 5))
        print("total [sec] :", round(float(np.sum(times)), 3))


if __name__ == "__main__":
    main()
