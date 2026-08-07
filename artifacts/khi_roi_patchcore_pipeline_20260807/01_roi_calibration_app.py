# -*- coding: utf-8 -*-
"""
01_roi_calibration_app.py
=========================

新しい撮影環境・データセットに対して、少数の代表画像からROI Priorを校正する。

対応モード
----------
1. circle
   円周上を5点以上クリックし、RANSAC + 最小二乗で中心・半径を推定する。

2. polygon
   評価領域の頂点を順にクリックして多角形ROIを指定する。

3. smart
   評価領域の外周をマウスドラッグで大まかになぞり、GrabCutで境界を補正する。

基本操作
--------
左クリック            : 点追加（circle / polygon）
左ドラッグ            : 粗い輪郭を描く（smart）
右クリック            : 最後の点を削除
Enter                  : 現在画像のROIを確定
N                      : 次画像へ（未確定の場合は警告）
B                      : 前画像へ
R                      : 現在画像の入力をリセット
S                      : 現在までの結果からROI profileを保存
Q / Esc                : 終了

実行例
------
python .\01_roi_calibration_app.py `
  --image-dir "C:\data\khi_chip_6\train\good" `
  --mode circle `
  --max-images 8 `
  --output-dir ".\roi_profiles\khi_chip_6"

研究上の考え方
--------------
- ROIを毎画像ゼロから検出するのではなく、撮影環境固有の空間priorを少数画像で校正する。
- circleはKHI向けの具体実装であり、一般概念はROI Prior Calibrationである。
- polygon / smartにより非円形製品にも同じ枠組みを適用する。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from khi_roi_common import (
    Circle,
    fit_circle_ransac,
    grabcut_from_lasso,
    imread_unicode,
    imwrite_unicode,
    list_images,
    normalized_mad,
    polygon_mask,
    save_json,
)


WINDOW_NAME = "ROI Calibration"


class CalibrationApp:
    def __init__(
        self,
        image_paths: Sequence[Path],
        mode: str,
        output_dir: Path,
        circle_residual_px: float,
        probability_threshold: float,
    ):
        self.image_paths = list(image_paths)
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.circle_residual_px = float(circle_residual_px)
        self.probability_threshold = float(probability_threshold)

        self.index = 0
        self.points: List[Tuple[int, int]] = []
        self.smart_stroke: List[Tuple[int, int]] = []
        self.dragging = False

        self.current_result: Optional[Dict] = None
        self.current_mask: Optional[np.ndarray] = None
        self.results: Dict[str, Dict] = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "per_image_masks").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "previews").mkdir(parents=True, exist_ok=True)

    @property
    def current_path(self) -> Path:
        return self.image_paths[self.index]

    def reset_current(self) -> None:
        self.points = []
        self.smart_stroke = []
        self.dragging = False
        self.current_result = None
        self.current_mask = None

    def mouse_callback(self, event, x, y, flags, param):
        if self.mode in {"circle", "polygon"}:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                self.current_result = None
                self.current_mask = None
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.points:
                    self.points.pop()
                self.current_result = None
                self.current_mask = None
            return

        if self.mode == "smart":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.dragging = True
                self.smart_stroke = [(x, y)]
                self.current_result = None
                self.current_mask = None
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                # 点が過密になりすぎないよう距離3px以上だけ追加する。
                if not self.smart_stroke:
                    self.smart_stroke.append((x, y))
                else:
                    px, py = self.smart_stroke[-1]
                    if (x - px) ** 2 + (y - py) ** 2 >= 9:
                        self.smart_stroke.append((x, y))
            elif event == cv2.EVENT_LBUTTONUP and self.dragging:
                self.dragging = False
                if self.smart_stroke:
                    px, py = self.smart_stroke[0]
                    qx, qy = self.smart_stroke[-1]
                    if (px - qx) ** 2 + (py - qy) ** 2 > 25:
                        self.smart_stroke.append((px, py))
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.smart_stroke = []
                self.current_result = None
                self.current_mask = None

    def estimate_current(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]

        if self.mode == "circle":
            if len(self.points) < 3:
                raise ValueError("circleは最低3点、推奨5点以上をクリックすること。")
            circle, inliers = fit_circle_ransac(
                self.points,
                residual_threshold_px=self.circle_residual_px,
            )
            mask = np.zeros((h, w), np.uint8)
            cv2.circle(
                mask,
                (int(round(circle.cx)), int(round(circle.cy))),
                int(round(circle.r)),
                255,
                -1,
            )
            self.current_result = {
                "roi_type": "circle",
                "cx": circle.cx,
                "cy": circle.cy,
                "r": circle.r,
                "points": [[int(x), int(y)] for x, y in self.points],
                "inliers": [bool(v) for v in inliers.tolist()],
            }
            self.current_mask = mask
            return

        if self.mode == "polygon":
            if len(self.points) < 3:
                raise ValueError("polygonは3頂点以上必要。")
            mask = polygon_mask((h, w), self.points)
            self.current_result = {
                "roi_type": "polygon",
                "points": [[int(x), int(y)] for x, y in self.points],
            }
            self.current_mask = mask
            return

        if self.mode == "smart":
            if len(self.smart_stroke) < 3:
                raise ValueError("smartは評価領域の外周を閉じるようにドラッグすること。")
            mask, _ = grabcut_from_lasso(image, self.smart_stroke)
            if np.count_nonzero(mask) == 0:
                raise ValueError("GrabCut結果が空。輪郭をもう少し外側になぞること。")
            self.current_result = {
                "roi_type": "smart",
                "lasso_points": [[int(x), int(y)] for x, y in self.smart_stroke],
            }
            self.current_mask = mask
            return

        raise ValueError(f"未対応mode: {self.mode}")

    def confirm_current(self, image: np.ndarray) -> None:
        if self.current_result is None or self.current_mask is None:
            self.estimate_current(image)

        path = self.current_path
        key = path.name
        h, w = image.shape[:2]

        result = dict(self.current_result)
        result.update(
            {
                "file_name": key,
                "image_width": w,
                "image_height": h,
            }
        )
        self.results[key] = result

        mask_path = self.output_dir / "per_image_masks" / f"{path.stem}_roi.png"
        imwrite_unicode(mask_path, self.current_mask)

        preview = image.copy()
        contour, _ = cv2.findContours(
            self.current_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(preview, contour, -1, (0, 255, 0), 2)
        imwrite_unicode(
            self.output_dir / "previews" / f"{path.stem}_preview.jpg",
            preview,
        )

        save_json(self.output_dir / "calibration_raw.json", self.results)
        print(f"[確定] {path.name}")

    def build_profile(self) -> Dict:
        if not self.results:
            raise ValueError("確定済みROIがない。")

        rows = list(self.results.values())
        widths = [int(r["image_width"]) for r in rows]
        heights = [int(r["image_height"]) for r in rows]

        # 校正画像サイズが混在しても正規化座標で統合する。
        profile: Dict = {
            "profile_version": "roi_prior_v1",
            "roi_type": self.mode,
            "calibration_images": len(rows),
            "source_files": [r["file_name"] for r in rows],
            "probability_threshold": self.probability_threshold,
        }

        if self.mode == "circle":
            cx_norm = [r["cx"] / r["image_width"] for r in rows]
            cy_norm = [r["cy"] / r["image_height"] for r in rows]
            r_norm = [r["r"] / min(r["image_width"], r["image_height"]) for r in rows]

            profile["parameters"] = {
                "cx_norm": float(np.median(cx_norm)),
                "cy_norm": float(np.median(cy_norm)),
                "radius_norm": float(np.median(r_norm)),
            }
            profile["variability"] = {
                "cx_norm_mad": normalized_mad(cx_norm),
                "cy_norm_mad": normalized_mad(cy_norm),
                "radius_norm_mad": normalized_mad(r_norm),
            }
            return profile

        # polygon/smartは各画像maskを共通のreference解像度へ正規化し、
        # ROI存在確率を作って多数決profile maskへ変換する。
        ref_w = int(round(float(np.median(widths))))
        ref_h = int(round(float(np.median(heights))))
        prob = np.zeros((ref_h, ref_w), dtype=np.float32)

        for r in rows:
            src_mask = imread_unicode(
                self.output_dir / "per_image_masks" / f"{Path(r['file_name']).stem}_roi.png",
                cv2.IMREAD_GRAYSCALE,
            )
            resized = cv2.resize(src_mask, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
            prob += (resized > 127).astype(np.float32)

        prob /= float(len(rows))
        profile_mask = np.where(prob >= self.probability_threshold, 255, 0).astype(np.uint8)

        mask_file = "roi_profile_mask.png"
        prob_file = "roi_probability.npy"
        imwrite_unicode(self.output_dir / mask_file, profile_mask)
        np.save(self.output_dir / prob_file, prob)

        profile.update(
            {
                "roi_type": "mask" if self.mode == "smart" else "mask",
                "source_annotation_type": self.mode,
                "reference_width": ref_w,
                "reference_height": ref_h,
                "mask_file": mask_file,
                "probability_file": prob_file,
            }
        )
        return profile

    def save_profile(self) -> None:
        profile = self.build_profile()
        save_json(self.output_dir / "roi_profile.json", profile)
        print("[保存]", self.output_dir / "roi_profile.json")

    def draw(self, image: np.ndarray) -> np.ndarray:
        canvas = image.copy()
        h, w = canvas.shape[:2]

        # 入力点/線
        if self.mode in {"circle", "polygon"}:
            for i, (x, y) in enumerate(self.points):
                cv2.circle(canvas, (x, y), 4, (0, 255, 255), -1)
                cv2.putText(
                    canvas,
                    str(i + 1),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            if self.mode == "polygon" and len(self.points) >= 2:
                pts = np.asarray(self.points, np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], False, (0, 255, 255), 2)

        if self.mode == "smart" and len(self.smart_stroke) >= 2:
            pts = np.asarray(self.smart_stroke, np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, (0, 255, 255), 2)

        # 推定済みROI
        if self.current_mask is not None:
            contours, _ = cv2.findContours(
                self.current_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

        # 状態表示
        confirmed = self.current_path.name in self.results
        lines = [
            f"mode={self.mode}  image={self.index + 1}/{len(self.image_paths)}",
            f"file={self.current_path.name}",
            f"confirmed={'YES' if confirmed else 'NO'}  confirmed_total={len(self.results)}",
            "Enter: estimate/confirm  N/B: next/back  R: reset  S: save profile  Q: quit",
        ]
        y = 25
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            y += 22

        return canvas

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            image = imread_unicode(self.current_path)
            canvas = self.draw(image)
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (13, 10):
                try:
                    self.estimate_current(image)
                    self.confirm_current(image)
                except Exception as exc:
                    print("[確定失敗]", exc)

            elif key in (ord("r"), ord("R")):
                self.reset_current()

            elif key in (ord("n"), ord("N")):
                if self.current_path.name not in self.results:
                    print("[警告] 現在画像は未確定。Enterで確定するか、不要ならそのまま移動する。")
                if self.index < len(self.image_paths) - 1:
                    self.index += 1
                    self.reset_current()

            elif key in (ord("b"), ord("B")):
                if self.index > 0:
                    self.index -= 1
                    self.reset_current()

            elif key in (ord("s"), ord("S")):
                try:
                    self.save_profile()
                except Exception as exc:
                    print("[保存失敗]", exc)

        cv2.destroyAllWindows()

        if self.results:
            try:
                self.save_profile()
            except Exception as exc:
                print("[終了時profile保存失敗]", exc)


def parse_args():
    p = argparse.ArgumentParser(description="Dataset-specific ROI Prior Calibration")
    p.add_argument("--image-dir", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=["circle", "polygon", "smart"])
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-images", type=int, default=8)
    p.add_argument("--circle-residual-px", type=float, default=3.0)
    p.add_argument("--probability-threshold", type=float, default=0.6)
    return p.parse_args()


def main():
    args = parse_args()
    images = list_images(args.image_dir)
    if not images:
        raise FileNotFoundError(f"画像が見つからない: {args.image_dir}")

    if args.max_images > 0:
        # 時系列偏りを減らすため、フォルダ全体から等間隔に選ぶ。
        if len(images) > args.max_images:
            ids = np.linspace(0, len(images) - 1, args.max_images).round().astype(int)
            images = [images[int(i)] for i in ids]

    print("ROI校正画像:")
    for path in images:
        print(" ", path.name)

    app = CalibrationApp(
        image_paths=images,
        mode=args.mode,
        output_dir=args.output_dir,
        circle_residual_px=args.circle_residual_px,
        probability_threshold=args.probability_threshold,
    )
    app.run()


if __name__ == "__main__":
    main()
