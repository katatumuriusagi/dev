# -*- coding: utf-8 -*-
"""
KHI / MVTec AD向け ROI校正・canonical化 共通処理
==================================================

このファイルは次の処理を共通化する。

- 日本語Windowsパスでも扱いやすい画像入出力
- 円・楕円・多角形ROIのマスク生成
- 円周クリック点からのrobust circle fitting
- Smart Brush用GrabCut初期化
- ROI profileの読込み
- ROI maskから正方形cropを作る幾何処理
- 縦横比を変えない等方resize

重要:
元画像を直接 Resize((N, N)) して縦横比を変える処理は行わない。
必ずROIを含む正方形cropを作ってから正方形へresizeする。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path, recursive: bool = False) -> List[Path]:
    """対応拡張子の画像をファイル名順で返す。"""
    folder = Path(folder)
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """日本語を含むWindowsパスでも読みやすいOpenCV画像読込み。"""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(f"画像を読み込めない: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    """日本語を含むWindowsパスでも保存しやすいOpenCV画像保存。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"画像エンコードに失敗: {path}")
    encoded.tofile(str(path))


def save_json(path: Path, data: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float


@dataclass(frozen=True)
class Ellipse:
    cx: float
    cy: float
    major: float
    minor: float
    angle_deg: float


@dataclass(frozen=True)
class SquareTransform:
    """元画像から正方形cropを作るための情報。"""

    # 元画像座標でのcrop左上と一辺。
    # x0/y0は負値になり得る。その場合はpaddingする。
    x0: int
    y0: int
    side: int

    # 元画像サイズ
    src_width: int
    src_height: int

    # crop時に必要なpadding量
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


# =============================================================================
# 円フィッティング
# =============================================================================


def _circle_from_three_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[Circle]:
    """3点を通る円を求める。ほぼ一直線ならNone。"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    d = 2.0 * (
        x1 * (y2 - y3)
        + x2 * (y3 - y1)
        + x3 * (y1 - y2)
    )
    if abs(d) < 1e-8:
        return None

    u1 = x1 * x1 + y1 * y1
    u2 = x2 * x2 + y2 * y2
    u3 = x3 * x3 + y3 * y3

    cx = (
        u1 * (y2 - y3)
        + u2 * (y3 - y1)
        + u3 * (y1 - y2)
    ) / d

    cy = (
        u1 * (x3 - x2)
        + u2 * (x1 - x3)
        + u3 * (x2 - x1)
    ) / d

    r = math.hypot(x1 - cx, y1 - cy)
    if not np.isfinite([cx, cy, r]).all() or r <= 0:
        return None
    return Circle(float(cx), float(cy), float(r))


def fit_circle_least_squares(points: Sequence[Sequence[float]]) -> Circle:
    """
    円周点へ代数的最小二乗で円をfitする。

    (x-cx)^2 + (y-cy)^2 = r^2
    を整理し、線形最小二乗でcx, cyを求める。
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError("円fitには3点以上の2次元点が必要。")

    x = pts[:, 0]
    y = pts[:, 1]

    a = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y

    params, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy, c = params
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        raise ValueError("円半径が不正。クリック点を確認すること。")
    return Circle(float(cx), float(cy), float(math.sqrt(r2)))


def fit_circle_ransac(
    points: Sequence[Sequence[float]],
    residual_threshold_px: float = 3.0,
    iterations: int = 500,
    seed: int = 319,
) -> Tuple[Circle, np.ndarray]:
    """
    クリック誤差や外れ点に強いRANSAC円fit。

    戻り値:
      circle: 最終fit円
      inliers: 入力点ごとのinlier真偽値
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError("RANSAC円fitには3点以上必要。")

    # 3点しかない場合はそのまま最小二乗。
    if len(pts) == 3:
        circle = fit_circle_least_squares(pts)
        return circle, np.ones(3, dtype=bool)

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(len(pts), dtype=bool)
    best_error = np.inf

    for _ in range(iterations):
        ids = rng.choice(len(pts), size=3, replace=False)
        circle = _circle_from_three_points(pts[ids[0]], pts[ids[1]], pts[ids[2]])
        if circle is None:
            continue

        radius = np.sqrt((pts[:, 0] - circle.cx) ** 2 + (pts[:, 1] - circle.cy) ** 2)
        residual = np.abs(radius - circle.r)
        inliers = residual <= residual_threshold_px

        count = int(inliers.sum())
        if count < 3:
            continue

        mean_error = float(residual[inliers].mean())
        if count > int(best_inliers.sum()) or (
            count == int(best_inliers.sum()) and mean_error < best_error
        ):
            best_inliers = inliers
            best_error = mean_error

    if int(best_inliers.sum()) < 3:
        best_inliers[:] = True

    final_circle = fit_circle_least_squares(pts[best_inliers])
    return final_circle, best_inliers


# =============================================================================
# ROI mask生成
# =============================================================================


def circle_mask(shape_hw: Tuple[int, int], circle: Circle) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(
        mask,
        (int(round(circle.cx)), int(round(circle.cy))),
        max(1, int(round(circle.r))),
        255,
        thickness=-1,
        lineType=cv2.LINE_8,
    )
    return mask


def ellipse_mask(shape_hw: Tuple[int, int], ellipse: Ellipse) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    axes = (
        max(1, int(round(ellipse.major / 2.0))),
        max(1, int(round(ellipse.minor / 2.0))),
    )
    cv2.ellipse(
        mask,
        (int(round(ellipse.cx)), int(round(ellipse.cy))),
        axes,
        float(ellipse.angle_deg),
        0,
        360,
        255,
        thickness=-1,
        lineType=cv2.LINE_8,
    )
    return mask


def polygon_mask(shape_hw: Tuple[int, int], points: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError("多角形ROIには3頂点以上必要。")
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    return mask


def fit_ellipse(points: Sequence[Sequence[float]]) -> Ellipse:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 5 or pts.shape[1] != 2:
        raise ValueError("楕円fitには5点以上必要。")

    (cx, cy), (width, height), angle = cv2.fitEllipse(pts.reshape(-1, 1, 2))
    # major >= minorへ統一する。
    if width >= height:
        major, minor = width, height
        angle_deg = angle
    else:
        major, minor = height, width
        angle_deg = angle + 90.0

    angle_deg = float(angle_deg % 180.0)
    return Ellipse(float(cx), float(cy), float(major), float(minor), angle_deg)


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """非0領域のx,y,w,h。空maskならエラー。"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("ROI maskが空。")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


# =============================================================================
# GrabCut Smart Brush
# =============================================================================


def grabcut_from_lasso(
    image_bgr: np.ndarray,
    lasso_points: Sequence[Sequence[int]],
    iterations: int = 5,
    erode_ratio: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    人間が大まかになぞった閉曲線からGrabCutを初期化する。

    返り値:
      result_mask: 0/255のROI mask
      gc_mask: GrabCut内部ラベル。後からbrush修正に使える。
    """
    h, w = image_bgr.shape[:2]
    rough = polygon_mask((h, w), lasso_points)

    # 初期状態は外側をsure background、内側をprobable foregroundとする。
    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[rough > 0] = cv2.GC_PR_FGD

    # 内側を少しerodeしてsure foreground seedを作る。
    x, y, bw, bh = mask_bbox(rough)
    radius = max(1, int(round(erode_ratio * max(bw, bh))))
    ksize = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    inner = cv2.erode(rough, kernel)
    if np.count_nonzero(inner) == 0:
        # 小さいROIの場合は中心付近の円をsure foregroundにする。
        center = (x + bw // 2, y + bh // 2)
        cv2.circle(inner, center, max(1, min(bw, bh) // 8), 255, -1)
    gc_mask[inner > 0] = cv2.GC_FGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        image_bgr,
        gc_mask,
        None,
        bg_model,
        fg_model,
        iterations,
        cv2.GC_INIT_WITH_MASK,
    )

    result = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    return result, gc_mask


def rerun_grabcut(
    image_bgr: np.ndarray,
    gc_mask: np.ndarray,
    iterations: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """brush修正済みGrabCut maskを再計算する。"""
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        image_bgr,
        gc_mask,
        None,
        bg_model,
        fg_model,
        iterations,
        cv2.GC_INIT_WITH_MASK,
    )
    result = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    return result, gc_mask


# =============================================================================
# ROI profileからmaskを復元
# =============================================================================


def _denorm(value: float, scale: int) -> float:
    return float(value) * float(scale)


def build_mask_from_profile(
    profile: Dict,
    image_shape_hw: Tuple[int, int],
    profile_dir: Optional[Path] = None,
) -> np.ndarray:
    """ROI profileを任意解像度のmaskへ変換する。"""
    h, w = image_shape_hw
    roi_type = profile["roi_type"]

    if roi_type == "circle":
        p = profile["parameters"]
        circle = Circle(
            cx=_denorm(p["cx_norm"], w),
            cy=_denorm(p["cy_norm"], h),
            r=_denorm(p["radius_norm"], min(w, h)),
        )
        return circle_mask((h, w), circle)

    if roi_type == "ellipse":
        p = profile["parameters"]
        ellipse = Ellipse(
            cx=_denorm(p["cx_norm"], w),
            cy=_denorm(p["cy_norm"], h),
            major=_denorm(p["major_norm"], min(w, h)),
            minor=_denorm(p["minor_norm"], min(w, h)),
            angle_deg=float(p["angle_deg"]),
        )
        return ellipse_mask((h, w), ellipse)

    if roi_type == "polygon":
        p = profile["parameters"]
        points = [
            [float(xn) * w, float(yn) * h]
            for xn, yn in p["points_norm"]
        ]
        return polygon_mask((h, w), points)

    if roi_type in {"mask", "smart"}:
        if profile_dir is None:
            raise ValueError("mask型profileにはprofile_dirが必要。")
        mask_name = profile["mask_file"]
        mask = imread_unicode(Path(profile_dir) / mask_name, cv2.IMREAD_GRAYSCALE)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.where(mask > 127, 255, 0).astype(np.uint8)

    raise ValueError(f"未対応roi_type: {roi_type}")


# =============================================================================
# canonical crop / resize
# =============================================================================


def make_square_transform(
    mask: np.ndarray,
    margin_ratio: float = 0.10,
    min_side: int = 16,
) -> SquareTransform:
    """
    ROIのbounding boxを包含する正方形cropを作る。

    side = max(ROI幅, ROI高さ) * (1 + 2*margin_ratio)
    とし、ROI中心を正方形中心へ置く。
    """
    h, w = mask.shape[:2]
    x, y, bw, bh = mask_bbox(mask)

    cx = x + (bw - 1) / 2.0
    cy = y + (bh - 1) / 2.0
    side = max(min_side, int(math.ceil(max(bw, bh) * (1.0 + 2.0 * margin_ratio))))

    # 奇数/偶数は問わないが、一貫してroundで中心合わせ。
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    return SquareTransform(
        x0=x0,
        y0=y0,
        side=side,
        src_width=w,
        src_height=h,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )


def apply_square_transform(
    image: np.ndarray,
    transform: SquareTransform,
    output_size: int,
    interpolation: int,
    border_mode: int = cv2.BORDER_REFLECT_101,
    border_value: int = 0,
) -> np.ndarray:
    """同じSquareTransformを画像・GT・ROI maskへ適用する。"""
    if image.shape[0] != transform.src_height or image.shape[1] != transform.src_width:
        raise ValueError(
            "SquareTransform作成時と入力画像サイズが異なる。"
            f" expected=({transform.src_height},{transform.src_width})"
            f" actual={image.shape[:2]}"
        )

    padded = cv2.copyMakeBorder(
        image,
        transform.pad_top,
        transform.pad_bottom,
        transform.pad_left,
        transform.pad_right,
        border_mode,
        value=border_value,
    )

    x0 = transform.x0 + transform.pad_left
    y0 = transform.y0 + transform.pad_top
    crop = padded[y0:y0 + transform.side, x0:x0 + transform.side]

    if crop.shape[0] != transform.side or crop.shape[1] != transform.side:
        raise RuntimeError("正方形crop生成に失敗。")

    return cv2.resize(crop, (output_size, output_size), interpolation=interpolation)


def canonicalize_image_and_mask(
    image_bgr: np.ndarray,
    roi_mask: np.ndarray,
    output_size: int,
    margin_ratio: float,
    gt_mask: Optional[np.ndarray] = None,
    border_mode: int = cv2.BORDER_REFLECT_101,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], SquareTransform]:
    """画像、ROI mask、任意のGTを同じ幾何変換でcanonical化する。"""
    transform = make_square_transform(roi_mask, margin_ratio=margin_ratio)

    image_out = apply_square_transform(
        image_bgr,
        transform,
        output_size=output_size,
        interpolation=cv2.INTER_AREA,
        border_mode=border_mode,
    )

    roi_out = apply_square_transform(
        roi_mask,
        transform,
        output_size=output_size,
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0,
    )
    roi_out = np.where(roi_out > 127, 255, 0).astype(np.uint8)

    gt_out = None
    if gt_mask is not None:
        gt_out = apply_square_transform(
            gt_mask,
            transform,
            output_size=output_size,
            interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=0,
        )
        gt_out = np.where(gt_out > 127, 255, 0).astype(np.uint8)

    return image_out, roi_out, gt_out, transform


def resize_square_preserve_geometry(image: np.ndarray, output_size: int, is_mask: bool = False) -> np.ndarray:
    """
    すでに正方形のcanonical imageだけを別解像度へ変換する。

    正方形→正方形なのでx/y倍率が同じであり、円が楕円にならない。
    """
    if image.shape[0] != image.shape[1]:
        raise ValueError(
            "この関数はcanonical化済み正方形画像専用。"
            "非正方形画像を直接押し潰してはいけない。"
        )
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(image, (output_size, output_size), interpolation=interpolation)


def normalized_mad(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))
