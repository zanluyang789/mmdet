"""
YOLO -> COCO 自动转换
======================

平台下发的数据有时是 YOLO 风格：
    <root>/image/xxx.tif         # 图像
    <root>/labels/xxx.txt        # 标签，每行 "cls cx cy w h"，cx/cy/w/h 归一化到 [0,1]

但 mmdet 检测 dataloader 必须吃 COCO JSON。本模块负责：
    1. 给定 image_dir / label_dir / 类别名 -> 生成 COCO JSON
    2. 自动从 image_dir 推断同级的 label_dir（labels / label 两个 fallback）
    3. 读图像宽高优先用 rasterio（GeoTIFF 友好，只读 header 不解码像素）

被 integration.config_builder._maybe_apply_data_yaml 透明调用，调度方无感知。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


# ============================================================
# 标签目录探测
# ============================================================
def find_labels_dir(image_dir: str) -> Optional[str]:
    """
    在 image_dir 的同级和父级，按优先级查找 label 目录。

    YOLO 习惯目录布局：
        <root>/image/      # 图像
        <root>/labels/     # 标签（也见 "label"，单数）

    也兼容图像和标签同目录（.txt 跟 .tif 放一起）。
    """
    if not image_dir or not os.path.isdir(image_dir):
        return None

    image_dir = os.path.abspath(image_dir)
    parent = os.path.dirname(image_dir)

    # 优先 sibling: labels / label
    for cand_name in ("labels", "label"):
        cand = os.path.join(parent, cand_name)
        if os.path.isdir(cand) and _has_txt(cand):
            return cand

    # 兜底：图像同目录有 .txt
    if _has_txt(image_dir):
        return image_dir

    return None


def _has_txt(d: str) -> bool:
    try:
        for name in os.listdir(d):
            if name.lower().endswith(".txt"):
                return True
    except OSError:
        pass
    return False


# ============================================================
# 图像宽高
# ============================================================
def _read_dims(img_path: str) -> Tuple[int, int]:
    """返回 (width, height)。GeoTIFF 优先走 rasterio header；其它走 PIL；最终 cv2 兜底。"""
    suffix = Path(img_path).suffix.lower()

    if suffix in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(img_path) as src:
                return int(src.width), int(src.height)
        except Exception:
            pass

    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass

    try:
        import cv2
        import numpy as np
        arr = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    except Exception:
        pass

    raise RuntimeError(f"读不到图像尺寸: {img_path}")


# ============================================================
# 缓存校验
# ============================================================
def _cache_valid(output_json: str, label_dir: str, image_dir: str) -> bool:
    """
    若 output_json 已存在，且 label_dir 和 image_dir 的目录 mtime 都比它老，
    认为缓存有效。文件增删会更新目录 mtime，对常见场景够用。
    """
    try:
        out_mtime = os.path.getmtime(output_json)
    except OSError:
        return False
    for d in (label_dir, image_dir):
        try:
            if os.path.getmtime(d) > out_mtime:
                return False
        except OSError:
            return False
    return True


# ============================================================
# 主转换
# ============================================================
def yolo_to_coco(
    image_dir: str,
    label_dir: str,
    names: Sequence[str],
    output_json: str,
    *,
    force: bool = False,
    log_every: int = 500,
) -> str:
    """
    Args:
        image_dir:   YOLO 图像目录（每张图一个文件）
        label_dir:   YOLO 标签目录（每张图对应一个 .txt；图无目标可以没有 .txt）
        names:       类别名列表，下标 == YOLO class_id
        output_json: 输出 COCO JSON 绝对路径
        force:       True 时忽略缓存强制重生成
        log_every:   每处理 N 张图打一行进度

    Returns:
        output_json 路径（缓存命中或新生成）。
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"image_dir 不存在: {image_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"label_dir 不存在: {label_dir}")
    if not names:
        raise ValueError("names 不能为空（至少一个类别）")

    if not force and _cache_valid(output_json, label_dir, image_dir):
        print(f"[yolo2coco] 缓存命中，复用 {output_json}", flush=True)
        return output_json

    print(f"[yolo2coco] 开始转换:", flush=True)
    print(f"             image_dir = {image_dir}", flush=True)
    print(f"             label_dir = {label_dir}", flush=True)
    print(f"             names     = {list(names)}", flush=True)
    print(f"             output    = {output_json}", flush=True)

    image_files = sorted(
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in IMG_EXTS
    )
    if not image_files:
        raise RuntimeError(f"image_dir 下没有图像: {image_dir}")

    categories = [dict(id=i, name=str(n)) for i, n in enumerate(names)]
    images: List[dict] = []
    annotations: List[dict] = []

    image_id = 0
    ann_id = 0
    n_anns_total = 0
    n_missing_label = 0
    n_bad_lines = 0

    t0 = time.time()
    for idx, fname in enumerate(image_files):
        img_path = os.path.join(image_dir, fname)
        stem = Path(fname).stem

        try:
            w, h = _read_dims(img_path)
        except Exception as e:
            print(f"[yolo2coco] 跳过 {fname}: {e}", flush=True)
            continue

        images.append(dict(
            id=image_id,
            file_name=fname,
            width=w,
            height=h,
        ))

        label_path = os.path.join(label_dir, stem + ".txt")
        if not os.path.exists(label_path):
            n_missing_label += 1
        else:
            with open(label_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        n_bad_lines += 1
                        continue
                    try:
                        cls = int(parts[0])
                        cx, cy, bw, bh = (float(p) for p in parts[1:5])
                    except ValueError:
                        n_bad_lines += 1
                        continue
                    if cls < 0 or cls >= len(names):
                        n_bad_lines += 1
                        continue
                    # YOLO normalized (cx, cy, w, h) -> COCO pixel (x, y, w, h)
                    x = max(0.0, (cx - bw / 2.0) * w)
                    y = max(0.0, (cy - bh / 2.0) * h)
                    pw = min(float(w) - x, bw * w)
                    ph = min(float(h) - y, bh * h)
                    if pw <= 1 or ph <= 1:
                        n_bad_lines += 1
                        continue
                    annotations.append(dict(
                        id=ann_id,
                        image_id=image_id,
                        category_id=cls,
                        bbox=[round(x, 2), round(y, 2),
                              round(pw, 2), round(ph, 2)],
                        area=round(pw * ph, 2),
                        iscrowd=0,
                        segmentation=[],
                    ))
                    ann_id += 1
                    n_anns_total += 1

        image_id += 1
        if (idx + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"[yolo2coco] {idx + 1}/{len(image_files)} elapsed={elapsed:.1f}s "
                  f"anns={n_anns_total}", flush=True)

    coco = dict(
        info=dict(description="auto-generated from YOLO labels",
                  date_created=time.strftime("%Y-%m-%d %H:%M:%S")),
        licenses=[],
        images=images,
        annotations=annotations,
        categories=categories,
    )
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"[yolo2coco] 完成: images={len(images)} anns={n_anns_total} "
          f"(missing_label={n_missing_label} bad_lines={n_bad_lines}) "
          f"elapsed={elapsed:.1f}s -> {output_json}", flush=True)
    return output_json


if __name__ == "__main__":
    # 调试入口：python -m integration.yolo2coco <image_dir> <label_dir> <out.json> [names_csv]
    import sys
    if len(sys.argv) < 4:
        print("用法: python -m integration.yolo2coco "
              "<image_dir> <label_dir> <out.json> [name1,name2,...]")
        sys.exit(2)
    image_dir = sys.argv[1]
    label_dir = sys.argv[2]
    out = sys.argv[3]
    if len(sys.argv) > 4:
        names = sys.argv[4].split(",")
    else:
        names = ["object"]
    yolo_to_coco(image_dir, label_dir, names, out, force=True)
