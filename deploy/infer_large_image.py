"""
检测大图推理（滑窗 + 跨瓦片框合并）
====================================

应用场景：超大遥感影像（>10000 像素），整图推理 OOM。
做法：
    1) 用滑窗切瓦片，每个瓦片单独推理
    2) 瓦片框坐标 +offset 还原到全图
    3) 全图做一次 NMS 合并跨瓦片重复框
    4) 输出 GeoTIFF 同坐标系下的 SHP（每个框一条 polygon）

环境变量 / 命令行：
    PATH_MODEL_RESOURCE / --model    模型目录或路径（.pth/.onnx/.om）
    DATA_INPUT_DIR1 / --input        输入 GeoTIFF
    DATA_OUTPUT_DIR / --output-dir   输出目录
    BAND_ORDER / --band-order        多波段映射，默认 "3,2,1"
    TILE  / --tile                   瓦片边长，默认 800
    STRIDE / --stride                滑窗步长，默认 600（重叠 200）
    SCORE_THR / --score-thr          置信度阈值，默认 0.3
    IOU_THR / --iou-thr              全图 NMS IoU 阈值，默认 0.5
    CFG / --cfg                      pth 推理时的 mmdet config
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from send_kafka_msg import send as kafka_send  # noqa: E402


def _import_runner():
    """延迟导入，避免 PC 没装 pyACL 时无法 import"""
    from integration.infer_runner import make_runner_any  # noqa: E402
    return make_runner_any


# ============================================================
# 影像读取 / 波段处理（与 mmseg deploy 那一份保持一致）
# ============================================================
def pick_rgb(arr_chw, band_order_str):
    """从多波段影像里取 3 个波段拼 RGB
    arr_chw: (C, H, W) uint8/uint16, band_order_str: '3,2,1'
    """
    idx = [int(x) - 1 for x in band_order_str.split(',')]
    assert len(idx) == 3, f'BAND_ORDER 必须 3 个: {band_order_str}'
    rgb = arr_chw[idx]
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        out = np.zeros_like(rgb, dtype=np.uint8)
        for c in range(3):
            ch = rgb[c]
            lo, hi = np.percentile(ch[ch > 0], (2, 98)) if (ch > 0).any() else (0, 255)
            if hi <= lo:
                hi = lo + 1
            ch = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255)
            out[c] = ch.astype(np.uint8)
        rgb = out
    return rgb  # (3, H, W) uint8


def gen_starts(L, t, s):
    if L <= t:
        return [0]
    starts = list(range(0, L - t + 1, s))
    if starts[-1] + t < L:
        starts.append(L - t)
    return starts


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """简单 numpy NMS。boxes: (N,4) xyxy。返回 keep indices。"""
    if boxes.size == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        keep_idx = np.where(iou <= iou_thr)[0]
        order = order[keep_idx + 1]
    return np.array(keep, dtype=np.int64)


def slide_detect(rgb_chw_u8, runner, tile=800, stride=600,
                  score_thr=0.3, iou_thr=0.5, progress_cb=None):
    """
    rgb_chw_u8: (3, H, W) uint8
    返回: dict(bboxes, scores, labels) —— 全图坐标，已 NMS
    """
    import cv2
    _, H, W = rgb_chw_u8.shape
    ys = gen_starts(H, tile, stride)
    xs = gen_starts(W, tile, stride)
    total = len(ys) * len(xs)
    print(f'[slide] image {H}x{W}, tiles {len(ys)}x{len(xs)} = {total}', flush=True)

    all_boxes, all_scores, all_labels = [], [], []
    t0 = time.time()
    done = 0

    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            tile_rgb = rgb_chw_u8[:, y:y+tile, x:x+tile]
            ph = tile - tile_rgb.shape[1]
            pw = tile - tile_rgb.shape[2]
            if ph > 0 or pw > 0:
                tile_rgb = np.pad(tile_rgb, ((0, 0), (0, ph), (0, pw)), mode='reflect')
            # (H,W,3) BGR 喂给 runner
            tile_bgr = cv2.cvtColor(tile_rgb.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            dets = runner.infer(tile_bgr)
            b = dets['bboxes']
            s = dets['scores']
            l = dets['labels']
            keep = s >= score_thr
            if keep.any():
                bb = b[keep].copy()
                bb[:, 0] += x; bb[:, 2] += x
                bb[:, 1] += y; bb[:, 3] += y
                # 截到原图范围内
                bb[:, 0::2] = np.clip(bb[:, 0::2], 0, W)
                bb[:, 1::2] = np.clip(bb[:, 1::2], 0, H)
                all_boxes.append(bb)
                all_scores.append(s[keep])
                all_labels.append(l[keep])
            done += 1
            if done % 20 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f'  [{done}/{total}] elapsed={elapsed:.1f}s eta={eta:.1f}s '
                      f'cur_dets={sum(b.shape[0] for b in all_boxes)}', flush=True)
                if progress_cb:
                    progress_cb(done, total)

    if not all_boxes:
        return dict(
            bboxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
        )

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 按类做 NMS
    keep_all = []
    for c in np.unique(labels):
        m = labels == c
        idx = np.where(m)[0]
        keep = nms_numpy(boxes[idx], scores[idx], iou_thr)
        keep_all.append(idx[keep])
    keep_idx = np.concatenate(keep_all) if keep_all else np.array([], dtype=np.int64)

    print(f'[slide] before nms: {len(boxes)}  after: {len(keep_idx)}', flush=True)
    return dict(
        bboxes=boxes[keep_idx],
        scores=scores[keep_idx],
        labels=labels[keep_idx],
    )


# ============================================================
# 主流程
# ============================================================
def process_one(input_tif, output_dir, model_path, band_order, tile, stride,
                score_thr, iou_thr, base_config,
                progress_range=(0, 100)):
    import rasterio
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_tif).stem
    out_json = output_dir / f'{stem}_dets.json'
    out_shp = output_dir / f'{stem}_dets.shp'

    p_lo, p_hi = progress_range

    with rasterio.open(input_tif) as src:
        print(f'[input] {input_tif}', flush=True)
        print(f'         crs={src.crs}  size={src.width}x{src.height}'
              f'  bands={src.count}  dtype={src.dtypes[0]}', flush=True)
        arr = src.read()
        transform = src.transform
        crs = src.crs

    rgb = pick_rgb(arr, band_order)
    del arr

    make_runner_any = _import_runner()
    runner = make_runner_any(model_path, device=os.environ.get('INTEGRATION_DEVICE', 'cuda'),
                              base_config=base_config, score_thr=score_thr)

    def cb(done, total):
        prog = p_lo + (p_hi - p_lo) * done / max(total, 1)
        kafka_send(int(prog), 'running', f'推理中 {done}/{total} 瓦片')

    try:
        dets = slide_detect(rgb, runner, tile=tile, stride=stride,
                             score_thr=score_thr, iou_thr=iou_thr, progress_cb=cb)
    finally:
        runner.close()

    # JSON 输出
    js = []
    for b, s, lab in zip(dets['bboxes'], dets['scores'], dets['labels']):
        x1, y1, x2, y2 = [float(v) for v in b]
        js.append(dict(
            image_path=str(input_tif),
            category_id=int(lab),
            bbox=[x1, y1, x2 - x1, y2 - y1],
            score=float(s),
        ))
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(js, f, ensure_ascii=False, indent=2)
    print(f'[ok] json -> {out_json}  ({len(js)} dets)', flush=True)

    # SHP
    try:
        from integration.infer_runner import _save_shp
        # 不带 class name 时用 cls_idx 占位
        names = [f'cls_{i}' for i in range(max(int(dets['labels'].max()) + 1
                                                if dets['labels'].size else 1, 1))]
        _save_shp(dets, input_tif, str(out_shp), names)
        print(f'[ok] shp -> {out_shp}', flush=True)
    except Exception as e:
        print(f'[warn] shp 写出失败: {e}', flush=True)

    return str(out_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', help='输入 GeoTIFF(覆盖 DATA_INPUT_DIR1)')
    ap.add_argument('--output-dir', help='输出目录(覆盖 DATA_OUTPUT_DIR)')
    ap.add_argument('--model', help='模型路径(.pth/.onnx/.om)，覆盖 PATH_MODEL_RESOURCE')
    ap.add_argument('--band-order', default=None)
    ap.add_argument('--tile', type=int, default=None)
    ap.add_argument('--stride', type=int, default=None)
    ap.add_argument('--score-thr', type=float, default=None)
    ap.add_argument('--iou-thr', type=float, default=None)
    ap.add_argument('--cfg', default=None, help='pth 推理时的 mmdet config')
    ap.add_argument('--progress-lo', type=int, default=0)
    ap.add_argument('--progress-hi', type=int, default=100)
    args = ap.parse_args()

    input_tif = args.input or os.environ.get('DATA_INPUT_DIR1')
    output_dir = args.output_dir or os.environ.get('DATA_OUTPUT_DIR')
    model_path = args.model
    if not model_path:
        mdir = os.environ.get('PATH_MODEL_RESOURCE', '/app/module')
        for cand in ('detect.om', 'detect.onnx', 'detect.pth'):
            p = os.path.join(mdir, cand)
            if os.path.exists(p):
                model_path = p
                break
    band_order = args.band_order or os.environ.get('BAND_ORDER', '3,2,1')
    tile = args.tile or int(os.environ.get('TILE', '800'))
    stride = args.stride or int(os.environ.get('STRIDE', '600'))
    score_thr = args.score_thr if args.score_thr is not None \
        else float(os.environ.get('SCORE_THR', '0.3'))
    iou_thr = args.iou_thr if args.iou_thr is not None \
        else float(os.environ.get('IOU_THR', '0.5'))
    base_config = args.cfg or os.environ.get('CFG',
                                             'configs/faster_rcnn_r50_fpn_object.py')

    print('======== detect / infer_large_image ========', flush=True)
    print(f'input       = {input_tif}', flush=True)
    print(f'output_dir  = {output_dir}', flush=True)
    print(f'model       = {model_path}', flush=True)
    print(f'band_order  = {band_order}', flush=True)
    print(f'tile/stride = {tile}/{stride}  score_thr={score_thr}  iou={iou_thr}',
          flush=True)
    print('===========================================', flush=True)

    if not input_tif or not os.path.isfile(input_tif):
        print(f'[fatal] input 不存在: {input_tif}', file=sys.stderr); sys.exit(2)
    if not output_dir:
        print('[fatal] output_dir 未设置', file=sys.stderr); sys.exit(2)
    if not model_path or not os.path.isfile(model_path):
        print(f'[fatal] 模型不存在: {model_path}', file=sys.stderr); sys.exit(2)

    process_one(input_tif, output_dir, model_path, band_order, tile, stride,
                score_thr, iou_thr, base_config,
                progress_range=(args.progress_lo, args.progress_hi))


if __name__ == '__main__':
    main()
