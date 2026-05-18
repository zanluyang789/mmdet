"""
ACL 单图推理 + 可视化（检测版）
依赖: pyACL (NPU 镜像里应该自带), opencv-python, numpy
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

# 复用 integration.infer_runner 里的 OMRunner + 画框
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from integration.infer_runner import OMRunner, _draw_dets, _parse_color_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--om',  required=True)
    ap.add_argument('--img', required=True, help='输入图（任意尺寸，自动 resize 到 infer-size）')
    ap.add_argument('--out', default='result_detect.png')
    ap.add_argument('--infer-h', type=int, default=800)
    ap.add_argument('--infer-w', type=int, default=800)
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--num-classes', type=int, default=80)
    ap.add_argument('--color-table', default=None,
                    help='可选；color_table.txt 提供类别名 + 颜色')
    args = ap.parse_args()

    img_bgr = cv2.imread(args.img, cv2.IMREAD_UNCHANGED)
    assert img_bgr is not None, f'读不到 {args.img}'
    if img_bgr.ndim == 2:
        img_bgr = np.stack([img_bgr] * 3, axis=-1)
    if img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]

    runner = OMRunner(args.om, infer_size=(args.infer_h, args.infer_w))
    # 预热
    runner.infer(img_bgr)
    t0 = time.time()
    dets = runner.infer(img_bgr)
    print(f'inference latency: {(time.time()-t0)*1000:.2f} ms')

    keep = dets['scores'] >= args.score_thr
    dets = dict(bboxes=dets['bboxes'][keep],
                scores=dets['scores'][keep],
                labels=dets['labels'][keep])
    print(f'detections after score>={args.score_thr}: {len(dets["bboxes"])}')

    names, palette = _parse_color_table(args.color_table, args.num_classes)
    vis = _draw_dets(img_bgr, dets, names, palette)
    cv2.imwrite(args.out, vis)
    print(f'saved -> {args.out}')

    runner.close()


if __name__ == '__main__':
    main()
