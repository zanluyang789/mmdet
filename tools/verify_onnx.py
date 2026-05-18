"""
对比 PyTorch 原模型 和 导出的 ONNX，在同一张图上推理，看 mAP / 框数差异。
差异不会像分割那样能算 IoU，这里给一个粗糙的指标：
    - 平均 IoU (按类内最近匹配)
    - 框数差异
"""
import argparse
import numpy as np
import cv2
import onnxruntime as ort
import torch
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector


def _iou_matrix(a, b):
    """a:(N,4) b:(M,4) xyxy -> (N,M)"""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    x11, y11, x12, y12 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    x21, y21, x22, y22 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xx1 = np.maximum(x11, x21); yy1 = np.maximum(y11, y21)
    xx2 = np.minimum(x12, x22); yy2 = np.minimum(y12, y22)
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    a_area = (x12 - x11) * (y12 - y11)
    b_area = (x22 - x21) * (y22 - y21)
    return inter / np.maximum(a_area + b_area - inter, 1e-9)


def torch_infer(cfg_path, ckpt, img_bgr, device='cuda', score_thr=0.3):
    cfg = Config.fromfile(cfg_path)
    if 'norm_cfg' in cfg.model.get('backbone', {}):
        cfg.model.backbone['norm_cfg'] = dict(type='BN', requires_grad=True)
    if 'neck' in cfg.model and isinstance(cfg.model.neck, dict) and 'norm_cfg' in cfg.model.neck:
        cfg.model.neck['norm_cfg'] = dict(type='BN', requires_grad=True)
    model = init_detector(cfg, ckpt, device=f'{device}:0').eval()
    res = inference_detector(model, img_bgr)
    pi = res.pred_instances
    s = pi.scores.cpu().numpy()
    keep = s >= score_thr
    return dict(
        bboxes=pi.bboxes.cpu().numpy()[keep],
        scores=s[keep],
        labels=pi.labels.cpu().numpy()[keep],
    )


def onnx_infer(onnx_path, img_bgr, infer_h, infer_w, score_thr=0.3):
    sess = ort.InferenceSession(onnx_path,
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    h0, w0 = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.transpose(2, 0, 1).astype(np.float32)[None]
    outs = sess.run(out_names, {in_name: x})
    dets, labels = outs[0][0], outs[1][0]
    scores = dets[:, 4]
    keep = scores >= score_thr
    bboxes = dets[keep, :4]
    sx = w0 / float(infer_w)
    sy = h0 / float(infer_h)
    bboxes[:, 0::2] *= sx
    bboxes[:, 1::2] *= sy
    return dict(
        bboxes=bboxes,
        scores=scores[keep],
        labels=labels[keep],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt',   required=True)
    ap.add_argument('--onnx',   required=True)
    ap.add_argument('--img',    required=True)
    ap.add_argument('--img-h',  type=int, default=800)
    ap.add_argument('--img-w',  type=int, default=800)
    ap.add_argument('--score-thr', type=float, default=0.3)
    args = ap.parse_args()

    img = cv2.imread(args.img, cv2.IMREAD_UNCHANGED)
    assert img is not None, f'读不到图 {args.img}'
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    p_t = torch_infer(args.config, args.ckpt, img, score_thr=args.score_thr)
    p_o = onnx_infer(args.onnx, img, args.img_h, args.img_w, score_thr=args.score_thr)
    print(f'torch dets: {len(p_t["bboxes"])}')
    print(f'onnx  dets: {len(p_o["bboxes"])}')

    # 按类内匹配算 IoU
    ious = []
    for c in np.unique(p_t['labels']):
        tb = p_t['bboxes'][p_t['labels'] == c]
        ob = p_o['bboxes'][p_o['labels'] == c]
        if len(tb) == 0 or len(ob) == 0:
            continue
        m = _iou_matrix(tb, ob)
        ious.extend(m.max(axis=1).tolist())

    if ious:
        mean_iou = float(np.mean(ious))
        print(f'mean per-class best-match IoU: {mean_iou:.4f}')
        if mean_iou > 0.85:
            print('[OK] ONNX 和 PyTorch 框基本一致，可以拿去 ATC 转 OM')
        else:
            print('[WARN] 偏差较大，建议用 mmdeploy 重新导出')
    else:
        print('[WARN] 没有可对比的框')


if __name__ == '__main__':
    main()
