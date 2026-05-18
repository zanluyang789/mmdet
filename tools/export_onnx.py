"""
mmdet 模型导出 ONNX（端到端）
================================
- 输入: float32, NCHW, [0,255] RGB，固定 shape（默认 1x3x800x800）
- 内部做归一化(ImageNet mean/std)
- 输出:
    'dets':   (1, max_dets, 5) -> [x1, y1, x2, y2, score]
    'labels': (1, max_dets)    -> int64
  无效项 score==0 截断，OMRunner / ONNXRunner 会自动 drop。

注意：
- 两阶段检测（如 Faster R-CNN）的端到端导出在 mmdet 3.x 里推荐用 mmdeploy。
  本脚本走简易路径：用 torch.no_grad() + 模型 self.predict() 拿到 InstanceData，
  再 wrap 成定长输出。比 mmdeploy 简单，缺点是 NMS 在 PyTorch 里执行，
  ATC 转 OM 时会被视作 NPU 不友好算子；对 RetinaNet/YOLOX 等单阶段更友好。
- 想用 mmdeploy 的话：见 README 里的"ATC / OM 完整路径"段落。
"""
import argparse

import torch
import torch.nn as nn

IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD  = [58.395, 57.12, 57.375]


class DetONNX(nn.Module):
    """把预处理 + 主干 + head + nms + 定长 pad 全塞进 ONNX 图里"""

    def __init__(self, mmdet_model, img_h=800, img_w=800, max_dets=100):
        super().__init__()
        self.model = mmdet_model
        self.img_h = img_h
        self.img_w = img_w
        self.max_dets = max_dets
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        # x: (1,3,H,W) float32, [0,255], RGB
        x = (x - self.mean) / self.std

        # mmdet 3.x: data_preprocessor 已经被我们绕过，
        # 直接走 backbone -> neck -> head 拿结果
        feats = self.model.extract_feat(x)

        # 构造一个最小 batch_data_samples
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData
        sample = DetDataSample()
        sample.set_metainfo(dict(
            img_shape=(self.img_h, self.img_w),
            ori_shape=(self.img_h, self.img_w),
            scale_factor=(1.0, 1.0),
            batch_input_shape=(self.img_h, self.img_w),
        ))
        sample.gt_instances = InstanceData()
        samples = [sample]

        # 单 / 两阶段统一调 predict
        results = self.model.bbox_head.predict(feats, samples, rescale=False) \
            if hasattr(self.model, 'bbox_head') and self.model.bbox_head is not None \
            else self.model.roi_head.predict(feats, self.model.rpn_head.predict(feats, samples), samples, rescale=False)

        ins = results[0]
        bboxes = ins.bboxes  # (N, 4) xyxy
        scores = ins.scores  # (N,)
        labels = ins.labels  # (N,)

        # pad / truncate 到 max_dets
        n = bboxes.shape[0]
        if n >= self.max_dets:
            bboxes = bboxes[:self.max_dets]
            scores = scores[:self.max_dets]
            labels = labels[:self.max_dets]
        else:
            pad = self.max_dets - n
            bboxes = torch.cat([
                bboxes,
                bboxes.new_zeros((pad, 4)),
            ], dim=0)
            scores = torch.cat([
                scores,
                scores.new_zeros((pad,)),
            ], dim=0)
            labels = torch.cat([
                labels,
                labels.new_zeros((pad,), dtype=labels.dtype),
            ], dim=0)

        dets = torch.cat([bboxes, scores.unsqueeze(1)], dim=1)  # (max,5)
        return dets.unsqueeze(0), labels.unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True,
                    help='mmdet config，比如 configs/retinanet_r50_fpn_object.py')
    ap.add_argument('--ckpt',   required=True, help='训练产物 .pth')
    ap.add_argument('--out',    default='detect.onnx')
    ap.add_argument('--img-h',  type=int, default=800)
    ap.add_argument('--img-w',  type=int, default=800)
    ap.add_argument('--max-dets', type=int, default=100)
    ap.add_argument('--opset',   type=int, default=11)
    ap.add_argument('--device',  default='cuda', choices=['cuda', 'cpu'])
    args = ap.parse_args()

    from mmengine.config import Config
    from mmdet.apis import init_detector

    cfg = Config.fromfile(args.config)
    if 'norm_cfg' in cfg.model.get('backbone', {}):
        cfg.model.backbone['norm_cfg'] = dict(type='BN', requires_grad=True)
    if 'neck' in cfg.model and isinstance(cfg.model.neck, dict) and 'norm_cfg' in cfg.model.neck:
        cfg.model.neck['norm_cfg'] = dict(type='BN', requires_grad=True)

    model = init_detector(cfg, args.ckpt, device=args.device)
    model.eval()

    wrapper = DetONNX(model, args.img_h, args.img_w, args.max_dets).to(args.device).eval()

    dummy = torch.rand(1, 3, args.img_h, args.img_w, device=args.device) * 255.0
    with torch.no_grad():
        dets, labels = wrapper(dummy)
        print(f'[smoke] dets {dets.shape} {dets.dtype}, '
              f'labels {labels.shape} {labels.dtype}', flush=True)

    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        opset_version=args.opset,
        input_names=['input'],
        output_names=['dets', 'labels'],
        dynamic_axes=None,
        do_constant_folding=True,
    )
    print(f'[ok] exported to {args.out}', flush=True)

    try:
        import onnx
        import onnxsim
        m = onnx.load(args.out)
        m_sim, ok = onnxsim.simplify(m)
        assert ok, 'onnxsim simplify failed'
        onnx.save(m_sim, args.out)
        print(f'[ok] simplified -> {args.out}', flush=True)
    except Exception as e:
        print(f'[warn] onnxsim skipped: {e}', flush=True)


if __name__ == '__main__':
    main()
