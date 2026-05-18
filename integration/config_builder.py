"""
task.conf -> mmdet Config 注入
==============================

输入：base config 路径 + task.conf 解析出来的 dict
输出：注入完所有参数的 mmengine.config.Config 对象，可直接喂给 Runner

注入的字段（与 mmseg 集成层的字段表保持一致，差异是检测专有的几项）：
    通用：
        work_dir          -> cfg.work_dir
        checkpoint_path   -> 触发 CkptSyncHook 把 .pth 同步到该目录
        log_path          -> cfg.work_dir（mmengine 自动在下面生成 log 子目录）
        tensorboard_log_path 启动 TB hook，dir 指向这里
        pretrained        -> 仅作为查找 backbone 预训练 pth 的根目录提示
        retrain_pth_url   -> cfg.load_from（再训练）
        use_tensorboard_scalar / use_tensorboard_image -> 配置 TB hook

    检测专有：
        data_file         -> YOLO 风格 data.yaml，覆盖 train/val 路径 & 类别
        use_filelist=True 时切到 FileListCocoDataset，需要：
            train_img_list / train_ann_file
            val_img_list   / val_ann_file
        否则用标准 CocoDataset，需要：
            train_ann_file (data_root/ann_file)
            val_ann_file
            train_img_prefix / val_img_prefix（默认空串）
        num_classes / classes_name / palette
            -> metainfo + 各 head 的 num_classes
        mean_file / std_file -> data_preprocessor.mean / .std

    训练：
        batch_size / max_iters / val_interval

    设备 / 分布式：
        norm_cfg -> SyncBN/BN
        dist_cfg.backend -> nccl / hccl
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional

from mmengine.config import Config

from .device_utils import detect_device, get_dist_backend, get_norm_cfg


def _read_floats_file(path: str, fallback):
    """读 mean/std 文件，每行一个 float（或一行逗号分隔），返回 list[float]"""
    if not path or not os.path.exists(path):
        return fallback
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for x in line.replace(",", " ").split():
                try:
                    vals.append(float(x))
                except ValueError:
                    pass
    if not vals:
        return fallback
    return vals


def _ensure_tuple(v):
    if isinstance(v, str):
        return tuple(s.strip().strip("'\"") for s in v.strip("()[] ").split(","))
    if isinstance(v, (list, tuple)):
        return tuple(v)
    return (str(v),)


def _find_local_pth(directory: str, key: str) -> Optional[str]:
    """在 directory 及其子目录（最多深 2 层）里找文件名包含 key 的 .pth。"""
    if not os.path.isdir(directory):
        return None
    for root, dirs, files in os.walk(directory):
        rel = os.path.relpath(root, directory)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 2:
            dirs[:] = []
            continue
        for f in files:
            if f.endswith(".pth") and key in f:
                return os.path.join(root, f)
    return None


def _resolve_pretrained(cfg: Config, pretrained_dir: str) -> None:
    """
    把 cfg.model.backbone.init_cfg 里的 'open-mmlab://xxx' / URL 替换成 pretrained_dir
    下的本地 .pth；找不到时设 TORCH_HOME=pretrained_dir，让 torch.hub 后续下载也
    落到这个目录（路径会是 pretrained_dir/hub/checkpoints/xxx.pth）。
    """
    if not pretrained_dir or not os.path.isdir(pretrained_dir):
        return

    # mmdet 3.x 主要通过 backbone.init_cfg=dict(type='Pretrained', checkpoint=...) 加载预训练
    backbone = cfg.model.get("backbone", {})
    init_cfg = backbone.get("init_cfg") if isinstance(backbone, dict) else None
    current = None
    if isinstance(init_cfg, dict):
        current = init_cfg.get("checkpoint")
    elif isinstance(init_cfg, (list, tuple)):
        for ic in init_cfg:
            if isinstance(ic, dict) and ic.get("checkpoint"):
                current = ic.get("checkpoint")
                break
    # 兼容老式 cfg.model.pretrained
    if not current:
        current = cfg.model.get("pretrained", None)

    if current and (os.path.isabs(current) and os.path.exists(current)):
        return

    key = None
    if current:
        last = current.split("//")[-1].split("/")[-1].split(":")[-1]
        key = last.replace(".pth", "")

    local = _find_local_pth(pretrained_dir, key) if key else None
    if local:
        if isinstance(init_cfg, dict):
            init_cfg["checkpoint"] = local
            cfg.model.backbone.init_cfg = init_cfg
        elif isinstance(init_cfg, (list, tuple)):
            new_init = []
            for ic in init_cfg:
                if isinstance(ic, dict) and ic.get("checkpoint"):
                    ic = dict(ic)
                    ic["checkpoint"] = local
                new_init.append(ic)
            cfg.model.backbone.init_cfg = new_init
        else:
            cfg.model.backbone.init_cfg = dict(type="Pretrained", checkpoint=local)
        # 老接口同时同步
        if "pretrained" in cfg.model:
            cfg.model.pretrained = local
        print(f"[pretrained] {current} -> {local}", flush=True)
    else:
        os.environ["TORCH_HOME"] = pretrained_dir
        print(
            f"[pretrained] 本地没找到 '{key}' 相关 pth，"
            f"已设 TORCH_HOME={pretrained_dir}，"
            f"torch.hub 会下载到 {pretrained_dir}/hub/checkpoints/",
            flush=True,
        )


def _patch_detector_classes(cfg: Config, num_classes: int):
    """把检测模型里所有 num_classes 字段统一改成新值。

    覆盖：
      - roi_head.bbox_head.num_classes  (Faster R-CNN 等两阶段)
      - bbox_head.num_classes           (单阶段：RetinaNet/FCOS/YOLOX/YOLOv3 等)
      - mask_head 同步
    """
    model = cfg.model

    def _set_nc(obj):
        if isinstance(obj, dict) and "num_classes" in obj:
            obj["num_classes"] = num_classes
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _set_nc(item)

    # roi_head -> bbox_head / mask_head
    roi = model.get("roi_head")
    if isinstance(roi, dict):
        _set_nc(roi.get("bbox_head"))
        _set_nc(roi.get("mask_head"))

    # 单阶段
    _set_nc(model.get("bbox_head"))

    # RPN 通常 num_classes=1（前景/背景），不动


def _switch_to_filelist_dataset(
    loader_cfg: Dict,
    img_list: str,
    ann_file: str,
    metainfo: Dict,
    pipeline=None,
):
    """把现成的 dataloader.dataset 改成 FileListCocoDataset"""
    new_ds = dict(
        type="FileListCocoDataset",
        img_list_file=img_list,
        ann_file=ann_file,
        metainfo=metainfo,
        pipeline=pipeline if pipeline is not None else loader_cfg["dataset"].get("pipeline"),
    )
    for k in ("filter_cfg", "backend_args"):
        if k in loader_cfg["dataset"]:
            new_ds[k] = loader_cfg["dataset"][k]
    loader_cfg["dataset"] = new_ds


def _switch_to_coco_dataset(
    loader_cfg: Dict,
    ann_file: str,
    img_prefix: str,
    metainfo: Dict,
    pipeline=None,
):
    """切到标准 CocoDataset：用 ann_file + data_prefix"""
    new_ds = dict(
        type="CocoDataset",
        ann_file=ann_file,
        data_prefix=dict(img=img_prefix or ""),
        metainfo=metainfo,
        pipeline=pipeline if pipeline is not None else loader_cfg["dataset"].get("pipeline"),
    )
    for k in ("filter_cfg", "backend_args"):
        if k in loader_cfg["dataset"]:
            new_ds[k] = loader_cfg["dataset"][k]
    loader_cfg["dataset"] = new_ds


def _maybe_apply_data_yaml(task_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    若 task_cfg 里给了 data_file（YOLO 风格 yaml），解析它，并把字段反填到
    task_cfg（不存在的才填，不覆盖已显式给的）。返回处理后的 task_cfg。

    扩展：如果 data.yaml 只给了图像目录、没给 train_ann/val_ann，
    自动在 <image_dir>/../{labels,label} 探测 YOLO .txt，转 COCO JSON 缓存到
    work_dir/auto_coco/{train,val}.json，并回填 train_ann_file / val_ann_file。
    """
    data_file = task_cfg.get("data_file")
    if not data_file or not os.path.exists(data_file):
        return task_cfg

    from .data_yaml import parse_data_yaml

    info = parse_data_yaml(data_file)
    print(f"[data_yaml] 解析 {data_file} -> {info}", flush=True)

    # train/val 图像目录 / 列表
    if info.get("train_img") and not task_cfg.get("train_img_prefix"):
        task_cfg.setdefault("train_img_prefix", info["train_img"])
    if info.get("val_img") and not task_cfg.get("val_img_prefix"):
        task_cfg.setdefault("val_img_prefix", info["val_img"])
    # 标注文件
    if info.get("train_ann") and not task_cfg.get("train_ann_file"):
        task_cfg.setdefault("train_ann_file", info["train_ann"])
    if info.get("val_ann") and not task_cfg.get("val_ann_file"):
        task_cfg.setdefault("val_ann_file", info["val_ann"])
    # 类别
    if info.get("classes") and not task_cfg.get("classes_name"):
        task_cfg.setdefault("classes_name", tuple(info["classes"]))
    if info.get("num_classes") and not task_cfg.get("num_classes"):
        task_cfg.setdefault("num_classes", int(info["num_classes"]))

    # YOLO -> COCO 自动转换（仅在没有 COCO ann 但有 image_dir + names 时触发）
    _autogen_coco_from_yolo(task_cfg, info)

    return task_cfg


def _autogen_coco_from_yolo(task_cfg: Dict[str, Any], info: Dict[str, Any]) -> None:
    """
    探测 YOLO 标签目录并转 COCO JSON，反填 task_cfg。
    无效 / 无标签情况静默跳过（让 train_ann_file 保持 None，
    下游的 'train_ann_file 是空字符串' 报错更清晰）。
    """
    classes = task_cfg.get("classes_name") or info.get("classes")
    if not classes:
        return
    if isinstance(classes, str):
        classes = tuple(s.strip().strip("'\"") for s in
                        classes.strip("()[] ").split(",") if s.strip())
    names = list(classes)
    if not names:
        return

    work_dir = (task_cfg.get("work_dir") or task_cfg.get("work_space")
                or task_cfg.get("log_path") or ".")
    cache_dir = os.path.join(work_dir, "auto_coco")

    from .yolo2coco import find_labels_dir, yolo_to_coco

    # train
    if not task_cfg.get("train_ann_file"):
        img_dir = info.get("train_img") or task_cfg.get("train_img_prefix")
        if img_dir and os.path.isdir(img_dir):
            label_dir = find_labels_dir(img_dir)
            if label_dir:
                out = os.path.join(cache_dir, "train.json")
                try:
                    yolo_to_coco(img_dir, label_dir, names, out)
                    task_cfg["train_ann_file"] = out
                    task_cfg.setdefault("train_img_prefix", img_dir)
                except Exception as e:
                    print(f"[yolo2coco] train 转换失败: {e}", flush=True)
            else:
                print(f"[yolo2coco] 没找到 train labels 目录 (image_dir={img_dir})",
                      flush=True)

    # val
    if not task_cfg.get("val_ann_file"):
        img_dir = info.get("val_img") or task_cfg.get("val_img_prefix")
        if img_dir and os.path.isdir(img_dir):
            label_dir = find_labels_dir(img_dir)
            if label_dir:
                out = os.path.join(cache_dir, "val.json")
                try:
                    yolo_to_coco(img_dir, label_dir, names, out)
                    task_cfg["val_ann_file"] = out
                    task_cfg.setdefault("val_img_prefix", img_dir)
                except Exception as e:
                    print(f"[yolo2coco] val 转换失败: {e}", flush=True)
            else:
                print(f"[yolo2coco] 没找到 val labels 目录 (image_dir={img_dir})",
                      flush=True)


def build_train_cfg(
    base_cfg_path: str,
    task_cfg: Dict[str, Any],
    device: Optional[str] = None,
) -> Config:
    """
    主入口：base config + task.conf -> 注入后的 Config
    """
    if not os.path.exists(base_cfg_path):
        raise FileNotFoundError(f"base config 不存在: {base_cfg_path}")

    cfg: Config = Config.fromfile(base_cfg_path)
    device = device or detect_device()

    # 先吃掉 data_file
    task_cfg = _maybe_apply_data_yaml(task_cfg)

    # -------- 输出 / 工作目录 --------
    log_path = task_cfg.get("log_path")
    work_dir_cfg = task_cfg.get("work_dir") or task_cfg.get("work_space")
    if log_path:
        cfg.work_dir = log_path
    elif work_dir_cfg:
        cfg.work_dir = work_dir_cfg
    os.makedirs(cfg.work_dir, exist_ok=True)

    # ckpt 同步
    ckpt_dir = task_cfg.get("checkpoint_path")
    if ckpt_dir:
        from . import hooks  # noqa: F401  触发 CkptSyncHook 注册

        os.makedirs(ckpt_dir, exist_ok=True)
        custom_hooks = list(cfg.get("custom_hooks", []) or [])
        custom_hooks.append(dict(type="CkptSyncHook", ckpt_dir=ckpt_dir, link=False))
        cfg.custom_hooks = custom_hooks

    # -------- 再训练 --------
    retrain = task_cfg.get("retrain_pth_url")
    if retrain:
        cfg.load_from = retrain
        cfg.resume = bool(task_cfg.get("resume_mode") and task_cfg["resume_mode"] != "None")

    # -------- 预训练 backbone 权重 --------
    pretrained_dir = task_cfg.get("pretrained")
    if pretrained_dir:
        _resolve_pretrained(cfg, pretrained_dir)

    # -------- 类别 / palette --------
    num_classes = task_cfg.get("num_classes")
    classes_name = task_cfg.get("classes_name")
    palette = task_cfg.get("palette")

    if num_classes is not None:
        _patch_detector_classes(cfg, int(num_classes))

    metainfo = None
    if classes_name:
        classes_name = _ensure_tuple(classes_name)
        # palette 没给就自动生成
        if not palette:
            try:
                import numpy as np
                rng = np.random.default_rng(seed=42)
                palette = rng.integers(0, 255, size=(len(classes_name), 3)).tolist()
            except Exception:
                palette = [[0, 0, 0]] * len(classes_name)
        metainfo = dict(classes=classes_name, palette=list(palette))
        for loader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
            loader = cfg.get(loader_name)
            if loader and "dataset" in loader:
                loader["dataset"]["metainfo"] = metainfo
    else:
        # 沿用 base config 自带的 metainfo
        for loader_name in ("train_dataloader",):
            loader = cfg.get(loader_name)
            if loader and "dataset" in loader and "metainfo" in loader["dataset"]:
                metainfo = loader["dataset"]["metainfo"]
                break

    # -------- mean / std --------
    mean = _read_floats_file(task_cfg.get("mean_file", ""), None)
    std = _read_floats_file(task_cfg.get("std_file", ""), None)
    if mean and std and "data_preprocessor" in cfg.model:
        cfg.model.data_preprocessor.mean = mean
        cfg.model.data_preprocessor.std = std

    # -------- 数据集：文件列表 / COCO --------
    use_filelist = bool(task_cfg.get("use_filelist"))
    train_ann_file = task_cfg.get("train_ann_file")
    val_ann_file = task_cfg.get("val_ann_file")
    train_img_prefix = task_cfg.get("train_img_prefix") or ""
    val_img_prefix = task_cfg.get("val_img_prefix") or ""

    if use_filelist:
        from . import filelist_dataset  # noqa: F401

        # metainfo 兜底
        if metainfo is None:
            for loader_name in ("train_dataloader",):
                loader = cfg.get(loader_name)
                if loader and "dataset" in loader and "metainfo" in loader["dataset"]:
                    metainfo = loader["dataset"]["metainfo"]
                    break

        train_img_list = task_cfg.get("train_img_list")
        val_img_list = task_cfg.get("val_img_list")

        if train_img_list and train_ann_file and cfg.get("train_dataloader"):
            _switch_to_filelist_dataset(
                cfg.train_dataloader, train_img_list, train_ann_file, metainfo
            )
        if val_img_list and val_ann_file and cfg.get("val_dataloader"):
            _switch_to_filelist_dataset(
                cfg.val_dataloader, val_img_list, val_ann_file, metainfo
            )
            if cfg.get("test_dataloader"):
                cfg.test_dataloader = deepcopy(cfg.val_dataloader)
    else:
        # 标准 COCO 模式：只在 train/val_ann_file 显式给了的时候去覆盖
        if train_ann_file and cfg.get("train_dataloader"):
            _switch_to_coco_dataset(
                cfg.train_dataloader, train_ann_file, train_img_prefix, metainfo
            )
        if val_ann_file and cfg.get("val_dataloader"):
            _switch_to_coco_dataset(
                cfg.val_dataloader, val_ann_file, val_img_prefix, metainfo
            )
            if cfg.get("test_dataloader"):
                cfg.test_dataloader = deepcopy(cfg.val_dataloader)

    # val_evaluator 的 ann_file 同步
    if val_ann_file:
        ve = cfg.get("val_evaluator")
        if isinstance(ve, dict) and ve.get("type") == "CocoMetric":
            ve["ann_file"] = val_ann_file
        if cfg.get("test_evaluator") and isinstance(cfg.test_evaluator, dict):
            if cfg.test_evaluator.get("type") == "CocoMetric":
                cfg.test_evaluator["ann_file"] = val_ann_file

    # -------- 批大小 / 训练长度 --------
    batch_size = task_cfg.get("batch_size")
    if batch_size and cfg.get("train_dataloader"):
        cfg.train_dataloader.batch_size = int(batch_size)

    # mmdet 默认 EpochBasedTrainLoop，但本集成层走 iter-based 更对齐 mmseg 风格
    max_iters = task_cfg.get("max_iters")
    val_interval = task_cfg.get("val_interval")
    if max_iters and cfg.get("train_cfg"):
        tc = cfg.train_cfg
        if tc.get("type", "").endswith("EpochBasedTrainLoop"):
            # 简单换算：1 epoch ≈ ? iter 没法精确，这里直接换 IterBasedTrainLoop
            tc["type"] = "IterBasedTrainLoop"
            tc.pop("max_epochs", None)
        tc["max_iters"] = int(max_iters)
        if val_interval:
            tc["val_interval"] = int(val_interval)
        cfg.train_cfg = tc

    # -------- 设备 / 分布式 --------
    if "backbone" in cfg.model:
        cfg.model.backbone["norm_cfg"] = get_norm_cfg(device)
    if "neck" in cfg.model and isinstance(cfg.model.neck, dict):
        # FPN 用 GN/BN 居多，norm_cfg 不一定有；有就改
        if "norm_cfg" in cfg.model.neck:
            cfg.model.neck["norm_cfg"] = get_norm_cfg(device)
    if "roi_head" in cfg.model and isinstance(cfg.model.roi_head, dict):
        bbox = cfg.model.roi_head.get("bbox_head")
        if isinstance(bbox, dict) and "norm_cfg" in bbox:
            bbox["norm_cfg"] = get_norm_cfg(device)
    if "bbox_head" in cfg.model and isinstance(cfg.model.bbox_head, dict):
        if "norm_cfg" in cfg.model.bbox_head:
            cfg.model.bbox_head["norm_cfg"] = get_norm_cfg(device)

    cfg.env_cfg.dist_cfg.backend = get_dist_backend(device)

    # -------- TensorBoard --------
    tb_dir = task_cfg.get("tensorboard_log_path")
    use_scalar = task_cfg.get("use_tensorboard_scalar", False)
    use_image = task_cfg.get("use_tensorboard_image", False)
    if tb_dir and (use_scalar or use_image):
        os.makedirs(tb_dir, exist_ok=True)
        backends = list(cfg.get("vis_backends", []) or [])
        backends.append(dict(type="TensorboardVisBackend", save_dir=tb_dir))
        cfg.vis_backends = backends
        if "visualizer" in cfg:
            cfg.visualizer["vis_backends"] = backends
            cfg.visualizer["save_dir"] = tb_dir

    # -------- env 标记 --------
    env = task_cfg.get("env")
    if env:
        cfg.experiment_name = str(env)

    return cfg


if __name__ == "__main__":
    # 调试入口：python -m integration.config_builder configs/retinanet_r50_fpn_object.py configs/task.conf
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else "configs/retinanet_r50_fpn_object.py"
    conf = sys.argv[2] if len(sys.argv) > 2 else "configs/task.conf"
    from .conf_parser import load_task_conf

    tc = load_task_conf(conf)
    cfg = build_train_cfg(base, tc)
    print(cfg.pretty_text)
