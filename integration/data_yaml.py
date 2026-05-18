"""
YOLO 风格 data.yaml 解析
========================

平台下发的 task.conf 里有时只给一个 data_file 字段（看推理 task.conf 例子），
内容是 YOLO/Ultralytics 风格的 yaml：

    path: /data/.../dataset
    train: images/train     # 子目录，或 train.txt（每行一张图）
    val:   images/val
    test:  images/test
    nc:    20
    names: ['car', 'truck', ...]    # 或 {0: car, 1: truck}
    # 可选 mmdet 风格扩展：
    train_ann: annotations/train.json
    val_ann:   annotations/val.json

本模块负责把这一份 yaml 解析成统一字典：
    {
        'data_root':   path,
        'train_img':   绝对/相对 images 目录 或 .txt 文件,
        'val_img':     同上,
        'test_img':    同上,
        'train_ann':   .json 路径（若提供），否则 None
        'val_ann':     同上
        'test_ann':    同上
        'classes':     ('car','truck',...) 元组
        'num_classes': 20
    }

约定：
- 缺 yaml 解析库时回退到极简的 line-by-line 解析（只支持 key: value）
- 路径里 ~ 自动展开；相对路径相对 data_root 解析
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple


def _try_yaml_load(path: str) -> Dict[str, Any]:
    """优先用 pyyaml，没装就走极简 fallback。"""
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"data.yaml 顶层不是字典: {type(data)}")
        return data
    except ImportError:
        pass

    # ---- fallback: 仅支持 'key: value' 一行写完的形式 ----
    result: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip().strip(",")
            if not v:
                continue
            # 简易类型推断
            if v.startswith("[") and v.endswith("]"):
                inner = v[1:-1]
                items = [x.strip().strip("'\"") for x in re.split(r",\s*", inner) if x.strip()]
                result[k] = items
            elif re.fullmatch(r"-?\d+", v):
                result[k] = int(v)
            elif re.fullmatch(r"-?\d+\.\d+", v):
                result[k] = float(v)
            elif v.lower() in ("true", "false"):
                result[k] = (v.lower() == "true")
            else:
                result[k] = v.strip("'\"")
    return result


def _names_to_tuple(names) -> Tuple[str, ...]:
    """names 可以是 list 也可以是 dict（{0:'car',1:'truck',...}）。"""
    if names is None:
        return tuple()
    if isinstance(names, dict):
        # 按 key 排序
        items = sorted(names.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])
        return tuple(str(v) for _, v in items)
    if isinstance(names, (list, tuple)):
        return tuple(str(v) for v in names)
    raise ValueError(f"names 不识别的类型: {type(names)}")


def _abs(base: str, p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base, p))


def parse_data_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    解析 data.yaml -> 统一字典。
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml 不存在: {yaml_path}")

    raw = _try_yaml_load(yaml_path)

    data_root = raw.get("path") or os.path.dirname(os.path.abspath(yaml_path))
    data_root = os.path.expanduser(data_root)
    if not os.path.isabs(data_root):
        data_root = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(yaml_path)), data_root))

    classes = _names_to_tuple(raw.get("names"))
    nc = raw.get("nc")
    if nc is None and classes:
        nc = len(classes)

    return {
        "data_root":   data_root,
        "train_img":   _abs(data_root, raw.get("train")),
        "val_img":     _abs(data_root, raw.get("val")),
        "test_img":    _abs(data_root, raw.get("test")),
        "train_ann":   _abs(data_root, raw.get("train_ann") or raw.get("train_ann_file")),
        "val_ann":     _abs(data_root, raw.get("val_ann")   or raw.get("val_ann_file")),
        "test_ann":    _abs(data_root, raw.get("test_ann")  or raw.get("test_ann_file")),
        "classes":     classes,
        "num_classes": int(nc) if nc is not None else None,
    }


if __name__ == "__main__":
    import json
    import sys

    p = sys.argv[1]
    print(json.dumps(parse_data_yaml(p), ensure_ascii=False, indent=2))
