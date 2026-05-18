"""
FileListCocoDataset
===================

mmseg 那套用的是 use_filelist + train_img_list（每行一张图绝对路径）+ train_gt_list
（每行一张 mask 绝对路径）。检测换成 COCO 标注后，没有"一张图一个标注文件"
的对应关系，所以做法是：

    - img_list_file: 每行一张图的绝对路径（系统下发的标准格式）
    - ann_file:      标准 COCO JSON（含 images / annotations / categories）

我们在 mmdet 的 CocoDataset 之上加一层"路径覆盖"：
    加载 COCO JSON 后，遍历 self.data_list 里每条 img_info，
    用其 basename 去 img_list_file 里查对应的绝对路径，匹配上就覆盖 img_path，
    匹配不上的样本保留 ann_file 里的相对路径（兜底）。

这样平台可以：
    1) 训练数据散在 /data/foo/.../*.png（路径任意）
    2) COCO JSON 里只写 file_name（不带目录）
    3) img_list.txt 给出真实绝对路径，FileListCocoDataset 自动配对

注册名: 'FileListCocoDataset'
"""

from __future__ import annotations

import os
from typing import List, Optional

from mmdet.datasets.coco import CocoDataset
from mmdet.registry import DATASETS


def _read_list(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"列表文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


@DATASETS.register_module()
class FileListCocoDataset(CocoDataset):
    """
    Args:
        img_list_file: txt，每行一张训练图的绝对路径
        ann_file:      COCO JSON 路径（mmdet CocoDataset 的标准参数）
        其它参数透传 CocoDataset
    """

    # 同 mmseg：不预设 classes/palette，由 task.conf 注入 metainfo
    METAINFO = dict(classes=tuple(), palette=tuple())

    def __init__(
        self,
        img_list_file: str,
        ann_file: str,
        pipeline=None,
        metainfo=None,
        data_root: Optional[str] = None,
        data_prefix: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self._img_list_file = img_list_file
        # CocoDataset 需要 data_prefix['img']，给个空串避免拼接乱
        data_prefix = data_prefix or dict(img="")
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            metainfo=metainfo,
            data_root=data_root or "",
            data_prefix=data_prefix,
            **kwargs,
        )

    # CocoDataset.load_data_list() 在 __init__ 里被调用一次；
    # 这里覆盖它，先调父类拿到标准 data_list，再按 basename 覆盖 img_path。
    def load_data_list(self) -> List[dict]:
        data_list = super().load_data_list()

        try:
            abs_paths = _read_list(self._img_list_file)
        except FileNotFoundError as e:
            print(f"[FileListCocoDataset] {e}, 保留 COCO 原 file_name", flush=True)
            return data_list

        # basename(无后缀大小写) -> 绝对路径
        name2abs = {}
        for p in abs_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            name2abs.setdefault(stem, p)
            # 也支持完全匹配 basename（带后缀）
            name2abs.setdefault(os.path.basename(p), p)

        miss = 0
        for info in data_list:
            ip = info.get("img_path", "")
            if not ip:
                continue
            base = os.path.basename(ip)
            stem = os.path.splitext(base)[0]
            if base in name2abs:
                info["img_path"] = name2abs[base]
            elif stem in name2abs:
                info["img_path"] = name2abs[stem]
            else:
                miss += 1

        if miss:
            print(f"[FileListCocoDataset] {miss}/{len(data_list)} 张图未在 "
                  f"{self._img_list_file} 中找到匹配，将沿用 COCO 原路径", flush=True)
        else:
            print(f"[FileListCocoDataset] 全部 {len(data_list)} 张图均匹配 "
                  f"{self._img_list_file}", flush=True)
        return data_list
