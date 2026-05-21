"""
系统集成入口 - 推理
====================

文档要求："模型推理时默认启动脚本：predict.py infer --custom-config='task'"

参数从 clie_lib/configs/task.conf 读取。
"""

import os
import sys


# ============================================================================
# 加载 Ascend CANN 环境
# ----------------------------------------------------------------------------
# 跟 run.py 顶部那段完全一样：torch_npu 第一次让算子下沉 NPU 时会调
# AOE / GE 初始化 TBE。tbe 是 CANN toolkit 自带的 Python 模块（不在 pip 上），
# 必须把 `<toolkit>/python/site-packages` 加到 sys.path，否则推理也会撞：
#     ModuleNotFoundError: No module named 'tbe'
#     -> AOE failed to call InitCannKB
#     -> GEInitialize failed
#     -> RuntimeError: SetPrecisionMode ... error code is 500001
# 在 GPU / CPU 机器上找不到 set_env.sh 时静默跳过，不影响别的路径。
# ============================================================================
def _load_ascend_env(
    candidates=(
        "/usr/local/Ascend/ascend-toolkit/set_env.sh",
        "/usr/local/Ascend/nnae/set_env.sh",
        "/usr/local/Ascend/nnrt/set_env.sh",
        "/usr/local/Ascend/mindie/set_env.sh",
    ),
):
    import shlex
    import subprocess

    for script in candidates:
        if not os.path.isfile(script):
            continue
        try:
            out = subprocess.check_output(
                ["bash", "-c", f"source {shlex.quote(script)} >/dev/null 2>&1 && env -0"],
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(f"[predict.py] source {script} 失败: {exc}", flush=True)
            continue

        for entry in out.split(b"\x00"):
            if b"=" not in entry:
                continue
            k, _, v = entry.partition(b"=")
            os.environ[k.decode("utf-8", "ignore")] = v.decode("utf-8", "ignore")

        # PYTHONPATH 改 os.environ 对当前 Python 进程的 import 不生效，必须同步 sys.path
        for p in os.environ.get("PYTHONPATH", "").split(os.pathsep):
            if p and p not in sys.path:
                sys.path.insert(0, p)

        print(f"[predict.py] Ascend env loaded: {script}", flush=True)
        try:
            __import__("tbe")
            print("[predict.py] tbe importable ✓", flush=True)
        except ImportError as exc:
            print(f"[predict.py] WARN tbe still not importable: {exc}", flush=True)
        return

    print("[predict.py] 没找到任何 Ascend set_env.sh，跳过 CANN env 注入", flush=True)


_load_ascend_env()


# ============================================================================
# Patch mmcv.ops.nms 到 torchvision.ops.nms（CPU）
# ----------------------------------------------------------------------------
# 推理跟训练 val 走同一条路径（检测头 batched_nms -> mmcv.ops.nms），
# mmcv 在 mindie base 里编出来的 ext_module 没有 NPU NMS kernel，会报：
#     RuntimeError: nms_impl: implementation for device npu:0 not found.
# 把 mmcv.ops.nms.nms 替换成走 torchvision.ops.nms (CPU)，绕开 NPU 缺失。
# GPU / CPU 路径走原版，不受影响。
# ============================================================================
def _patch_mmcv_nms_for_npu():
    try:
        import mmcv.ops.nms as mmcv_nms_mod
        import numpy as np
        import torch
        import torchvision
    except ImportError as exc:
        print(f"[predict.py] mmcv/torchvision 不可用，跳过 NMS patch: {exc}", flush=True)
        return

    _orig_nms = mmcv_nms_mod.nms

    def _patched_nms(boxes, scores, iou_threshold,
                     offset=0, score_threshold=0, max_num=-1):
        if isinstance(boxes, np.ndarray):
            return _orig_nms(boxes, scores, iou_threshold,
                             offset, score_threshold, max_num)
        if not (hasattr(boxes, "device") and "npu" in str(boxes.device)):
            return _orig_nms(boxes, scores, iou_threshold,
                             offset, score_threshold, max_num)

        is_filtering_by_score = score_threshold > 0
        if is_filtering_by_score:
            valid_mask = scores > score_threshold
            boxes, scores = boxes[valid_mask], scores[valid_mask]
            valid_inds = torch.nonzero(valid_mask, as_tuple=False).squeeze(dim=1)

        device = boxes.device
        bx = boxes.detach().cpu().float()
        sc = scores.detach().cpu().float()
        if offset == 1:
            bx = bx.clone()
            bx[:, 2] += 1
            bx[:, 3] += 1

        inds = torchvision.ops.nms(bx, sc, float(iou_threshold)).to(device)

        if max_num > 0:
            inds = inds[:max_num]
        if is_filtering_by_score:
            inds = valid_inds[inds]

        dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
        return dets, inds

    mmcv_nms_mod.nms = _patched_nms
    print("[predict.py] patched mmcv.ops.nms.nms -> torchvision.ops.nms (CPU) for NPU", flush=True)


_patch_mmcv_nms_for_npu()
# ============================================================================
# 以上是 Ascend 环境注入 + mmcv NMS patch，下面是原 predict.py 业务逻辑（未改动）
# ============================================================================


def _ensure_repo_on_path():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


def main():
    _ensure_repo_on_path()
    from integration.infer_runner import main_cli

    sys.exit(main_cli())


if __name__ == "__main__":
    main()
