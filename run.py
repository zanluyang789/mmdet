"""
系统集成入口 - 训练
====================

文档要求："模型训练默认启动脚本：python run.py train"

支持的命令行（系统会用 ${nnodes} 等替换后调过来）：
    单卡：       python run.py train
    多卡：       python -m torch.distributed.run --nnodes=1 --node_rank=0 \
                    --nproc_per_node=4 --master_addr=127.0.0.1 \
                    --master_port=29500 run.py train

    或者按文档原文的写法：
        python --nnodes=${nnodes} --node_rank=${node_rank} \
            --nproc_per_node=${nproc_per_node} --master_addr=${master_addr} \
            --master_port=${master_port} run.py train

参数从 configs/task.conf 读取。
设备由 INTEGRATION_DEVICE 强制 / 自动探测（torch_npu 优先 -> CUDA -> CPU）。
"""

import os
import sys


# ============================================================================
# 加载 Ascend CANN 环境
# ----------------------------------------------------------------------------
# NPU 机器跟内网 git 不通、镜像也不方便重打，所以把"source set_env.sh"挪到
# Python 进程启动时做：bash 跑一次 source，把它产出的 env 抄进 os.environ，
# PYTHONPATH 再同步到 sys.path（PYTHONPATH 改 env 对**当前** Python 进程
# 的 import 不生效，必须同时改 sys.path）。
#
# 为什么这步必须有：torch_npu 在第一次让算子下沉 NPU 时会调 AOE / GE
# 初始化 TBE。tbe 是 CANN toolkit 自带的 Python 模块（**不在 pip 上**），
# 必须把 `<toolkit>/python/site-packages` 加到 sys.path。没做就报：
#     ModuleNotFoundError: No module named 'tbe'
#     -> AOE failed to call InitCannKB
#     -> GEInitialize failed
#     -> RuntimeError: SetPrecisionMode ... error code is 500001
#
# 找不到 set_env.sh（比如在 GPU/CPU 机器上跑）就静默跳过，不影响别的路径。
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
            print(f"[run.py] source {script} 失败: {exc}", flush=True)
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

        print(f"[run.py] Ascend env loaded: {script}", flush=True)
        try:
            __import__("tbe")
            print("[run.py] tbe importable ✓", flush=True)
        except ImportError as exc:
            print(f"[run.py] WARN tbe still not importable: {exc}", flush=True)
        return

    print("[run.py] 没找到任何 Ascend set_env.sh，跳过 CANN env 注入", flush=True)


_load_ascend_env()


# ============================================================================
# Patch mmcv.ops.nms 到 torchvision.ops.nms（CPU）
# ----------------------------------------------------------------------------
# mmcv 在 mindie base 里编 C++ extension 时只带了 CPU/CUDA NMS kernel，
# 没注册 NPU 实现。val 阶段检测头跑 batched_nms 时会报：
#     RuntimeError: nms_impl: implementation for device npu:0 not found.
#
# mmcv.ops.nms.batched_nms 内部用 eval('nms') 在本模块作用域取 nms 函数，
# 所以替换 mmcv.ops.nms.nms 这个 module attribute 就够，不需要 patch
# autograd.Function。NPU tensor 拷到 CPU 跑 torchvision.ops.nms，inds
# 再拷回原 device。NMS 前已经按 score_thr 过滤，剩下几百~几千个 bbox，
# 拷贝 + CPU NMS 总开销 < 1ms，对 val 时长几乎无感。
#
# GPU / CPU 路径走原版 nms，不受影响。
# ============================================================================
def _patch_mmcv_nms_for_npu():
    try:
        import mmcv.ops.nms as mmcv_nms_mod
        import numpy as np
        import torch
        import torchvision
    except ImportError as exc:
        print(f"[run.py] mmcv/torchvision 不可用，跳过 NMS patch: {exc}", flush=True)
        return

    _orig_nms = mmcv_nms_mod.nms

    def _patched_nms(boxes, scores, iou_threshold,
                     offset=0, score_threshold=0, max_num=-1):
        # ndarray 走原版（不上 NPU）
        if isinstance(boxes, np.ndarray):
            return _orig_nms(boxes, scores, iou_threshold,
                             offset, score_threshold, max_num)
        # 非 NPU device 走原版（GPU / CPU 都正常）
        if not (hasattr(boxes, "device") and "npu" in str(boxes.device)):
            return _orig_nms(boxes, scores, iou_threshold,
                             offset, score_threshold, max_num)

        # NPU tensor：拷 CPU + torchvision NMS
        is_filtering_by_score = score_threshold > 0
        if is_filtering_by_score:
            valid_mask = scores > score_threshold
            boxes, scores = boxes[valid_mask], scores[valid_mask]
            valid_inds = torch.nonzero(valid_mask, as_tuple=False).squeeze(dim=1)

        device = boxes.device
        bx = boxes.detach().cpu().float()
        sc = scores.detach().cpu().float()
        # mmcv offset=1 风格：bbox 是 (x1,y1,x2-1,y2-1)；torchvision 是 offset=0 风格
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
    print("[run.py] patched mmcv.ops.nms.nms -> torchvision.ops.nms (CPU) for NPU", flush=True)


_patch_mmcv_nms_for_npu()
# ============================================================================
# 以上是 Ascend 环境注入 + mmcv NMS patch，下面是原 run.py 业务逻辑（未改动）
# ============================================================================


def _ensure_repo_on_path():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


def main():
    _ensure_repo_on_path()
    from integration.train_runner import main_cli

    sys.exit(main_cli())


if __name__ == "__main__":
    main()
