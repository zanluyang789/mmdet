"""
设备探测：自动决定走 GPU / NPU / CPU
====================================

约定：
- 优先级 NPU > GPU > CPU，可被环境变量 INTEGRATION_DEVICE 强制覆盖（cuda/npu/cpu）
- 训练时根据设备返回合适的 norm_cfg（NPU 不支持 SyncBN，要回退到 BN）
- 分布式 backend：cuda -> nccl，npu -> hccl，cpu -> gloo

返回结构尽量小，调用方不依赖 torch 类型，避免"导一下设备探测就要装 torch_npu"。
"""

from __future__ import annotations

import os
from typing import Dict, Optional


def _try_torch_npu_available() -> bool:
    """torch_npu 装了 && 至少 1 张可用 NPU"""
    try:
        import torch_npu  # noqa: F401
        import torch

        # torch_npu 注册后 torch.npu.is_available() 可用
        return bool(torch.npu.is_available())
    except Exception:
        return False


def _try_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def detect_device(force: Optional[str] = None) -> str:
    """返回 'npu' / 'cuda' / 'cpu'"""
    force = force or os.environ.get("INTEGRATION_DEVICE", "")
    force = force.lower() if force else ""
    if force in ("npu", "cuda", "cpu"):
        return force

    if _try_torch_npu_available():
        return "npu"
    if _try_cuda_available():
        return "cuda"
    return "cpu"


def get_dist_backend(device: Optional[str] = None) -> str:
    """根据设备给 mmengine env_cfg.dist_cfg.backend"""
    d = device or detect_device()
    return {"cuda": "nccl", "npu": "hccl", "cpu": "gloo"}.get(d, "gloo")


def get_norm_cfg(device: Optional[str] = None) -> Dict:
    """
    NPU 上 SyncBN 不可用（torch_npu 没有完整支持），回退到 BN。
    GPU 多卡场景仍推荐 SyncBN。
    """
    d = device or detect_device()
    if d == "npu":
        return dict(type="BN", requires_grad=True)
    return dict(type="SyncBN", requires_grad=True)


def setup_device_env(device: Optional[str] = None) -> str:
    """
    在 main() 开头调用一次：
        - 如果是 NPU，import torch_npu（触发 backend 注册）
        - 设置 OMP / CUBLAS 之类的环境变量（NPU 上有一些坑）
    返回最终生效的 device。
    """
    d = device or detect_device()
    if d == "npu":
        try:
            import torch_npu  # noqa: F401
            # NPU 上 dataloader 的 fork 行为有坑，统一切 spawn
            os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
        except ImportError:
            raise RuntimeError(
                "INTEGRATION_DEVICE=npu 但找不到 torch_npu，"
                "请确认 CANN + torch_npu 已正确安装"
            )
    return d


if __name__ == "__main__":
    d = detect_device()
    print(f"detected device: {d}")
    print(f"dist backend:    {get_dist_backend(d)}")
    print(f"norm_cfg:        {get_norm_cfg(d)}")
