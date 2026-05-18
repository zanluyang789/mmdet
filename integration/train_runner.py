"""
训练核心
========

提供两个函数：
    run_training(base_config, task_conf, device=None)
        单卡 / 已经在 torchrun 启动子进程下，直接 Runner.train()
    main_cli(argv=None)
        命令行入口，给 run.py 调

多卡场景：
    文档明确：python --nnodes=${nnodes} --node_rank=${node_rank} \
              --nproc_per_node=${nproc_per_node} --master_addr=${master_addr} \
              --master_port=${master_port} run.py
    其实就是 torchrun 包了一层，启动后每个进程都会跑到这里。
    我们读 env 里的 LOCAL_RANK / RANK / WORLD_SIZE 就够了。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from mmengine.runner import Runner

# 让相对导入在脚本式运行时也能 work
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from integration.conf_parser import load_task_conf
    from integration.config_builder import build_train_cfg
    from integration.device_utils import detect_device, setup_device_env
    from integration import filelist_dataset  # noqa: F401  触发注册
else:
    from .conf_parser import load_task_conf
    from .config_builder import build_train_cfg
    from .device_utils import detect_device, setup_device_env
    from . import filelist_dataset  # noqa: F401

# mmdet 3.x 的 default_scope='mmdet' 只触发 `import mmdet`，
# 不会自动把 mmdet.models.* / mmdet.datasets.* / mmdet.engine.* 全部注册进各自 registry。
# 我们这里走"双保险"：
#   1) 调 register_all_modules —— 大多数 mmdet 版本足够了
#   2) 再显式 import 关键子模块，避免某些 mmdet 安装里 __init__ 没拉全的情况
#      （观察到过的现象：register_all_modules 跑过但 DetDataPreProcessor 仍未注册）
import mmdet.models                       # noqa: F401  triggers @register_module
import mmdet.models.data_preprocessors    # noqa: F401  triggers DetDataPreProcessor
import mmdet.datasets                     # noqa: F401
import mmdet.engine                       # noqa: F401
import mmdet.evaluation                   # noqa: F401
import mmdet.structures                   # noqa: F401
import mmdet.visualization                # noqa: F401
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)


DEFAULT_BASE_CONFIG = "configs/retinanet_r50_fpn_object.py"
DEFAULT_TASK_CONF = "configs/task.conf"


def _pick_base_config(task_cfg) -> str:
    """允许 task.conf 用 base_config 字段覆盖默认 config 路径"""
    return task_cfg.get("base_config") or DEFAULT_BASE_CONFIG


def run_training(
    base_config: Optional[str] = None,
    task_conf_path: str = DEFAULT_TASK_CONF,
    device: Optional[str] = None,
):
    print(f"[run_training] task_conf = {os.path.abspath(task_conf_path)}", flush=True)
    task_cfg = load_task_conf(task_conf_path)
    print(f"[run_training] task_cfg keys = {sorted(task_cfg.keys())}", flush=True)

    base_config = base_config or _pick_base_config(task_cfg)
    print(f"[run_training] base_config = {base_config}", flush=True)

    device = setup_device_env(device or detect_device())
    print(f"[run_training] device = {device}", flush=True)

    cfg = build_train_cfg(base_config, task_cfg, device=device)

    if "RANK" in os.environ:
        print(
            f"[run_training] distributed: rank={os.environ.get('RANK')} "
            f"world_size={os.environ.get('WORLD_SIZE')} "
            f"local_rank={os.environ.get('LOCAL_RANK')}",
            flush=True,
        )
        cfg.launcher = "pytorch"
    else:
        cfg.launcher = "none"

    if cfg.work_dir:
        os.makedirs(cfg.work_dir, exist_ok=True)

    runner = Runner.from_cfg(cfg)
    runner.train()


def main_cli(argv=None) -> int:
    """
    系统调度调用：python run.py train
    可选 argv:
        --task-conf  configs/task.conf
        --base       configs/retinanet_r50_fpn_object.py
        --device     cuda/npu/cpu (覆盖自动探测)
    """
    parser = argparse.ArgumentParser("mmdet train (integration)")
    parser.add_argument(
        "action", nargs="?", default="train", choices=["train"],
        help="文档要求第一个位置参数是 'train'"
    )
    parser.add_argument("--task-conf", default=DEFAULT_TASK_CONF)
    parser.add_argument("--base", default=None,
                        help="mmdet base config（不传则用 configs/retinanet_r50_fpn_object.py "
                        "或 task.conf 里的 base_config）")
    parser.add_argument("--device", default=None, choices=[None, "cuda", "npu", "cpu"])
    # 文档里 ${nnodes} 等参数实际由 torchrun 处理，这里加几个占位
    parser.add_argument("--nnodes", default=None)
    parser.add_argument("--node_rank", default=None)
    parser.add_argument("--nproc_per_node", default=None)
    parser.add_argument("--master_addr", default=None)
    parser.add_argument("--master_port", default=None)
    args = parser.parse_args(argv)

    run_training(base_config=args.base, task_conf_path=args.task_conf, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
