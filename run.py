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
