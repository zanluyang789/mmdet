"""
系统集成入口 - 推理
====================

文档要求："模型推理时默认启动脚本：predict.py infer --custom-config='task'"

参数从 clie_lib/configs/task.conf 读取。
"""

import os
import sys


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
