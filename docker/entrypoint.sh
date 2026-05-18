#!/usr/bin/env bash
# 统一入口：第一个位置参数决定走训练 / 推理 / shell
#
# 调度系统会按文档约定调：
#   python run.py train
#   python predict.py infer --custom-config='task'
# 所以容器 ENTRYPOINT 用本脚本，可由 docker run 后面加 "train" 或 "infer"
# 也允许直接 docker run ... <镜像> bash 进交互调试。

set -e
cd /workspace/mmdet

export PYTHONUNBUFFERED=1
export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8

CMD="${1:-train}"
shift || true

case "$CMD" in
    train)
        echo ">>> python run.py train $*"
        exec python run.py train "$@"
        ;;
    infer|predict)
        echo ">>> python predict.py infer --custom-config='task' $*"
        exec python predict.py infer --custom-config='task' "$@"
        ;;
    sh|bash|shell)
        exec bash "$@"
        ;;
    *)
        echo "未知子命令: $CMD"
        echo "可用: train | infer | bash"
        exit 2
        ;;
esac
