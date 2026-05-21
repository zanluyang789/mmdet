#!/usr/bin/env bash
# 统一入口：第一个位置参数决定走训练 / 推理 / shell
#
# 调度系统会按文档约定调：
#   python run.py train
#   python predict.py infer --custom-config='task'
# 所以容器 ENTRYPOINT 用本脚本，可由 docker run 后面加 "train" 或 "infer"
# 也允许直接 docker run ... <镜像> bash 进交互调试。

cd /workspace/mmdet

# =====================================================================
# 加载 Ascend CANN 环境
# ---------------------------------------------------------------------
# torch_npu 在第一次让算子下沉到 NPU 时会调 AOE/GE 初始化 TBE。tbe 是 CANN
# toolkit 自带的 Python 模块（不在 pip 上），必须 source set_env.sh 才能把
#   - PYTHONPATH      (.../python/site-packages, 含 tbe / te / auto_tune)
#   - LD_LIBRARY_PATH (.../lib64, 含 libascendcl.so 等)
#   - ASCEND_OPP_PATH / ASCEND_AICPU_PATH / TOOLCHAIN_HOME
# 注入到容器里。没做这一步会看到这条经典报错链：
#   ModuleNotFoundError: No module named 'tbe'
#   -> AOE failed to call InitCannKB
#   -> GEInitialize failed
#   -> RuntimeError: SetPrecisionMode ... error code is 500001
# =====================================================================
# set_env.sh 里偶尔会 return 非 0，先关掉 errexit 再 source。
set +e
_ASCEND_ENV_LOADED=0
for _f in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/nnae/set_env.sh \
    /usr/local/Ascend/nnrt/set_env.sh \
    /usr/local/Ascend/mindie/set_env.sh \
    /usr/local/Ascend/atc/bin/setenv.bash ; do
    if [ -f "$_f" ]; then
        echo ">>> source $_f"
        # shellcheck disable=SC1090
        source "$_f"
        _ASCEND_ENV_LOADED=1
    fi
done

# 兜底：set_env.sh 没找到时，按默认安装路径手动注入 tbe / opp / lib64
if [ "$_ASCEND_ENV_LOADED" -eq 0 ]; then
    echo "[warn] 没找到任何 Ascend set_env.sh，尝试手动注入默认路径"
    _ATK=/usr/local/Ascend/ascend-toolkit/latest
    if [ -d "$_ATK" ]; then
        export ASCEND_TOOLKIT_HOME="$_ATK"
        export ASCEND_HOME_PATH="$_ATK"
        export ASCEND_OPP_PATH="$_ATK/opp"
        export ASCEND_AICPU_PATH="$_ATK"
        export TOOLCHAIN_HOME="$_ATK/toolkit"
        export PATH="$_ATK/bin:$_ATK/compiler/ccec_compiler/bin:$PATH"
        export LD_LIBRARY_PATH="$_ATK/lib64:$_ATK/lib64/plugin/opskernel:$_ATK/lib64/plugin/nnengine:${LD_LIBRARY_PATH}"
        export PYTHONPATH="$_ATK/python/site-packages:$_ATK/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH}"
    else
        echo "[error] 也找不到 $_ATK，tbe 模块仍然不会在 PYTHONPATH 里！"
    fi
fi

# 显式校验一下 tbe 能不能 import，能 import 才说明环境真的注入成功
if python -c "import tbe" >/dev/null 2>&1; then
    echo ">>> Ascend env OK: tbe importable, PYTHONPATH=${PYTHONPATH%%:*}..."
else
    echo "[warn] tbe 仍然 import 不到，训练大概率会在 SetPrecisionMode 处挂"
    echo "[warn] 当前 PYTHONPATH=${PYTHONPATH}"
fi

# 业务阶段回归严格模式
set -e

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
