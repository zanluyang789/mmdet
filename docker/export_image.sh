#!/usr/bin/env bash
# 把 build 完的镜像导出成可分发的 tar 包（docker save）
# 用法：
#   bash docker/export_image.sh gpu     # 导出 mmdet-gpu:v1
#   bash docker/export_image.sh npu     # 导出 mmdet-npu:v1
#
# 输出在仓库根目录下 dist/ 里。

set -e
TARGET="${1:-gpu}"
cd "$(dirname "$0")/.."

mkdir -p dist

case "$TARGET" in
    gpu)
        IMG=mmdet-gpu:v1
        OUT=dist/mmdet-gpu-v1.tar
        ;;
    npu)
        IMG=mmdet-npu:v1
        OUT=dist/mmdet-npu-v1.tar
        ;;
    *)
        echo "用法: bash docker/export_image.sh [gpu|npu]"
        exit 2
        ;;
esac

echo "导出 $IMG -> $OUT"
docker save -o "$OUT" "$IMG"
gzip -f "$OUT"
echo "完成: ${OUT}.gz  ($(du -h ${OUT}.gz | cut -f1))"
