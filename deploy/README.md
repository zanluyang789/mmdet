# mmdet 部署目录

这一层独立于 `integration/`，专门给离线 / 平台两种部署都能复用：

- `infer_large_image.py`：滑窗大图推理（GeoTIFF 优先），跨瓦片 NMS 合并
- `postprocess.py`：检测 JSON → SHP / GeoJSON
- `send_kafka_msg.py` + `MessageClient/`：进度回报 Kafka，依赖 `KAFKA_SERVER_IP_PORT` / `KAFKA_TOPIC` / `KAFKA_TASK_ID` 三个环境变量。被 `integration/infer_runner.py` 的 `_kafka_send` 自动设置，所以在调度场景下不用手工管。

## 离线使用示例

```bash
# 大图推理（NPU 镜像里跑）
export INTEGRATION_DEVICE=npu
export PATH_MODEL_RESOURCE=/path/to/models   # 内含 detect.om 或 detect.pth
export DATA_INPUT_DIR1=/path/to/big.tif
export DATA_OUTPUT_DIR=/path/to/out
python deploy/infer_large_image.py \
    --tile 800 --stride 600 --score-thr 0.3

# 单独把已有 dets JSON 转 SHP
python deploy/postprocess.py \
    --dets out/big_dets.json \
    --ref  /path/to/big.tif \
    --out  out/big_dets.shp
```
