# mmdet 系统集成版

> mmdetection 3.x 之上的"统一入口"层，对接平台调度（`run.py train` / `predict.py infer --custom-config='task'`），同时支持 **GPU 训练 / GPU 推理 / NPU 训练 / NPU 推理**。

设计与 [mmseg 集成版](https://192.168.1.166:9500/template/mmseg) 一一对齐：相同的 task.conf 字段、相同的目录结构、相同的 Docker 入口约定。

## 最短上手

```bash
# 1) 准备数据（COCO 格式即可）
python tools/prepare_data.py --src /data/raw/your_dataset --dst /data/your_dataset_split

# 2) 改一下 configs/task.conf 里的路径与类别，然后训练
python run.py train

# 3) 推理（按需把 load_model_path 改成 .pth / .onnx / .om）
python predict.py infer --custom-config='task'
```

详细字段、Docker 用法、ATC / OM 完整路径见 [INTEGRATION_README.md](./INTEGRATION_README.md)。

## 设备 / 后端

| | GPU (CUDA) | NPU (Ascend) |
|---|---|---|
| 训练 | `INTEGRATION_DEVICE=cuda python run.py train` | `INTEGRATION_DEVICE=npu python run.py train` |
| 推理 - pth | ✓ | ✓ (`torch_npu`) |
| 推理 - onnx | ✓ | x（建议转 om） |
| 推理 - om | x | ✓ (pyACL) |

切换全自动，**不需要改任何代码**。系统下发哪种模型后缀就跑哪种栈。

## 主要文件

* `run.py` / `predict.py`：训练 / 推理入口（薄壳）
* `integration/`：task.conf 解析、设备探测、Runner、自定义 dataset / hook
* `configs/retinanet_r50_fpn_object.py`：默认 mmdet config（单阶段，ATC -> OM 友好）
* `tools/`：数据准备、ONNX 导出、ONNX 验证、OM 单图推理
* `deploy/`：大图滑窗推理、矢量化、Kafka 进度上报
* `docker/Dockerfile.gpu` / `docker/Dockerfile.npu`：训练+推理一体镜像
