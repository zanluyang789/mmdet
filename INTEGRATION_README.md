# mmdet 系统集成总览（GPU/NPU 训练 + GPU/NPU 推理）

按"地灾项目-算法集成"文档实现的统一入口，与 mmseg 集成层一一对齐。**离线流程一行没删**，只是在上面套了一层 task.conf 驱动的系统调度入口。

## 全景图

```
mmdet/
├── run.py                              # 系统训练入口  (python run.py train)
├── predict.py                          # 系统推理入口  (python predict.py infer --custom-config='task')
│
├── configs/
│   ├── task.conf                       # 系统下发的【训练】参数文件 (位置固定)
│   └── retinanet_r50_fpn_object.py     # mmdet base config (单阶段 RetinaNet，单类默认)
│
├── clie_lib/
│   └── configs/
│       └── task.conf                   # 系统下发的【推理】参数文件 (位置固定)
│
├── integration/                        # 系统集成层
│   ├── conf_parser.py                  # task.conf 解析器 (带类型推断)
│   ├── device_utils.py                 # GPU/NPU 自动探测
│   ├── data_yaml.py                    # YOLO 风格 data.yaml 解析
│   ├── filelist_dataset.py             # FileListCocoDataset (img_list + COCO json)
│   ├── config_builder.py               # task.conf 参数注入 mmdet Config
│   ├── train_runner.py                 # 训练核心
│   ├── infer_runner.py                 # 推理核心 (pth/onnx/om 三栈统一)
│   └── hooks.py                        # CkptSyncHook
│
├── tools/                              # 离线工具
│   ├── prepare_data.py                 # COCO 数据划分
│   ├── export_onnx.py                  # mmdet pth -> ONNX (定长输出)
│   ├── verify_onnx.py                  # PyTorch vs ONNX 一致性检查
│   └── infer_om.py                     # NPU 单图 OM 推理 + 可视化
│
├── deploy/                             # 部署目录（被 integration.infer_runner 复用）
│   ├── infer_large_image.py            # 滑窗大图检测 + 跨瓦片 NMS
│   ├── postprocess.py                  # 检测 JSON -> SHP / GeoJSON
│   ├── send_kafka_msg.py               # Kafka 进度上报
│   ├── MessageClient/
│   └── README.md
│
├── docker/                             # 系统集成镜像
│   ├── Dockerfile.gpu                  # GPU 训练 + GPU 推理 一体镜像
│   ├── Dockerfile.npu                  # NPU 训练 + NPU 推理 一体镜像
│   ├── entrypoint.sh                   # 根据 train|infer 分发
│   └── export_image.sh                 # docker save -> tar.gz
│
├── INTEGRATION_README.md               # 本文件
└── README.md                           # 简短入门
```

## 两条流程的关系

| 流程 | 入口 | 配置来源 | 适用场景 |
|---|---|---|---|
| 离线 | `tools/train.py`（用户自己写）、`tools/export_onnx.py`、`deploy/infer_large_image.py` | 命令行参数 / 环境变量 | 算法工程师本地调试 |
| 系统集成 | `run.py train`、`predict.py infer --custom-config='task'` | `configs/task.conf`、`clie_lib/configs/task.conf` | 平台调度 |

两套流程互不影响，**底层模型 config、滑窗推理函数、矢量化函数 100% 共享**。

## 训练（GPU / NPU 通用）

### 1. 系统填好 task.conf

平台会在容器启动前把参数写入 `configs/task.conf`，位置固定。常用字段：

```ini
env = 'mmdet-detect-3266'
work_dir = '/data/train/model/train-3266'
checkpoint_path = '/data/train/model/train-3266/pth'
log_path = '/data/train/model/train-3266/log'
tensorboard_log_path = '/data/train/tensorboard/train-3266'
use_tensorboard_scalar = True
use_tensorboard_image = True
pretrained = '/data/train/common_pth'
retrain_pth_url = ''                      # 再训练时给 .pth 绝对路径

# 数据：两种方式二选一
# ---- 方式 A：use_filelist + 图像 list + COCO ann ----
use_filelist = True
train_img_list = '.../config/train_img_list.txt'
train_ann_file = '.../config/train.json'
val_img_list   = '.../config/val_img_list.txt'
val_ann_file   = '.../config/val.json'

# ---- 方式 B：YOLO 风格 data.yaml ----
# data_file = '.../config/data.yaml'

mean_file = '.../config/mean_value.txt'
std_file  = '.../config/std_value.txt'
num_classes = 20
classes_name = ('plane','ship','vehicle','...')
palette = None             # None 时自动按 seed=42 随机生成

batch_size = 2
max_iters = 20000
val_interval = 2000
gpu_num = 1
```

> 文档里还可能下发其它字段，整套 PDF 通用表都已被 `integration/conf_parser.py` + `integration/config_builder.py` 处理，对不上的字段会被忽略。

### 2. 单卡

```bash
python run.py train
```

设备自动探测：装了 `torch_npu` 走 NPU，否则走 CUDA，再否则 CPU。也可强制指定：

```bash
INTEGRATION_DEVICE=npu  python run.py train
INTEGRATION_DEVICE=cuda python run.py train
```

### 3. 多卡（按文档原文）

```bash
python --nnodes=${nnodes} --node_rank=${node_rank} \
       --nproc_per_node=${nproc_per_node} --master_addr=${master_addr} \
       --master_port=${master_port} run.py train
```

平台会替换变量，实际等价于 `python -m torch.distributed.run …`。`run.py` 探测到 `RANK` 环境变量后会把 `cfg.launcher` 切成 `'pytorch'`。NPU 上 dist backend 自动切到 `hccl`，norm_cfg 自动从 `SyncBN` 退化到 `BN`。

## 推理（GPU pth / NPU om / ONNX 三栈）

### 1. 系统填好 clie_lib/configs/task.conf

```ini
env = 'mmdet-infer-3266'
load_model_path = '/data/train/model/train-3266/pth/best_coco_bbox_mAP.pth'
# 或:
# load_model_path = '/data/.../detect.onnx'
# load_model_path = '/data/.../detect.om'

img_list_file = '/data/.../infer/img_list.txt'
output_root   = '/share/.../output/predict/detect_3266'
color_table_file = '/data/.../config/color_table.txt'   # 0,car,255,0,0 每行一类

bootstrap_servers = '192.168.1.166:12007'
topic = 'ib-theme.algorithm_callback_topic'
projectId = 288
taskId = 5224

use_color_out = True        # 输出画框可视化 PNG
use_shapefile_out = True    # GeoTIFF 输入时输出 SHP

# 可选
num_classes = 20
classes_name = ('plane','ship','vehicle','...')
score_thr = 0.3             # 置信度阈值
infer_size = (800, 800)     # ONNX/OM 输入 HxW；pth 自动决定
```

### 2. 启动

```bash
python predict.py infer --custom-config='task'
```

`--custom-config='task'` 是文档约定，对应 `clie_lib/configs/task.conf`。也可以传绝对路径：

```bash
python predict.py infer --custom-config=/path/to/whatever.conf
```

### 3. 模型后缀决定后端

| 后缀 | runner | 设备 | 说明 |
|---|---|---|---|
| `.pth` | `PthRunner` (mmdet.apis) | cuda / npu | GPU 推理（直接吃训练产物） |
| `.onnx` | `ONNXRunner` (onnxruntime) | cpu / cuda | 跨平台调试 |
| `.om` | `OMRunner` (pyACL) | npu | 昇腾 NPU 推理 |

切换全自动，**不需要改任何代码**。系统下发哪种 `load_model_path` 就跑哪种栈。

### 4. 输出

每张图固定输出：

| 文件 | 说明 | 触发条件 |
|---|---|---|
| `<stem>_dets.json` | COCO 风格框列表 `[{bbox:[x,y,w,h], score, category_id, category}]` | 总是 |
| `<stem>_vis.png` | 画框可视化 | `use_color_out=True` |
| `<stem>_dets.shp` | 多边形=外接矩形，带 class/score | `use_shapefile_out=True` 且输入是 GeoTIFF |

## Docker 镜像

文档要求："将代码运行环境打包成一个 docker 镜像"，提供两个：

### GPU 镜像

```bash
cd C:\Users\zanly\Desktop\mmdet
docker build -f docker/Dockerfile.gpu -t mmdet-gpu:v1 .

# 训练
docker run --rm --gpus all \
    -v /data/train:/data/train \
    -v $PWD/configs/task.conf:/workspace/mmdet/configs/task.conf:ro \
    mmdet-gpu:v1 train

# 推理
docker run --rm --gpus all \
    -v /data/predict:/data/predict \
    -v /share:/share \
    -v $PWD/clie_lib/configs/task.conf:/workspace/mmdet/clie_lib/configs/task.conf:ro \
    mmdet-gpu:v1 infer
```

### NPU 镜像

```bash
docker build -f docker/Dockerfile.npu -t mmdet-npu:v1 .

# 训练
docker run --rm \
    --device=/dev/davinci0 --device=/dev/davinci_manager \
    --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /data/train:/data/train \
    -v $PWD/configs/task.conf:/workspace/mmdet/configs/task.conf:ro \
    mmdet-npu:v1 train

# 推理（把 task.conf 里的 load_model_path 换成 .om 即可）
docker run --rm \
    --device=/dev/davinci0 ... \
    -v $PWD/clie_lib/configs/task.conf:/workspace/mmdet/clie_lib/configs/task.conf:ro \
    mmdet-npu:v1 infer
```

### 导出 docker save 文件

```bash
bash docker/export_image.sh gpu     # -> dist/mmdet-gpu-v1.tar.gz
bash docker/export_image.sh npu     # -> dist/mmdet-npu-v1.tar.gz
```

## ATC / OM 完整路径

```bash
# 1) PyTorch -> ONNX
python tools/export_onnx.py \
    --config configs/retinanet_r50_fpn_object.py \
    --ckpt work_dirs/best_coco_bbox_mAP.pth \
    --out detect.onnx \
    --img-h 800 --img-w 800 --max-dets 100

# 2) 验证一致性（NPU 服务器之前在 GPU 上跑）
python tools/verify_onnx.py \
    --config configs/retinanet_r50_fpn_object.py \
    --ckpt work_dirs/best_coco_bbox_mAP.pth \
    --onnx detect.onnx \
    --img some_test.jpg

# 3) ATC 转 OM（NPU 服务器）
atc --model=detect.onnx --output=detect \
    --soc_version=Ascend910B --framework=5 \
    --input_format=NCHW --input_shape="input:1,3,800,800"

# 4) NPU 单图推理验证
python tools/infer_om.py --om detect.om --img some_test.jpg \
    --infer-h 800 --infer-w 800 --color-table color_table.txt

# 5) 把 .om 配进推理 task.conf：load_model_path = '.../detect.om'
#    然后  python predict.py infer --custom-config='task'
```

**默认 base config 已是 RetinaNet（单阶段，ATC -> OM 友好）。** 如果想换检测器：

- 换其它单阶段（FCOS / YOLOX / ATSS）：在 `configs/` 下加一份 mmdet config，把 `task.conf` 里 `base_config` 指向新 config 即可，集成层自动适配。
- 换回两阶段（Faster R-CNN / Cascade R-CNN）：能跑 GPU pth 推理，但 ATC 转 OM 时 RPN+RoI 二段 NMS 算子可能报错或精度下降；这种场景建议改用 [mmdeploy](https://github.com/open-mmlab/mmdeploy) 替代 `tools/export_onnx.py`，它内置了对 NPU/ATC 友好的算子映射。

## 设备 / 后端兼容矩阵

|  | GPU (CUDA) | NPU (Ascend) |
|---|---|---|
| 训练 | `INTEGRATION_DEVICE=cuda python run.py train`（默认） | `INTEGRATION_DEVICE=npu python run.py train` |
| 训练 norm_cfg | SyncBN | 自动降级 BN |
| 训练 dist backend | nccl | hccl |
| 推理 - pth | ✓ | ✓（torch_npu） |
| 推理 - onnx | ✓ | x（ORT 没 NPU EP，建议 om） |
| 推理 - om | x | ✓（pyACL） |

## 调试小贴士

* `python -m integration.conf_parser configs/task.conf` 单独看 task.conf 解析效果
* `python -m integration.device_utils` 看当前设备探测结果
* `python -m integration.data_yaml /path/to/data.yaml` 看 yaml 解析效果
* `python -m integration.config_builder configs/retinanet_r50_fpn_object.py configs/task.conf` 打印注入后的完整 Config
* CPU 模式跑通最快：`INTEGRATION_DEVICE=cpu python run.py train`（用一两个样本 + max_iters=10 测路径有没有错）
