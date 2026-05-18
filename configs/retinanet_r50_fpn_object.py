# =====================================================================
# RetinaNet R50-FPN 目标检测基础 config
# ---------------------------------------------------------------------
# 沿用 mmdet 官方 retinanet_r50-fpn_1x_coco.py 的骨架，做了以下改动：
#   1. data_root / ann_file 改成本项目占位路径，部署时由 task.conf 覆盖
#   2. num_classes=1（单类默认；task.conf 里 num_classes/classes_name 会覆盖）
#   3. data_preprocessor.bgr_to_rgb=True 保持和 ONNX 导出一致
#   4. 训练长度切到 iter-based（与 mmseg 集成层风格统一），由 task.conf 注入
#   5. 默认 SyncBN，NPU 上 config_builder 会自动降级 BN
#
# 单阶段 + Focal Loss + 单次 NMS，对 ATC -> OM 比 Faster R-CNN 友好得多。
# =====================================================================

# ============ 数据集 ============
dataset_type = 'CocoDataset'
data_root = '/data/coco/'

# 占位 metainfo —— task.conf 会用 classes_name / palette 覆盖
metainfo = dict(
    classes=tuple(),
    palette=tuple(),
)

backend_args = None

# 训练/验证 pipeline
img_scale = (1333, 800)  # mmdet 经典尺寸 (W, H)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# ============ 模型 ============
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='RetinaNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(type='PseudoSampler'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        # score_thr=0.001 是 RetinaNet 论文 eval 的标准值，比 mmdet 默认的 0.05 低。
        # 选 0.001 而非 0.05 的原因：RetinaHead 分类 bias 初始化为 ~ -4.6（Focal
        # Loss 标准 trick），前几千 iter 所有 anchor 的 sigmoid 输出都被压在
        # 0.01-0.05 之间。score_thr=0.05 会把整个 val set 过滤成空，让 mmdet
        # coco_metric 打 ERROR 并跳过保存 best ckpt。下游推理 task.conf 里
        # score_thr（PthRunner 默认 0.3）会再过滤一遍，所以这里给低点不影响线上。
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# ============ 训练策略 ============
# RetinaNet 官方 lr=0.01（Faster R-CNN 是 0.02）。
# 单卡时一般再降到 0.0025 量级，但这里保持 mmdet 默认，多卡时直接用。
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=None)

# IterBased：与 mmseg 集成层风格一致；task.conf 里 max_iters / val_interval 会覆盖
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20000,
        by_epoch=False,
        milestones=[14000, 18000],
        gamma=0.1)
]

# ============ Hook ============
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=2000,
        save_best='coco/bbox_mAP', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

randomness = dict(seed=42)
