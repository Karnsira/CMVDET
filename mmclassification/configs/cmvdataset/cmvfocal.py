_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

# Model config
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        strides=(1, 1, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(3, ),
        style='pytorch', 
        avg_down=False,
        frozen_stages=2, 
        drop_path_rate=0.5,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',
            prefix='backbone')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=512,
        loss=dict(type='FocalLoss', loss_weight=1.0),
        topk=(1, 2)))

# Dataset config
dataset_type = 'CMVDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu = 32,
    workers_per_gpu=2,

    train = dict(
        type='CMVDataset',
        data_prefix='CMV/2nd_stage/Images4',
        ann_file = 'CMV/2nd_stage/Database4/train.txt',
        pipeline=train_pipeline),

    val = dict(
        type='CMVDataset',
        data_prefix = 'CMV/2nd_stage/Images4',
        ann_file = 'CMV/2nd_stage/Database4/val.txt',
        pipeline=test_pipeline),

    test = dict(
        type='CMVDataset',
        data_prefix = 'CMV/2nd_stage/Images4',
        ann_file = 'CMV/2nd_stage/Database4/val.txt',
        pipeline=test_pipeline)
)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', step=[60,90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
log_config = dict(interval=100)
checkpoint_config = dict(interval=10)
work_dir = 'notebook/mmclassification/work_dirs/cmvdataset'
workflow = [('train', 1), ('val', 1)]