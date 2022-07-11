_base_ = [
    '../_base_/models/efficientnet_b4.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b4', 
                  init_cfg=dict(
                      type='Pretrained',
                      checkpoint='https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty_8xb32-aa-advprop_in1k_20220119-38c2238c.pth',
                      prefix='backbone')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1792,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))

# dataset settings
dataset_type = 'CMVDataset'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=380,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CenterCrop',
        crop_size=380,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu = 16,
    workers_per_gpu=2,

    train = dict(
        type='CMVDataset',
        data_prefix='CMV/2nd_stage/Images8',
        ann_file = 'CMV/2nd_stage/Database8/train.txt',
        pipeline=train_pipeline),

    val = dict(
        type='CMVDataset',
        data_prefix = 'CMV/2nd_stage/Images8',
        ann_file = 'CMV/2nd_stage/Database8/val.txt',
        pipeline=test_pipeline),

    test = dict(
        type='CMVDataset',
        data_prefix = 'CMV/2nd_stage/Images8',
        ann_file = 'CMV/2nd_stage/Database8/val.txt',
        pipeline=test_pipeline)
)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', step=[16,22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
log_config = dict(interval=100)
checkpoint_config = dict(interval=1)
work_dir = 'notebook/mmclassification/work_dirs/cmvdataset'
workflow = [('train', 1), ('val', 1)]