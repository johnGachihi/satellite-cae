import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


default_scope = "mmseg"

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

### Wandb
wandb_kwargs = dict(
    project='mae-sen1floods11-finetuning',
    entity='johngachihi-carnegie-mellon-university',
    #  name='your_run_name',
    resume='allow',
    dir='output'
)

vis_backends = [
    dict(type='LocalVisBackend'), 
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(wandb_kwargs)
    )
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# custom_hooks = [dict(type='WandbLoggerHook', interval=50)]



log_level = "INFO"
load_from = None
resume_from = None

custom_imports = dict(imports=["geospatial_fm"])


### Configs
# Data
# TO BE DEFINED BY USER: Data root to sen1floods11 downloaded dataset
data_root = "/home/ubuntu/satellite-cae/data/sen1floods11"

dataset_type = "GeospatialDataset"
num_classes = 2
num_frames = 1
img_size = 224
num_workers = 2
samples_per_gpu = 5
CLASSES = (0, 1)

img_norm_cfg = dict(
    means=[0.14245495, 0.13921481, 0.12434631, 0.5, 0.5, 0.5, 0.5, 0.31420089, 0.5, 0.5, 0.20743526, 0.12046503],
    stds=[0.04036231, 0.04186983, 0.05267646, 0.5, 0.5, 0.5, 0.5, 0.0822221, 0.5, 0.5, 0.06834774, 0.05294205],
)

bands = [1, 2, 3, 8, 11, 12]
tile_size = img_size
orig_nsize = 512
crop_size = (tile_size, tile_size)

img_dir = data_root + "v1.1/data/flood_events/HandLabeled/S2Hand"
ann_dir = data_root + "v1.1/data/flood_events/HandLabeled/LabelHand"
img_suffix = f"_S2Hand.tif"
seg_map_suffix = f"_LabelHand.tif"

splits = {
    "train": "data_splits/sen1floods11/train_split.txt",
    "val": "data_splits/sen1floods11/val_split.txt",
    "test": "data_splits/sen1floods11/test_split.txt",
}
splits = {k: os.path.abspath(v) for (k, v) in splits.items()}

ignore_index = 2
label_nodata = -1
image_nodata = -9999
image_nodata_replace = 0
constant = 0.0001

# Model
# TO BE DEFINED BY USER: path to pretrained backbone weights
pretrained_weights_path = "<path to pretrained weights>"
num_layers = 12
patch_size = 16
embed_dim = 768
num_heads = 12
tubelet_size = 1

# TRAINING
epochs = 100
eval_epoch_interval = 5

# TO BE DEFINED BY USER: Save directory
experiment = "<experiment name>"
project_dir = "output"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir

# Pipelines
train_pipeline = [
    dict(
        type="LoadGeospatialImageFromFile",
        to_float32=False,
        nodata=image_nodata,
        nodata_replace=image_nodata_replace,
    ),
    dict(
        type="LoadGeospatialAnnotations",
        reduce_zero_label=False,
        nodata=label_nodata,
        nodata_replace=ignore_index,
    ),
    dict(type="BandsExtract", bands=bands),
    dict(type="ConstantMultiply", constant=constant),
    dict(type="RandomFlip", prob=0.5),
    dict(type="ToTensor_", keys=["img", "gt_semantic_seg"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(type="TorchRandomCrop", crop_size=crop_size),
    # dict(
    #     type="Reshape",
    #     keys=["img"],
    #     new_shape=(len(bands), num_frames, tile_size, tile_size),
    # ),
    dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(1, tile_size, tile_size)),
    dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),
    dict(type="PackSegInputs_"),
]

val_pipeline = [
    dict(
        type="LoadGeospatialImageFromFile",
        to_float32=False,
        nodata=image_nodata,
        nodata_replace=image_nodata_replace,
    ),
    dict(
        type="LoadGeospatialAnnotations",
        reduce_zero_label=False,
        nodata=label_nodata,
        nodata_replace=ignore_index,
    ),
    dict(type="BandsExtract", bands=bands),
    dict(type="ConstantMultiply", constant=constant),
    dict(type="ToTensor_", keys=["img", "gt_semantic_seg"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
#     dict(type="TorchRandomCrop", crop_size=crop_size),
#    dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(1, tile_size, tile_size)),
    dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),
    dict(type="PackSegInputs_", meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]


test_pipeline = [
    dict(
        type="LoadGeospatialImageFromFile",
        to_float32=False,
        nodata=image_nodata,
        nodata_replace=image_nodata_replace,
    ),
    dict(type="BandsExtract", bands=bands),
    dict(type="ConstantMultiply", constant=constant),
    dict(type="ToTensor_", keys=["img"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    # dict(
    #     type="Reshape",
    #     keys=["img"],
    #     new_shape=(len(bands), num_frames, -1, -1),
    #     look_up={"2": 1, "3": 2},
    # ),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(type="PackSegInputs_"),
]

train_dataloader = dict(
    batch_size=samples_per_gpu,  # Number of samples per GPU
    num_workers=num_workers,     # Number of workers per GPU
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),  # Default sampler with shuffling for training
    dataset=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        data_prefix=dict(     # Use data_prefix to specify directories
            img_path='train/s2',
            seg_map_path='train/labelhand',
        ),
        # img_dir=img_dir,
        # ann_dir=ann_dir,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=train_pipeline,  # Use your defined train pipeline
        ignore_index=ignore_index,
        # split=splits["train"],    # Training split file path
    )
)

val_dataloader = dict(
    batch_size=1,  # Validation typically uses batch size of 1 for evaluation
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),  # No shuffling for validation/testing
    dataset=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        data_prefix=dict(     # Use data_prefix to specify directories
            img_path='val/s2',
            seg_map_path='val/labelhand',
        ),
        # img_dir=img_dir,
        # ann_dir=ann_dir,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=val_pipeline,   # Use your defined test pipeline for validation
        ignore_index=ignore_index,
        # split=splits["val"],      # Validation split file path
#        gt_seg_map_loader_cfg=dict(nodata=label_nodata, nodata_replace=ignore_index),
        metainfo=dict(classes=CLASSES)
    )
)

test_dataloader = dict(
    batch_size=1,  # Testing typically uses batch size of 1 for evaluation
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),  # No shuffling for validation/testing
    dataset=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        data_prefix=dict(     # Use data_prefix to specify directories
            img_path='test/s2',
            seg_map_path='test/labelhand',
        ),
        # img_dir=img_dir,
        # ann_dir=ann_dir,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,   # Use your defined test pipeline for testing
        ignore_index=ignore_index,
        # split=splits["test"],     # Testing split file path
        gt_seg_map_loader_cfg=dict(nodata=label_nodata, nodata_replace=ignore_index),
    )
)

val_evaluator = dict(
    type='IoUMetric',  # Metric type (e.g., IoU)
    iou_metrics=['mIoU'],  # Specify mIoU as the metric
    pre_eval=True  # Perform pre-evaluation step
)

test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,  # Save checkpoint every epoch
        save_best='mIoU',  # Save the best model based on mIoU
        rule='greater'  # Higher mIoU is better
    )
)


# Training
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# runner = dict(type="EpochBasedRunner", max_epochs=epochs)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-5,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    ),
    clip_grad=None
)
# optimizer = dict(
#     type="AdamW",
#     lr=1.5e-5,
#     weight_decay=0.05,
#     betas=(0.9, 0.999),
# )
# optimizer_config = dict(grad_clip=None)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        by_epoch=False)
]
# lr_config = dict(
#     policy="poly",
#    warmup="linear",
#    warmup_iters=1500,
#    warmup_ratio=1e-6,
#    power=1.0,
#    min_lr=0.0,
#    by_epoch=False,
#)

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=True),
        dict(type="TensorboardLoggerHook", by_epoch=True),
    ],
)

checkpoint_config = dict(by_epoch=True, interval=10, out_dir=save_path)

workflow = [("train", 1), ("val", 1)]

norm_cfg = dict(type="BN", requires_grad=True)

ce_weights = [0.3, 0.7]

model = dict(
    type="EncoderDecoder_",
    data_preprocessor=dict(
        type='SegDataPreProcessor_',
        size=(img_size, img_size),
        size_divisor=None
    ),
    backbone=dict(
        type="VisionTransformer_",
        # pretrained=pretrained_weights_path,
        img_size=img_size,
        patch_size=patch_size,
        # num_frames=num_frames,
        # tubelet_size=1,
        in_chans=len(bands),
        # embed_dims=embed_dim,
        # feedforward_channels=embed_dim,
        # num_layers=num_layers,
        # num_heads=num_heads,
        # mlp_ratio=4.0,
        # norm_pix_loss=False,
    ),
    neck=dict(
        type="ConvTransformerTokensToEmbeddingNeck",
        embed_dim=num_frames * embed_dim,
        output_embed_dim=embed_dim,
        drop_cls_token=True,
        Hp=img_size // patch_size,
        Wp=img_size // patch_size,
    ),
    decode_head=dict(
        num_classes=num_classes,
        in_channels=embed_dim,
        type="FCNHead_",
        in_index=-1,
        ignore_index=ignore_index,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss_",
            use_sigmoid=False,
            loss_weight=1,
            class_weight=ce_weights,
            avg_non_ignore=True,
            reduction="sum"
        ),
    ),
    # auxiliary_head=dict(
    #     num_classes=num_classes,
    #     in_channels=embed_dim,
    #     ignore_index=ignore_index,
    #     type="FCNHead",
    #     in_index=-1,
    #     channels=256,
    #     num_convs=2,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type="CrossEntropyLoss",
    #         use_sigmoid=False,
    #         loss_weight=1,
    #         class_weight=ce_weights,
    #         avg_non_ignore=True,
    #     ),
    # ),
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        stride=(int(tile_size / 2), int(tile_size / 2)),
        crop_size=(tile_size, tile_size),
    ),
)
