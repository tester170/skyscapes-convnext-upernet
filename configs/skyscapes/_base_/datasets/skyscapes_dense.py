# DLR SkyScapes-Dense, 20 classes, in its original folder layout (SS_Dense from the DLR archive):
#   data/SS_Dense/{train,val}/images/*.jpg          5616 x 3744 RGB, 13 cm
#   data/SS_Dense/{train,val}/labels/grayscale/*.png  codes 0..20 (0 = not labelled)
# reduce_zero_label=True turns the codes into train IDs 0..19 and 0 into the ignore index 255.
# Evaluation as in the report: whole frames at native resolution, sliding window 1024 / stride 768 (model test_cfg),
# mIoU over the 20 classes of both validation frames (the test split has no published labels).
dataset_type = 'BaseSegDataset'
data_root = 'data/SS_Dense'
metainfo = dict(
    classes=('low vegetation', 'paved road', 'non paved road', 'paved parking place', 'non paved parking place',
             'bikeways', 'sidewalks', 'entrance exit', 'danger area', 'lane-markings', 'building', 'car', 'trailer',
             'van', 'truck', 'long truck', 'bus', 'clutter', 'impervious surface', 'tree'),
    palette=[[244, 35, 232], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [152, 251, 152],
             [70, 130, 180], [107, 142, 35], [220, 220, 0], [255, 0, 0], [220, 20, 60], [125, 0, 142], [170, 200, 40],
             [200, 60, 100], [150, 18, 80], [81, 180, 81], [190, 241, 21], [11, 119, 32], [120, 240, 120], [70, 70, 70]])
crop_size = (1024, 1024)
# Training augmentation of the experiments with standard MMSeg transforms: random scale 0.75-1.5 at native resolution,
# 1024 crop, mild photometric jitter (one flight, one camera: softer than the MMSeg defaults), the 8 dihedral views.
# The experiments sampled the crops differently (every grid cell of the training frames once per epoch + 25 % crops
# anchored on rare classes, skyscapes_mmseg/data.py: SkyScapesTiles); here crops are uniform random.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RandomResize', scale=(5616, 3744), ratio_range=(0.75, 1.5), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='PhotoMetricDistortion', brightness_delta=16, contrast_range=(0.8, 1.2), saturation_range=(0.8, 1.2),
         hue_delta=5),
    dict(type='RandomDihedral'),                     # skyscapes_mmseg: rot90^k and optional flip (aerial nadir views)
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs'),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type=dataset_type, data_root=data_root, metainfo=metainfo, reduce_zero_label=True,
                 img_suffix='.jpg', seg_map_suffix='.png',
                 data_prefix=dict(img_path='train/images', seg_map_path='train/labels/grayscale'),
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root, metainfo=metainfo, reduce_zero_label=True,
                 img_suffix='.jpg', seg_map_suffix='.png',
                 data_prefix=dict(img_path='val/images', seg_map_path='val/labels/grayscale'),
                 pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator
