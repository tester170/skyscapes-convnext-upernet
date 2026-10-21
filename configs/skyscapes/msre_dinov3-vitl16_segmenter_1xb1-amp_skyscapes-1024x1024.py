# MsRE (Wang et al., IEEE TGRS 2026, github.com/woldier/MsRE) on DLR SkyScapes-Dense, 20 classes: frozen DINOv3
# ViT-L/16 (LVD-1689M) + MsRE modules after every block + Segmenter mask-transformer head; 33.0 of 336.1 M parameters
# are trained. Result (report, best iteration 34020 = epoch 63 of 540 iterations): val mIoU 35.70 %, kappa 0.757, OA 79.8 %.
# Weights: msre_dinov3_vitl16_skyscapes_ep63.pth, release v1.0 of github.com/tester170/skyscapes-convnext-upernet.
# Model and optimiser are the authors' (their configs/_base_/models/dinov3_vit-large_segmenter.py and
# configs/msre/msre_dinov3_vit-large_segmentor_1xb2-amp-40k_potsdam-512x512.py): AdamW 1e-4 / wd 0.05, head LR x10, no
# decay for norms, biases, tokens, fp16, warm-up 1500 iterations. Their code is fetched at commit d515366 into
# vendor/MsRE (env/fetch_vendor.sh) and loaded by skyscapes_mmseg/msre_compat.py, which (1) stands in for the module
# msre.models.decode_heads missing from their release and (2) replaces their eager attention by torch's
# scaled_dot_product_attention (same computation; the eager version does not fit 24 GB at a 1024 crop).
# Differences from the authors: crop 1024 at native resolution (theirs 512), batch 1 (2 does not fit 24 GB at 1024),
# start LR 0.125 x theirs (1.25e-5, head 1.25e-4): chosen by a direct comparison from the same initialisation on the same
# batches (400 iterations: val mIoU 2.7 / 4.6 / 7.8 / 9.5 % at 1 / 0.5 / 0.25 / 0.125 x); after the warm-up the LR is
# controlled by the validation mIoU instead of their 40k poly decay.
_base_ = ['./_base_/datasets/skyscapes_dense.py', './_base_/default_runtime.py']
custom_imports = dict(imports=['skyscapes_mmseg', 'skyscapes_mmseg.msre_compat'], allow_failed_imports=False)
data_preprocessor = dict(type='SegDataPreProcessor', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
                         bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=(1024, 1024))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(type='MsReDINOv3', freeze=True, image_size=1024, hidden_size=1024, intermediate_size=4096,
                  num_attention_heads=16, num_hidden_layers=24, num_register_tokens=4,
                  msre_config=dict(token_length=100, embed_dims=1024, num_layers=24, patch_size=16),
                  # DINOv3 ViT-L/16 LVD-1689M in the MsRE (Hugging Face) layout, tools/convert_dinov3_vit.py; training only
                  init_cfg=dict(type='Pretrained', checkpoint='checkpoints/dinov3-vitl16-pretrain-lvd1689m-hf.pth')),
    decode_head=dict(type='SegmenterMaskTransformerHead', in_channels=1024, channels=1024, num_classes=20, num_layers=2,
                     num_heads=16, embed_dims=1024, dropout_ratio=0.0,
                     loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

optim_wrapper = dict(
    type='AmpOptimWrapper', dtype='float16', loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=1.25e-5, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={'.ln': dict(decay_mult=0.0), '.bias': dict(decay_mult=0.0),
                                    '.cls_token': dict(decay_mult=0.0), '.pos_embed': dict(decay_mult=0.0),
                                    'head': dict(lr_mult=10.0), 'learnable_tokens': dict(decay_mult=0.0),
                                    'level_encoding': dict(decay_mult=0.0)}))
warmup_iters = 1500
param_scheduler = [dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters)]

# "epoch" of the experiments = 540 crops = 540 iterations at batch 1; validation every epoch (fp16), at most 150
# epochs; LR x0.5 after 5 validations without a gain of the validation-mIoU EMA (> 0.1 pp), stop after 20.
iters_per_epoch = 540
train_dataloader = dict(batch_size=1)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=150 * iters_per_epoch, val_interval=iters_per_epoch)
val_cfg = dict(type='ValLoop', fp16=True)
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5 * iters_per_epoch, max_keep_ckpts=1,
                    save_last=True, save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
custom_hooks = [
    dict(type='PlateauControlHook', monitor='mIoU', ema_alpha=0.5, min_delta=0.1, lr_patience=5, factor=0.5,
         cooldown=2, max_reductions=4, stop_patience=20, start_iter=warmup_iters),
]
