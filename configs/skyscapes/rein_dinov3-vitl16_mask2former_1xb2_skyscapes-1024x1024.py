# Rein (Wei et al., CVPR 2024, github.com/w1oves/Rein) on DLR SkyScapes-Dense, 20 classes: frozen DINOv3 ViT-L/16
# (LVD-1689M) + LoRA Reins after every block + the authors' Mask2Former head; 23.6 of 326.7 M parameters are trained.
# Result (report, best iteration 8100 = epoch 30 of 270 iterations): val mIoU 41.17 %, kappa 0.774, OA 81.2 %.
# Weights: rein_dinov3_vitl16_m2f_skyscapes_ep30.pth, release v1.0 of github.com/tester170/skyscapes-convnext-upernet.
# Rein parts, head and optimiser are the authors' (configs/_base_/models/rein_dinov2_mask2former.py, copied except
# num_classes; configs/dinov2/rein_dinov2_mask2former_1024x1024_bs4x2.py: AdamW 6e-5 / wd 0.05 / eps 1e-8,
# PEFTOptimWrapperConstructor). Their code is fetched at commit dc06342 (branch train) into vendor/Rein
# (env/fetch_vendor.sh) and loaded by skyscapes_mmseg/rein_compat.py. Rein has no DINOv3 version: ReinsDINOv3
# (skyscapes_mmseg/dinov3_backbones.py) runs the official DINOv3 ViT-L/16 (vendor/dinov3) with the authors' LoRAReins
# after every block, outputs of blocks 8/12/16/24 resized to strides 4/8/16/32, as their DINOv2 backbone; DINOv3's class
# and 4 storage tokens are left untouched by Rein, as the class token of DINOv2 is. The frozen ViT runs in fp16 autocast,
# the Mask2Former head in fp32 (the authors: all fp32; MMCV 2.1 multi-scale deformable attention gave NaN in fp16).
# Differences: crop 1024 at native resolution, batch 2, warm-up 1500 iterations, then the LR is controlled by the
# validation mIoU instead of the authors' 40k poly decay.
_base_ = ['./_base_/datasets/skyscapes_dense.py', './_base_/default_runtime.py']
custom_imports = dict(imports=['skyscapes_mmseg', 'skyscapes_mmseg.rein_compat'], allow_failed_imports=False)
num_classes = 20
data_preprocessor = dict(type='SegDataPreProcessor', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
                         bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=(1024, 1024))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(type='ReinsDINOv3', backbone_dtype='fp16',
                  # official-format DINOv3 ViT-L/16 LVD-1689M (tools/convert_dinov3_vit.py); loaded for training only
                  checkpoint='checkpoints/dinov3_vitl16_lvd1689m_official.pth',
                  reins_config=dict(type='LoRAReins', token_length=100, embed_dims=1024, num_layers=24, patch_size=16,
                                    link_token_to_query=True, lora_dim=16)),
    decode_head=dict(
        type='ReinMask2FormerHead', replace_query_feat=True, in_channels=[1024, 1024, 1024, 1024], strides=[4, 8, 16, 32],
        feat_channels=256, out_channels=256, num_classes=num_classes, num_queries=100, num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder', num_outs=3, norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(num_layers=6,
                         layer_cfg=dict(self_attn_cfg=dict(embed_dims=256, num_heads=8, num_levels=3, num_points=4,
                                                           im2col_step=64, dropout=0.0, batch_first=True, norm_cfg=None,
                                                           init_cfg=None),
                                        ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, num_fcs=2, ffn_drop=0.0,
                                                     act_cfg=dict(type='ReLU', inplace=True))),
                         init_cfg=None),
            positional_encoding=dict(num_feats=128, normalize=True), init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True, num_layers=9,
            layer_cfg=dict(self_attn_cfg=dict(embed_dims=256, num_heads=8, attn_drop=0.0, proj_drop=0.0,
                                              dropout_layer=None, batch_first=True),
                           cross_attn_cfg=dict(embed_dims=256, num_heads=8, attn_drop=0.0, proj_drop=0.0,
                                               dropout_layer=None, batch_first=True),
                           ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, num_fcs=2,
                                        act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0.0, dropout_layer=None,
                                        add_identity=True)),
            init_cfg=None),
        loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0, reduction='mean',
                      class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=5.0),
        loss_dice=dict(type='mmdet.DiceLoss', use_sigmoid=True, activate=True, reduction='mean', naive_dice=True, eps=1.0,
                       loss_weight=5.0),
        train_cfg=dict(num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75,
                       assigner=dict(type='mmdet.HungarianAssigner',
                                     match_costs=[dict(type='mmdet.ClassificationCost', weight=2.0),
                                                  dict(type='mmdet.CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                                                  dict(type='mmdet.DiceCost', weight=5.0, pred_act=True, eps=1.0)]),
                       sampler=dict(type='mmdet.MaskPseudoSampler'))),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    constructor='PEFTOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=6e-5, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)),
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.0), 'query_embed': embed_multi, 'level_embed': embed_multi,
                                    'learnable_tokens': embed_multi, 'reins.scale': embed_multi},
                       norm_decay_mult=0.0))
warmup_iters = 1500
param_scheduler = [dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters)]

# "epoch" of the experiments = 540 crops = 270 iterations at batch 2; validation every epoch (no outer fp16: it would put
# the Mask2Former head into fp16), at most 150 epochs; LR x0.5 after 5 validations without a gain of the validation-mIoU
# EMA (> 0.1 pp), stop after 20 such validations.
iters_per_epoch = 270
train_dataloader = dict(batch_size=2)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=150 * iters_per_epoch, val_interval=iters_per_epoch)
val_cfg = dict(type='ValLoop')
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
