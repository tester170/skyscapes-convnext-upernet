# Runtime shared by the SkyScapes configs. skyscapes_mmseg (this repository) registers the modules the configs use
# beyond MMSegmentation 1.2.2: the DINOv3 ConvNeXt backbone (official facebookresearch/dinov3 code), the RandomDihedral
# transform and the training hooks; MsRE and Rein add their own modules in their configs.
custom_imports = dict(imports=['skyscapes_mmseg'], allow_failed_imports=False)
default_scope = 'mmseg'
env_cfg = dict(cudnn_benchmark=True, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
               dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=0)
tta_model = dict(type='SegTTAModel')
