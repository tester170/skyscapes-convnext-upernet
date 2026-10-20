"""DINOv3 backbones for MMSeg 1.2.2 built from the original facebookresearch/dinov3 code (vendor/dinov3, commit 6876159).

DINOv3ConvNeXt = the official ConvNeXt (dinov3/models/convnext.py) used as in MMSeg's UPerNet-ConvNeXt recipe
(configs/convnext/convnext-base_upernet_*: mmpretrain ConvNeXt, out_indices=[0, 1, 2, 3], gap_before_final_norm=False):
the 4 stage maps (strides 4, 8, 16, 32), each through a LayerNorm. The stage-4 norm is DINOv3's pretrained final norm,
stages 1-3 get new ones. Parameter names stay downsample_layers.* / stages.*, which is what MMSeg's
LearningRateDecayOptimizerConstructor (decay_type='stage_wise') parses."""
import sys
from pathlib import Path

import torch
import torch._dynamo.config as _dynamo_cfg
import torch.nn.functional as F
from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from torch import nn

VENDOR = Path(__file__).resolve().parents[1] / 'vendor' / 'dinov3'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
# dinov3/layers/block.py sets torch._dynamo.config.accumulated_cache_size_limit, a torch.compile cache knob that torch
# 2.1 does not have; torch.compile is not used, so the key is only allowed.
_dynamo_cfg._allowed_keys.add('accumulated_cache_size_limit')
from dinov3.models.convnext import ConvNeXt, LayerNorm, convnext_sizes  # noqa: E402


@MODELS.register_module()
class DINOv3ConvNeXt(ConvNeXt):

    def __init__(self, arch='base', drop_path_rate=0.4, checkpoint=None):
        size = convnext_sizes[arch]
        super().__init__(depths=size['depths'], dims=size['dims'], drop_path_rate=drop_path_rate)
        self.out_norms = nn.ModuleList([LayerNorm(d, eps=1e-6, data_format='channels_first') for d in size['dims'][:3]])
        for n in self.out_norms:
            n.init_weights()
        self.checkpoint = checkpoint

    def init_weights(self):
        # called by MMEngine when the segmentor is built for training; a resumed/loaded checkpoint overrides it anyway
        if not self.checkpoint:
            return super().init_weights()
        sd = torch.load(self.checkpoint, map_location='cpu')['state_dict']
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith('out_norms.') for k in missing), (missing, unexpected)
        print_log(f'DINOv3ConvNeXt: {len(sd)} pretrained tensors from {self.checkpoint}; new: {len(missing)} (out_norms)',
                  logger='current')

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.stages[i](self.downsample_layers[i](x))
            outs.append(self.out_norms[i](x) if i < 3 else self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return tuple(outs)


@MODELS.register_module()
class ReinsDINOv3(BaseModule):
    """Rein on the official DINOv3 ViT-L/16 (facebookresearch/dinov3 hub builder dinov3_vitl16). Same structure as the
    authors' ReinsDinoVisionTransformer (rein/models/backbones/reins_dinov2.py + dino_v2.DinoVisionTransformer.forward):
    the frozen VFM, Reins after every block, outputs of blocks 8/12/16/24 resized to strides 4/8/16/32, and
    reins.return_auto() linking the tokens to the Mask2Former queries. Rein has no DINOv3 version; our adaptation:
    DINOv3's class token and 4 storage tokens are left untouched by Rein, as the class token of DINOv2 is.
    backbone_dtype: 'fp16' / 'bf16' runs the frozen VFM (and the Rein tokens) under that autocast, 'fp32' does not;
    the Mask2Former head always gets fp32 features (MMCV 2.1 ms_deform_attn: fp16 gave NaN, no bf16 kernel)."""

    def __init__(self, reins_config, checkpoint, out_indices=(7, 11, 15, 23), backbone_dtype='fp16'):
        super().__init__()
        from dinov3.hub.backbones import dinov3_vitl16
        self.vit = dinov3_vitl16(pretrained=False)
        self.vit.requires_grad_(False)
        self.reins = MODELS.build(reins_config)
        self.checkpoint, self.out_indices, self.backbone_dtype = checkpoint, tuple(out_indices), backbone_dtype

    def init_weights(self):
        sd = torch.load(self.checkpoint, map_location='cpu')
        self.vit.load_state_dict(sd, strict=True)          # converted + verified (tools/convert_dinov3_vit.py)
        print_log(f'ReinsDINOv3: DINOv3 ViT-L/16 from {self.checkpoint}', logger='current')

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()                                     # frozen VFM, as Rein's set_train(self, ['reins'])
        return self

    def forward(self, x):
        b, _, h, w = x.shape
        n_extra = 1 + self.vit.n_storage_tokens
        outs = []
        with torch.autocast('cuda', dtype=torch.bfloat16 if self.backbone_dtype == 'bf16' else torch.float16,
                            enabled=self.backbone_dtype != 'fp32'):
            t, (H, W) = self.vit.prepare_tokens_with_masks(x)
            for i, blk in enumerate(self.vit.blocks):
                rope = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed is not None else None
                t = blk(t, rope)
                t = torch.cat([t[:, :n_extra], self.reins(t[:, n_extra:], i, batch_first=True, has_cls_token=False)], 1)
                if i in self.out_indices:
                    outs.append(t[:, n_extra:].permute(0, 2, 1).reshape(b, -1, H, W).contiguous())
        outs = [o.float() for o in outs]
        outs[0] = F.interpolate(outs[0], scale_factor=4, mode='bilinear', align_corners=False)
        outs[1] = F.interpolate(outs[1], scale_factor=2, mode='bilinear', align_corners=False)
        outs[3] = F.interpolate(outs[3], scale_factor=0.5, mode='bilinear', align_corners=False)
        return self.reins.return_auto(outs)
