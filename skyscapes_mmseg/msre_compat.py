"""MsRE (woldier/MsRE, IEEE TGRS 2026; vendor/MsRE, commit d515366) in MMSeg 1.2.2, the authors' code unchanged.

Two adaptations, both outside their files:
1. their msre/models/__init__.py imports msre.models.decode_heads, which the published repository does not contain
   (their configs use MMSeg's own SegmenterMaskTransformerHead) -> an empty module stands in for it;
2. their DINOv3 attention is eager (QK^T materialised, softmax in fp32): at our 1024 crop (4096 tokens) it does not fit
   into 24 GB even at batch 1 (reports/bench_modern.jsonl). It is replaced by torch's scaled_dot_product_attention,
   the same computation (and what the official DINOv3 code uses)."""
import sys
import types
from pathlib import Path

import torch.nn.functional as F

VENDOR = Path(__file__).resolve().parents[1] / 'vendor' / 'MsRE'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
sys.modules.setdefault('msre.models.decode_heads', types.ModuleType('msre.models.decode_heads'))
import msre  # noqa: E402,F401  (registers MsReDINOv3 / DINOv3ViT in the mmseg registry)
import msre.models.backbone.dinov3_vision_transfromer as _dv3  # noqa: E402


def sdpa_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    assert attention_mask is None
    out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout if module.training else 0.0, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


eager_attention_original = _dv3.eager_attention_forward       # the authors' function, kept for tools/check_msre_sdpa.py
_dv3.eager_attention_forward = sdpa_attention
