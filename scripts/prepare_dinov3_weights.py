"""DINOv3 initial weights for training the SkyScapes configs from scratch (configs/skyscapes/*.py), LVD-1689M pretraining.

Downloads the timm releases of DINOv3 from Hugging Face (repositories timm/*, no access request needed; SHA-256 checked)
and converts them with the mappings used in the experiments (verified there: ViT-L bit-exact against Meta's own file,
ConvNeXt-B outputs equal to timm's to 6e-6):
  checkpoints/dinov3_convnext_base_lvd1689m_official.pth   official dinov3 ConvNeXt-B   (ConvNeXt-B + UPerNet)
  checkpoints/dinov3_vitl16_lvd1689m_official.pth           official dinov3 ViT-L/16     (Rein)
  checkpoints/dinov3-vitl16-pretrain-lvd1689m-hf.pth        MsRE's Hugging Face layout   (MsRE)
Each result is loaded strictly into its model (official code in vendor/dinov3, MsRE code in vendor/MsRE).
The weights are DINO Materials under the DINOv3 License (LICENSE_DINOv3.md).
Usage (repository root, environment of env/setup_env.sh, vendor code of env/fetch_vendor.sh):
  python scripts/prepare_dinov3_weights.py [convnext] [vit] [--out checkpoints]"""
import argparse
import hashlib
import json
import re
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = {
    'convnext': ('timm/convnext_base.dinov3_lvd1689m', 'afa69ae6094b5b04a48810de78c52c92d4cbb6d9436462f07bf36defb09e26c1'),
    'vit': ('timm/vit_large_patch16_dinov3.lvd1689m', '45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941'),
}


def fetch(repo, sha, out_dir):
    dst = out_dir / 'timm' / repo.split('/')[1] / 'model.safetensors'
    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == sha:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/{repo}/resolve/main/model.safetensors'
    print('download', url, flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'skyscapes'}), timeout=60) as r, \
            open(dst.with_suffix('.part'), 'wb') as f:
        for chunk in iter(lambda: r.read(1 << 22), b''):
            f.write(chunk)
    got = hashlib.sha256(dst.with_suffix('.part').read_bytes()).hexdigest()
    assert got == sha, f'{repo}: SHA-256 {got} != {sha}'
    dst.with_suffix('.part').replace(dst)
    return dst


def load_safetensors(path):
    """Minimal reader of the safetensors format (8-byte header length, JSON header, raw little-endian buffer)."""
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[:8], 'little')
    header = json.loads(raw[8:8 + n])
    buf = memoryview(raw)[8 + n:]
    out = {}
    for k, v in header.items():
        if k == '__metadata__':
            continue
        a, b = v['data_offsets']
        if v['dtype'] == 'BF16':
            t = torch.from_numpy(np.frombuffer(buf[a:b], dtype=np.int16).copy()).view(torch.bfloat16)
        else:
            t = torch.from_numpy(np.frombuffer(buf[a:b], dtype={'F32': np.float32, 'F16': np.float16}[v['dtype']]).copy())
        out[k] = t.reshape(v['shape'])
    return out


def convnext_timm_to_official(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith('stem.'):                                   # stem: conv 4x4 + LayerNorm2d
            nk = 'downsample_layers.0.' + k[len('stem.'):]
        elif m := re.fullmatch(r'stages\.(\d+)\.downsample\.(\d)\.(weight|bias)', k):
            nk = f'downsample_layers.{m[1]}.{m[2]}.{m[3]}'          # LayerNorm2d + conv 2x2
        elif m := re.fullmatch(r'stages\.(\d+)\.blocks\.(\d+)\.(.+)', k):
            tail = {'conv_dw.weight': 'dwconv.weight', 'conv_dw.bias': 'dwconv.bias', 'norm.weight': 'norm.weight',
                    'norm.bias': 'norm.bias', 'mlp.fc1.weight': 'pwconv1.weight', 'mlp.fc1.bias': 'pwconv1.bias',
                    'mlp.fc2.weight': 'pwconv2.weight', 'mlp.fc2.bias': 'pwconv2.bias', 'gamma': 'gamma'}[m[3]]
            nk = f'stages.{m[1]}.{m[2]}.{tail}'
        elif k in ('head.norm.weight', 'head.norm.bias'):          # final norm (official: norm == norms[3])
            out['norms.3.' + k.split('.')[-1]] = v
            nk = 'norm.' + k.split('.')[-1]
        else:
            raise KeyError(f'unmapped timm key {k}')
        out[nk] = v
    return out


def vit_timm_to_official(t, ref_model):
    ref = ref_model.state_dict()
    out = {'cls_token': t['cls_token'], 'storage_tokens': t['reg_token'], 'mask_token': torch.zeros_like(ref['mask_token']),
           'patch_embed.proj.weight': t['patch_embed.proj.weight'], 'patch_embed.proj.bias': t['patch_embed.proj.bias'],
           # Meta's checkpoints store the RoPE periods rounded to bf16 (pretraining precision); timm has no buffer
           'rope_embed.periods': ref['rope_embed.periods'].to(torch.bfloat16),
           'norm.weight': t['norm.weight'], 'norm.bias': t['norm.bias']}
    for i in range(24):
        p = f'blocks.{i}.'
        for k in ('norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias', 'attn.qkv.weight', 'attn.proj.weight',
                  'attn.proj.bias', 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias'):
            out[p + k] = t[p + k]
        out[p + 'ls1.gamma'], out[p + 'ls2.gamma'] = t[p + 'gamma_1'], t[p + 'gamma_2']
        bias = t.get(p + 'attn.qkv.bias')                            # timm drops the qkv bias: zero in Meta's ViT-L
        out[p + 'attn.qkv.bias'] = bias if bias is not None else torch.zeros_like(ref[p + 'attn.qkv.bias'])
        out[p + 'attn.qkv.bias_mask'] = torch.zeros_like(ref[p + 'attn.qkv.bias_mask']) if bias is None \
            else (bias != 0).to(ref[p + 'attn.qkv.bias_mask'].dtype)
    assert set(out) == set(ref) and all(out[k].shape == ref[k].shape for k in ref), sorted(set(ref) ^ set(out))
    return out


def vit_official_to_msre(o):
    d = 1024
    out = {'embeddings.cls_token': o['cls_token'], 'embeddings.mask_token': o['mask_token'].reshape(1, 1, d),
           'embeddings.register_tokens': o['storage_tokens'], 'embeddings.patch_embeddings.weight': o['patch_embed.proj.weight'],
           'embeddings.patch_embeddings.bias': o['patch_embed.proj.bias'], 'norm.weight': o['norm.weight'], 'norm.bias': o['norm.bias']}
    for i in range(24):
        b, l = f'blocks.{i}.', f'layer.{i}.'
        w, bias = o[b + 'attn.qkv.weight'], o[b + 'attn.qkv.bias'] * o[b + 'attn.qkv.bias_mask']
        assert not bias[d:2 * d].any(), 'the MsRE port has no key bias'
        out.update({l + 'norm1.weight': o[b + 'norm1.weight'], l + 'norm1.bias': o[b + 'norm1.bias'],
                    l + 'attention.q_proj.weight': w[:d], l + 'attention.q_proj.bias': bias[:d],
                    l + 'attention.k_proj.weight': w[d:2 * d],
                    l + 'attention.v_proj.weight': w[2 * d:], l + 'attention.v_proj.bias': bias[2 * d:],
                    l + 'attention.o_proj.weight': o[b + 'attn.proj.weight'], l + 'attention.o_proj.bias': o[b + 'attn.proj.bias'],
                    l + 'layer_scale1.lambda1': o[b + 'ls1.gamma'], l + 'norm2.weight': o[b + 'norm2.weight'],
                    l + 'norm2.bias': o[b + 'norm2.bias'], l + 'mlp.up_proj.weight': o[b + 'mlp.fc1.weight'],
                    l + 'mlp.up_proj.bias': o[b + 'mlp.fc1.bias'], l + 'mlp.down_proj.weight': o[b + 'mlp.fc2.weight'],
                    l + 'mlp.down_proj.bias': o[b + 'mlp.fc2.bias'], l + 'layer_scale2.lambda1': o[b + 'ls2.gamma']})
    return {k: v.contiguous().clone() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', nargs='*', help='convnext, vit (default: both)')
    ap.add_argument('--out', default=str(ROOT / 'checkpoints'))
    a = ap.parse_args()
    a.what = a.what or ['convnext', 'vit']
    assert set(a.what) <= {'convnext', 'vit'}, a.what
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from skyscapes_mmseg.dinov3_backbones import ConvNeXt, convnext_sizes  # official dinov3 code (vendor/dinov3)
    if 'convnext' in a.what:
        src = fetch(*SOURCES['convnext'], out)
        conv = convnext_timm_to_official(load_safetensors(src))
        ConvNeXt(**{k: convnext_sizes['base'][k] for k in ('depths', 'dims')}).load_state_dict(conv, strict=True)
        torch.save(dict(state_dict=conv, meta=dict(source=SOURCES['convnext'][0], source_sha256=SOURCES['convnext'][1],
                                                   arch='dinov3 ConvNeXt-base (facebookresearch/dinov3)', pretrain='LVD-1689M')),
                   out / 'dinov3_convnext_base_lvd1689m_official.pth')
        print('written', out / 'dinov3_convnext_base_lvd1689m_official.pth', flush=True)
    if 'vit' in a.what:
        from dinov3.hub.backbones import dinov3_vitl16
        src = fetch(*SOURCES['vit'], out)
        off = dinov3_vitl16(pretrained=False)
        o = vit_timm_to_official(load_safetensors(src), off)
        off.load_state_dict(o, strict=True)
        torch.save(o, out / 'dinov3_vitl16_lvd1689m_official.pth')
        print('written', out / 'dinov3_vitl16_lvd1689m_official.pth', flush=True)
        p = vit_official_to_msre(o)
        if (ROOT / 'vendor' / 'MsRE' / 'msre').is_dir():
            sys.path.insert(0, str(ROOT / 'vendor' / 'MsRE'))
            from mmseg.utils import register_all_modules
            register_all_modules()
            sys.modules.setdefault('msre.models.decode_heads', types.ModuleType('msre.models.decode_heads'))
            from msre.models.backbone.dinov3_vision_transfromer import DINOv3ViT
            DINOv3ViT(hidden_size=1024, intermediate_size=4096, num_attention_heads=16, num_hidden_layers=24,
                      num_register_tokens=4, image_size=512, layerscale_value=1e-5).load_state_dict(p, strict=True)
        torch.save(dict(state_dict=p, meta=dict(source=SOURCES['vit'][0], source_sha256=SOURCES['vit'][1])),
                   out / 'dinov3-vitl16-pretrain-lvd1689m-hf.pth')
        print('written', out / 'dinov3-vitl16-pretrain-lvd1689m-hf.pth', flush=True)


if __name__ == '__main__':
    main()
