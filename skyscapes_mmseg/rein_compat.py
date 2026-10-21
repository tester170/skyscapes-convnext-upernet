"""Rein (w1oves/Rein, CVPR 2024; vendor/Rein, branch train, commit dc06342) modules registered in MMSeg 1.2.2 from the
authors' files, without importing their package __init__ (it pulls CLIP/EVA/SAM backbones and their dependencies):
  rein/models/backbones/reins.py               Reins, LoRAReins
  rein/models/heads/rein_mask2former.py         ReinMask2FormerHead
  rein/optimizers/peft_optimizer_constructor.py PEFTOptimWrapperConstructor"""
import importlib.util
import sys
from pathlib import Path

REIN = Path(__file__).resolve().parents[1] / 'vendor' / 'Rein' / 'rein'


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REIN / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # importable by name (torch.compile / pickling look modules up in sys.modules)
    spec.loader.exec_module(mod)
    return mod


reins = _load('rein_reins', 'models/backbones/reins.py')
heads = _load('rein_heads', 'models/heads/rein_mask2former.py')
optimizers = _load('rein_optimizers', 'optimizers/peft_optimizer_constructor.py')
