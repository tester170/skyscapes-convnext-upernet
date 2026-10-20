"""Fine-tune DINOv3 ConvNeXt-B + UPerNet with the MMEngine Runner (same pipeline as the experiment): patches 1024 x 1024
covering every training frame once per epoch plus 25 % rare-class patches, random scale 0.75-1.5, mild photometric
jitter, 8 dihedral views; AdamW with stage-wise LR decay 0.9, fp16 + channels_last; validation every epoch (fp16), LR x0.5
after 5 validations without mIoU gain, stop after 20, best checkpoint best_mIoU_epoch_N.pth on every gain.

  --init best   start from the bundled trained model (checkpoints/convnext_b_upernet_skyscapes_ep48.pth)
  --init PATH   start from any checkpoint of this model (.pth)
--lr is the peak LR of the last stage (earlier stages get 0.9x per stage, as in the experiment). The experiment used
1e-4 from DINOv3 weights; the default for fine-tuning a trained model is a quarter of it (2.5e-5). The linear warm-up
lasts --warmup-iters iterations (default: one epoch, at most 1500 as in the experiment).
If <work-dir>/last_checkpoint exists, training resumes from it (e.g. after a Colab disconnect).
Usage: python scripts/train.py --cache <cache> --work-dir <dir> [--init best] [--lr 2.5e-5] [--max-epochs 150] [...]"""
import argparse
import json
import math
import sys
from pathlib import Path

import common

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True, help='cache of prepare_own.py (own data) or prepare_data.py (SkyScapes)')
ap.add_argument('--work-dir', required=True)
ap.add_argument('--init', default='best')
ap.add_argument('--lr', type=float, default=2.5e-5)
ap.add_argument('--max-epochs', type=int, default=150, help='upper bound; early stopping usually ends sooner')
ap.add_argument('--val-interval', type=int, default=1, help='validation every N epochs')
ap.add_argument('--batch', type=int, default=2)
ap.add_argument('--workers', type=int, default=6, help='dataloader workers (capped by the number of CPUs)')
ap.add_argument('--warmup-iters', type=int, default=0, help='0 = one epoch, at most 1500')
ap.add_argument('--lr-patience', type=int, default=5, help='validations without gain before LR x0.5')
ap.add_argument('--stop-patience', type=int, default=20, help='validations without gain before stopping')
ap.add_argument('--fp32', action='store_true', help='disable fp16 mixed precision')
ap.add_argument('--progress-interval', type=float, default=30.0, help='seconds between progress blocks')
a = ap.parse_args()

common.cap_vram()
cfg = common.load_cfg(a.cache)
wd = Path(a.work_dir)
wd.mkdir(parents=True, exist_ok=True)
cfg.work_dir = str(wd)
cfg.train_cfg.max_epochs = a.max_epochs
cfg.train_cfg.val_interval = a.val_interval
cfg.train_dataloader.batch_size = a.batch
cfg.train_dataloader.num_workers = max(1, min(a.workers, common.cpu_count()))
# an incomplete last batch of 1 patch would reach the BatchNorm of the 1x1 pyramid-pooling branch with one value per
# channel (training error); the experiment had 540 patches = 270 full batches, own data can give an odd number
cfg.train_dataloader.drop_last = True
cfg.optim_wrapper.optimizer.lr = a.lr

# iterations per epoch = (grid cells of all training frames + 25 % rare-class patches) / batch, as SkyScapesTiles
man = json.loads((Path(a.cache) / 'manifest.json').read_text())['frames']
folds = json.loads((Path(a.cache) / 'folds.json').read_text())['folds'][0]
ds = cfg.train_dataloader.dataset
cmax = int(math.floor(min(ds.crop_size) / max(ds.ratio_range)))
cells = sum(math.ceil(man[s]['shape'][0] / cmax) * math.ceil(man[s]['shape'][1] / cmax) for s in folds['train'])
iters_per_epoch = (cells + round(ds.rcs_fraction * cells)) // a.batch
if iters_per_epoch < 1:
    sys.exit('too little training data for one batch: add images to my_data/train')
warmup = a.warmup_iters or min(1500, iters_per_epoch)
cfg.param_scheduler = [dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup)]
for h in cfg.custom_hooks:
    if h['type'] == 'PlateauControlHook':
        h.update(start_iter=warmup, lr_patience=a.lr_patience, stop_patience=a.stop_patience)
    if h['type'] == 'ConsoleProgressHook':
        h['interval_s'] = a.progress_interval
if a.fp32:
    cfg.optim_wrapper.type = 'OptimWrapper'
    cfg.optim_wrapper.pop('loss_scale', None)
    cfg.val_cfg = dict(type='ValLoop')
init = common.CKPT_BEST if a.init == 'best' else Path(a.init)
if a.init == 'best' and not init.exists():                     # GitHub release
    import subprocess
    subprocess.run([sys.executable, str(common.BUNDLE / 'scripts' / 'fetch.py'), 'model'], check=True)
if not init.exists():
    sys.exit(f'{init} not found')
if (wd / 'last_checkpoint').exists():
    cfg.resume, cfg.load_from = True, None      # mmengine finds the latest checkpoint in work_dir
    print(f'resuming from {(wd / "last_checkpoint").read_text().strip()}', flush=True)
else:
    cfg.resume, cfg.load_from = False, str(init)
    print(f'initial weights: {init}', flush=True)
print(f'{len(folds["train"])} training / {len(folds["val"])} validation frames, {iters_per_epoch} iterations per epoch, '
      f'batch {a.batch}, workers {cfg.train_dataloader.num_workers}, LR {a.lr:g} (warm-up {warmup} iterations), '
      f'max epochs {a.max_epochs}, validation every {a.val_interval} epoch(s), LR x0.5 after {a.lr_patience} and stop '
      f'after {a.stop_patience} validations without gain, fp16 {not a.fp32}', flush=True)

from mmengine.runner import Runner  # noqa: E402
Runner.from_cfg(cfg).train()
best = sorted(wd.glob('best_mIoU_epoch_*.pth'), key=lambda p: int(p.stem.split('_')[-1]))
print('best checkpoint:', best[-1] if best else 'none (no validation yet)')
