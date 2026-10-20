"""Your own labelled rasters -> the decoded-frame cache used for fine-tuning and evaluation (same format as
prepare_data.py builds for SkyScapes, so training, validation and metrics run unchanged).

Folder layout (masks have the same file name stem as their image):
  <data>/train/images/<name>.<ext>   <data>/train/masks/<name>.png|.tif
  <data>/val/images/<name>.<ext>     <data>/val/masks/<name>.png|.tif      optional
or, without train/ and val/:  <data>/images/... and <data>/masks/...
Without a validation part every 5th image (sorted by name) is held out for validation.

Images: JPG/JPEG/PNG/BMP/WEBP (decoded as in training, cv2) or GDAL rasters TIF/TIFF/JP2/IMG/VRT (rasterio; --bands
selects R G B, other bit depths are stretched to 8 bit, 2-98 %). Expected: 8-bit RGB, about 13 cm per pixel.
Masks: the same size as the image, either single-band class codes 1..20 (as SkyScapes labels/grayscale and the
masks written by infer.py; 0 and any other value = not labelled, ignored in training and metrics) or an RGB colour
mask in the SkyScapes palette (as labels/rgb; other colours = not labelled). Indexed-colour PNG masks are read as codes.
Cache: frames/<name>.img.npy (BGR uint8), frames/<name>.lbl.npy (train IDs 0..19, 255 = ignore), index/<name>.cls.npz,
manifest.json, folds.json. Re-running processes only new or changed files.
Usage: python scripts/prepare_own.py --data <folder> --cache <folder> [--bands 1 2 3]"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import common
import mmcv
import numpy as np

K, STRIDE = 20, 4
CV_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
GDAL_EXT = {'.tif', '.tiff', '.jp2', '.img', '.vrt'}
MASK_EXT = ('.png', '.tif', '.tiff')

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--cache', required=True)
ap.add_argument('--bands', type=int, nargs=3, default=[1, 2, 3], help='RGB band numbers of GDAL rasters (1-based)')
ap.add_argument('--val-every', type=int, default=5, help='without a val/ part: every N-th image goes to validation')
a = ap.parse_args()
src, dst = Path(a.data), Path(a.cache)
classes = json.loads((common.BUNDLE / 'splits' / 'classes.json').read_text(encoding='utf-8'))
PALETTE = np.asarray(classes['palette_rgb'], np.uint8)
NAMES = classes['names_ru']


def fail(msg):
    print('ОШИБКА ДАННЫХ:', msg, flush=True)
    sys.exit(2)


def pairs(part):
    """[(image, mask)] of one part (folder with images/ and masks/)."""
    imgs = sorted(p for p in (part / 'images').iterdir() if p.is_file() and p.suffix.lower() in CV_EXT | GDAL_EXT)
    out, missing = [], []
    for p in imgs:
        m = next((part / 'masks' / f'{p.stem}{e}' for e in MASK_EXT + tuple(e.upper() for e in MASK_EXT)
                  if (part / 'masks' / f'{p.stem}{e}').exists()), None)
        (out.append((p, m)) if m else missing.append(p.name))
    if missing:
        fail(f'нет масок для снимков {missing} в {part / "masks"} (имя маски = имя снимка, расширение .png или .tif)')
    return out


if (src / 'train' / 'images').is_dir():
    train = pairs(src / 'train')
    val = pairs(src / 'val') if (src / 'val' / 'images').is_dir() else None
elif (src / 'images').is_dir():
    train, val = pairs(src), None
else:
    fail(f'в {src} нет папки train/images (или images): структура описана в блокноте и в начале этого скрипта')
if val is None:
    if len(train) < 2:
        fail('для проверки нужен хотя бы один снимок помимо обучающих: добавьте папку val/ или ещё снимки')
    val = [p for i, p in enumerate(train) if i % a.val_every == a.val_every - 1] or [train[-1]]
    train = [p for p in train if p not in val]
    print(f'папки val/ нет: для проверки отложено {len(val)} из {len(train) + len(val)} снимков', flush=True)
if not train or not val:
    fail('нужны и обучающие, и проверочные снимки')
stems = [p.stem for p, _ in train + val]
if len(set(stems)) != len(stems):
    fail('имена снимков (без расширения) должны быть уникальными во всех папках')


def read_image(path):
    if path.suffix.lower() in CV_EXT:
        img = mmcv.imfrombytes(path.read_bytes(), flag='color', backend='cv2')       # BGR, as in training
        return img, None
    import rasterio
    with rasterio.open(path) as ds:
        bands = list(a.bands) if ds.count >= 3 else [1, 1, 1]
        rgb = ds.read(bands)
        info = None
        if ds.crs is not None and ds.crs.is_projected:
            info = f'{abs(ds.transform.a):.3f} м/пиксель'
            if not 0.09 <= abs(ds.transform.a) <= 0.18:
                warnings.warn(f'{path.name}: {info}; модель обучена на 0,13 м/пиксель, снимок лучше привести к нему')
    if rgb.dtype != np.uint8:
        lo, hi = np.percentile(rgb.reshape(3, -1)[:, ::97], [2, 98], axis=1)
        rgb = np.clip((rgb.astype(np.float64) - lo[:, None, None]) / np.maximum(hi - lo, 1e-6)[:, None, None] * 255, 0, 255)
        warnings.warn(f'{path.name}: {rgb.dtype} -> 8 bit (2-98 %); модель обучена на 8-битных RGB-снимках')
    return np.ascontiguousarray(rgb.astype(np.uint8).transpose(1, 2, 0)[..., ::-1]), info


def read_mask(path):
    """Train IDs 0..19, 255 = not labelled; also returns the number of pixels with unknown codes / colours."""
    if path.suffix.lower() in GDAL_EXT:
        import rasterio
        with rasterio.open(path) as ds:
            m = ds.read() if ds.count >= 3 else ds.read(1)
        m = m[:3].transpose(1, 2, 0) if m.ndim == 3 else m
    else:
        from PIL import Image
        im = Image.open(path)
        m = np.asarray(im if im.mode in ('L', 'P', 'I', 'I;16') else im.convert('RGB'))
    if m.ndim == 3:                                          # colour mask in the SkyScapes palette
        key = (m[..., 0].astype(np.int64) << 16) | (m[..., 1].astype(np.int64) << 8) | m[..., 2]
        lut = {(int(r) << 16) | (int(g) << 8) | int(b): i for i, (r, g, b) in enumerate(PALETTE)}
        out = np.full(key.shape, 255, np.uint8)
        for k, i in lut.items():
            out[key == k] = i
        unknown = int(((out == 255) & (key != 0)).sum())   # black = not labelled, as in labels/rgb
        return out, unknown, 'цветная'
    m = m.astype(np.int64)
    out = np.full(m.shape, 255, np.uint8)
    ok = (m >= 1) & (m <= K)
    out[ok] = (m[ok] - 1).astype(np.uint8)
    return out, int(((m > K)).sum()), 'коды'


(dst / 'frames').mkdir(parents=True, exist_ok=True)
(dst / 'index').mkdir(parents=True, exist_ok=True)
man_path = dst / 'manifest.json'
man = json.loads(man_path.read_text()) if man_path.exists() else dict(version='own_data_cache_v1', frames={})


def save(path, arr):
    tmp = path.with_suffix('.tmp.npy')
    np.save(tmp, arr)
    os.replace(tmp, path)


for split, items in (('train', train), ('val', val)):
    for img_p, msk_p in items:
        stem = img_p.stem
        sig = [str(img_p), img_p.stat().st_size, img_p.stat().st_mtime, str(msk_p), msk_p.stat().st_size, msk_p.stat().st_mtime]
        files = [dst / 'frames' / f'{stem}.img.npy', dst / 'frames' / f'{stem}.lbl.npy', dst / 'index' / f'{stem}.cls.npz']
        old = man['frames'].get(stem)
        if old and old.get('source') == sig and all(f.exists() for f in files):
            old['split'] = split
            continue
        t0 = time.perf_counter()
        img, info = read_image(img_p)
        lbl, unknown, kind = read_mask(msk_p)
        if img.shape[:2] != lbl.shape:
            fail(f'{img_p.name}: снимок {img.shape[1]}x{img.shape[0]}, маска {msk_p.name} {lbl.shape[1]}x{lbl.shape[0]} — '
                 f'размеры должны совпадать попиксельно')
        counts = np.bincount(lbl.ravel(), minlength=256)
        if counts[:K].sum() == 0:
            fail(f'{msk_p.name}: в маске нет ни одного пикселя с кодом 1–20 (или цветом классов SkyScapes)')
        g = lbl[STRIDE // 2::STRIDE, STRIDE // 2::STRIDE]
        gw = g.shape[1]
        g = g.ravel()
        order = np.argsort(g, kind='stable')
        cls = g[order]
        keep = cls < K
        order, cls = order[keep], cls[keep]
        ys, xs = np.divmod(order, gw)
        yx = np.stack([ys * STRIDE + STRIDE // 2, xs * STRIDE + STRIDE // 2], 1).astype(np.uint32)
        offsets = np.searchsorted(cls, np.arange(K + 1)).astype(np.int64)
        save(files[0], np.ascontiguousarray(img))
        save(files[1], lbl)
        tmp = files[2].with_suffix('.tmp.npz')
        np.savez(tmp, yx=yx, offsets=offsets)
        os.replace(tmp, files[2])
        man['frames'][stem] = dict(split=split, shape=list(lbl.shape), class_pixels=counts[:K].tolist(), source=sig)
        tmp = man_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(man, indent=1))
        os.replace(tmp, man_path)
        labelled = 100 * counts[:K].sum() / lbl.size
        print(f'{split}/{img_p.name}: {img.shape[1]}x{img.shape[0]}{", " + info if info else ""}, маска {msk_p.name} ({kind}), '
              f'размечено {labelled:.1f} % пикселей{f", неизвестных значений: {unknown} пикс." if unknown else ""} '
              f'[{time.perf_counter() - t0:.1f} с]', flush=True)

man['frames'] = {s: v for s, v in man['frames'].items() if s in stems}
tmp = man_path.with_suffix('.tmp')
tmp.write_text(json.dumps(man, indent=1))
os.replace(tmp, man_path)
folds = dict(description='own data: train -> fine-tuning, val -> validation',
             folds=[dict(fold=0, train=[p.stem for p, _ in train], val=[p.stem for p, _ in val])])
(dst / 'folds.json').write_text(json.dumps(folds, indent=1))
for split in ('train', 'val'):
    px = np.sum([man['frames'][s]['class_pixels'] for s in man['frames'] if man['frames'][s]['split'] == split], 0)
    share = {NAMES[c]: f'{100 * px[c] / px.sum():.1f} %' for c in np.argsort(-px) if px[c] > 0}
    print(f'{split}: {sum(1 for s in man["frames"].values() if s["split"] == split)} снимков, классы: {share}', flush=True)
print(f'кэш готов: {dst}', flush=True)
