# Конфигурации MMSegmentation для моделей SkyScapes-Dense

Конфигурации четырёх моделей отчёта в формате MMSegmentation 1.2.2 (наследование от `_base_`, штатные `EncoderDecoder`, `BaseSegDataset`, `IoUMetric`), запускаются штатными `tools/train.py` и `tools/test.py` MMSegmentation.

| Конфигурация | Модель | Веса | mIoU (val, 20 классов) |
|---|---|---|---|
| `segformer_mit-b0_1xb8-amp_skyscapes-1024x1024.py` | SegFormer-B0 | `segformer_b0_skyscapes_ep220.pth` ([релиз](https://github.com/tester170/skyscapes-segformer-b0/releases/tag/v1.0)) | 36,33 |
| `dinov3-convnext-b_upernet_1xb2-amp_skyscapes-1024x1024.py` | DINOv3 ConvNeXt-B + UPerNet | `convnext_b_upernet_skyscapes_ep48.pth` ([релиз](https://github.com/tester170/skyscapes-convnext-upernet/releases/tag/v1.0)) | 43,27 |
| `msre_dinov3-vitl16_segmenter_1xb1-amp_skyscapes-1024x1024.py` | MsRE (DINOv3 ViT-L + Segmenter) | `msre_dinov3_vitl16_skyscapes_ep63.pth` (тот же релиз) | 35,70 |
| `rein_dinov3-vitl16_mask2former_1xb2_skyscapes-1024x1024.py` | Rein (DINOv3 ViT-L + Mask2Former) | `rein_dinov3_vitl16_m2f_skyscapes_ep30.pth` (тот же релиз) | 41,17 |

Значения mIoU получены штатным `tools/test.py` с этими конфигурациями и весами и совпадают с отчётом.

## Подготовка (Linux, GPU NVIDIA)
```bash
git clone https://github.com/tester170/skyscapes-convnext-upernet.git && cd skyscapes-convnext-upernet
bash env/setup_env.sh /opt/py311          # окружение эксперимента: Python 3.11, PyTorch 2.1.0, MMCV 2.1.0, MMSegmentation 1.2.2
bash env/fetch_vendor.sh                   # код DINOv3, MsRE и Rein на коммитах эксперимента -> vendor/
git clone --depth 1 --branch v1.2.2 https://github.com/open-mmlab/mmsegmentation.git ../mmsegmentation   # tools/train.py, tools/test.py
python3 scripts/fetch.py dataset --out data   # набор DLR SkyScapes -> data/SS_Dense (официальный сервер DLR)
python3 scripts/fetch.py model              # ConvNeXt-B + UPerNet -> checkpoints/ (SHA-256 проверяется)
python3 scripts/fetch.py msre               # MsRE
python3 scripts/fetch.py rein               # Rein
curl -L -o checkpoints/segformer_b0_skyscapes_ep220.pth \n    https://github.com/tester170/skyscapes-segformer-b0/releases/download/v1.0/segformer_b0_skyscapes_ep220.pth
```
Команды запускаются из корня репозитория с `PYTHONPATH=.`: так MMSegmentation находит модули `skyscapes_mmseg`, перечисленные в `custom_imports` конфигураций.

## Оценка
```bash
PYTHONPATH=. /opt/py311/bin/python ../mmsegmentation/tools/test.py \
    configs/skyscapes/dinov3-convnext-b_upernet_1xb2-amp_skyscapes-1024x1024.py checkpoints/convnext_b_upernet_skyscapes_ep48.pth
```
Оценка выполняется, как в отчёте: валидационные снимки целиком, скользящее окно 1024 × 1024 с шагом 768, fp32 (у Rein замороженный ViT в fp16, декодер в fp32). Для MsRE и Rein нужен код авторов в `vendor/` (`env/fetch_vendor.sh`).

## Обучение
```bash
PYTHONPATH=. /opt/py311/bin/python scripts/prepare_dinov3_weights.py   # начальные веса DINOv3 -> checkpoints/ (не нужно для SegFormer-B0)
PYTHONPATH=. /opt/py311/bin/python ../mmsegmentation/tools/train.py \
    configs/skyscapes/dinov3-convnext-b_upernet_1xb2-amp_skyscapes-1024x1024.py --work-dir work_dirs/convnext_b_upernet
```
Начальные веса: SegFormer-B0 — кодировщик `mit_b0_loveda_encoder.pth` из [релиза SegFormer-B0](https://github.com/tester170/skyscapes-segformer-b0/releases/tag/v1.0); остальные модели — DINOv3 LVD-1689M, которые `scripts/prepare_dinov3_weights.py` скачивает с Hugging Face (репозитории timm, с проверкой SHA-256) и переводит в форматы авторских реализаций теми же преобразованиями, что в эксперименте (результат совпадает с файлами эксперимента тензор в тензор).

Обучение повторяет эксперимент по модели, оптимизатору, разогреву, точности вычислений, аугментациям и управлению скоростью обучения: валидация раз в «эпоху» эксперимента (540 патчей), снижение скорости обучения вдвое после 5 проверок без роста mIoU и остановка после 20 (у SegFormer-B0 — после 12, проверка раз в 10 эпох), лучшая модель сохраняется как `best_mIoU_iter_N.pth`. Отличие одно: патчи 1024 × 1024 берутся штатно, случайными вырезками из случайно масштабированных снимков, тогда как в эксперименте каждая эпоха покрывала сетку обучающих снимков целиком и добавляла 25 % патчей на редких классах. Поэтому повторное обучение даст результат того же уровня, но не совпадающий до сотых.

## Состав
* `_base_/datasets/skyscapes_dense.py` — набор DLR SkyScapes-Dense в исходной структуре (`train|val/images`, `train|val/labels/grayscale`), 20 классов и палитра, конвейеры обучения и оценки, метрика.
* `_base_/default_runtime.py` — рантайм и подключение модулей проекта (`custom_imports`).
* Модули проекта (`skyscapes_mmseg/`), которых нет в MMSegmentation: `DINOv3ConvNeXt` и `ReinsDINOv3` (обёртки официального кода DINOv3), `RandomDihedral` (повороты на 90° и отражения), `PlateauControlHook` и `SpeedOptionsHook` (управление скоростью обучения и формат channels_last), `msre_compat` и `rein_compat` (подключение кода авторов MsRE и Rein).

Лицензии: DINOv3 License (`LICENSE_DINOv3.md`) для весов на основе DINOv3; код MsRE — Apache-2.0, код Rein — GPL-3.0 (скачиваются из репозиториев авторов, в этот репозиторий не входят).
