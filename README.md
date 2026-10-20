# DINOv3 ConvNeXt-B + UPerNet для DLR SkyScapes-Dense (MMSegmentation 1.2.2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tester170/skyscapes-convnext-upernet/blob/v1.0/SkyScapes_ConvNeXt_UPerNet.ipynb)

Семантическая сегментация аэрофотоснимков (20 классов DLR SkyScapes-Dense) моделью DINOv3 ConvNeXt-B + UPerNet в MMSegmentation 1.2.2: обученная модель, код конвейера, точное окружение и блокнот Colab для инференса, оценки и дообучения на своих данных.

## Запуск в Colab
1. Откройте блокнот кнопкой «Open in Colab» выше.
2. *Среда выполнения → Сменить среду выполнения → GPU*.
3. Свои снимки для инференса положите в `/content/my_images`, размеченные данные для дообучения — в `/content/my_data` (структура описана ниже и в разделе 4 блокнота).
4. *Среда выполнения → Выполнить все*. Дообучение включается параметром `TRAIN` в разделе 5, веса и результаты скачиваются кнопками раздела 7.

Блокнот сам клонирует этот репозиторий, скачивает файлы релиза `v1.0` и официальный код DINOv3 и ставит окружение с точными версиями эксперимента (Python 3.11, PyTorch 2.1.0 + CUDA 12.1, MMCV 2.1.0, MMEngine 0.10.7, MMSegmentation 1.2.2) отдельно от окружения Colab.

## Свои данные для дообучения
```
my_data/
├── train/
│   ├── images/   снимки: GeoTIFF/TIFF, JP2, IMG, VRT, JPG, PNG, BMP, WEBP (8-битные RGB, ~13 см/пиксель)
│   └── masks/    маски с теми же именами: PNG или TIFF, коды классов 1–20 (0 — нет разметки) или цвета палитры SkyScapes
└── val/          необязательно (иначе для проверки откладывается каждый пятый снимок)
    ├── images/
    └── masks/
```
Маска совпадает со своим снимком по размеру пиксель в пиксель. Коды классов — как в `labels/grayscale` набора SkyScapes и в масках, которые пишет `scripts/infer.py` (таблица — в `MODEL_CARD_RU.md` и `splits/classes.json`).

## Файлы релиза `v1.0`
| Файл | Назначение |
|---|---|
| `convnext_b_upernet_skyscapes_ep48.pth` | обученная модель (чекпойнт MMSegmentation 1.2.2, эпоха 48) |
| `mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl` | официальная сборка MMCV 2.1.0 OpenMMLab для torch 2.1 / CUDA 12.1 |
| `LICENSE_DINOv3.md` | лицензия DINOv3 |

Контрольные суммы — в `env/assets.json`; `scripts/fetch.py` проверяет каждый скачанный файл. Код DINOv3 берётся из официального репозитория facebookresearch/dinov3 на зафиксированной версии, набор SkyScapes — с официального сервера DLR.

## Без блокнота (Linux с GPU NVIDIA)
```bash
git clone --branch v1.0 https://github.com/tester170/skyscapes-convnext-upernet.git && cd skyscapes-convnext-upernet
bash env/setup_env.sh /opt/py311                      # окружение Python 3.11 и код DINOv3
python3 scripts/fetch.py model                        # веса модели из релиза
/opt/py311/bin/python scripts/infer.py --input <снимок или папка> --out <папка> --schemes 9 6 5 --fp16
/opt/py311/bin/python scripts/prepare_own.py --data my_data --cache <кэш>              # свои данные
/opt/py311/bin/python scripts/train.py --cache <кэш> --work-dir <папка запуска>         # дообучение
/opt/py311/bin/python scripts/evaluate.py --cache <кэш> --checkpoint <best_mIoU_epoch_N.pth> --out <папка>
python3 scripts/fetch.py dataset --out /data          # набор SkyScapes -> /data/SS_Dense
/opt/py311/bin/python scripts/prepare_data.py --data /data/SS_Dense --cache <кэш SkyScapes>
```

## Модель
DINOv3 ConvNeXt-B + UPerNet, 122 млн параметров, 20 классов, патчи 1024, лучшая эпоха 48. Валидационная выборка SkyScapes: mIoU 43,3 % (20 классов), κ = 0,791, OA 82,6 %; при укрупнении до 9 классов — 67,8 %, до 6 (ISPRS) — 76,5 %, до 5 — 80,7 %. Тестовая выборка SkyScapes опубликована без эталонной разметки, поэтому качество оценено по валидационной. Подробнее — `MODEL_CARD_RU.md`.

## Данные и лицензии
Набор DLR SkyScapes подготовлен Германским аэрокосмическим центром (DLR): https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-skyscapes. При использовании ссылайтесь на статью: S. M. Azimi, C. Henry, L. Sommer, A. Schumann, E. Vig. *SkyScapes — Fine-Grained Semantic Understanding of Aerial Scenes*. ICCV 2019.

Кодировщик получен из DINOv3 (O. Siméoni et al., *DINOv3*, arXiv:2508.10104; код https://github.com/facebookresearch/dinov3). Модель распространяется на условиях DINOv3 License (`LICENSE_DINOv3.md`); при публикации результатов укажите использование DINOv3.

## Состав
`configs/` конфигурация · `skyscapes_mmseg/` кодировщик DINOv3 ConvNeXt, датасет, аугментации, метрики, хуки · `scripts/` загрузка файлов (fetch.py), подготовка данных (SkyScapes и своих), дообучение, оценка, инференс, матрицы ошибок · `env/` установка и проверка окружения, файлы релиза (`assets.json`) · `splits/` классы и схема укрупнения · `SkyScapes_ConvNeXt_UPerNet.ipynb` блокнот Colab.
