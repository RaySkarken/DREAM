# R-PKM: Reverse Parallel PKM

Реализация модуля Reverse Parallel PKM (R-PKM) для проекта DREAM. R-PKM предсказывает цвет и глубину из данных fMRI одновременно.

## Структура проекта

```
r-pkm/
├── models/                  # Модели нейронных сетей
│   ├── __init__.py
│   ├── encoder.py           # Энкодер для преобразования изображений в латентное представление
│   └── decoder.py           # Декодер для преобразования fMRI в RGB и глубину
├── utils/                   # Утилиты
│   ├── __init__.py
│   ├── dataset.py           # Классы датасетов и функции загрузки данных
│   ├── save_img_to_npy.py   # Сохранение изображений в формате npy
│   ├── save_img_from_npy.py # Загрузка изображений из формата npy
│   └── save_train_images.py # Сохранение обучающих изображений
├── scripts/                 # Скрипты для запуска обучения
│   ├── train_enc_rgb.sh     # Обучение энкодера для RGB
│   ├── train_dec_rgb.sh     # Обучение декодера для RGB
│   ├── train_enc_depth.sh   # Обучение энкодера для глубины
│   ├── train_dec_depth.sh   # Обучение декодера для глубины
│   ├── train_enc_rgbd.sh    # Обучение энкодера для RGBD
│   └── train_dec_rgbd.sh    # Обучение декодера для RGBD
├── train_encoder.py         # Основной скрипт для обучения энкодера
├── train_decoder.py         # Основной скрипт для обучения декодера
└── README.md                # Этот файл
```

## Установка

1. Убедитесь, что у вас установлены все необходимые зависимости:
   - PyTorch
   - NumPy
   - tqdm
   - PIL

2. Подготовьте данные в следующей структуре:
```
data/
└── processed_depth_data/
    └── subj01/
        ├── fmri_data.npy                # fMRI данные
        ├── nsd_train_stim_sub1.npy      # RGB изображения
        └── nsd_train_depth_sub1.npy     # Карты глубины
```

## Использование

### Обучение энкодера

Для обучения энкодера в режиме RGBD выполните:

```bash
bash scripts/train_enc_rgbd.sh
```

Для обучения энкодера только для RGB или только для глубины:

```bash
bash scripts/train_enc_rgb.sh
# или
bash scripts/train_enc_depth.sh
```

### Обучение декодера

После обучения энкодера вы можете обучить декодер:

```bash
bash scripts/train_dec_rgbd.sh
```

Для обучения декодера только для RGB или только для глубины:

```bash
bash scripts/train_dec_rgb.sh
# или
bash scripts/train_dec_depth.sh
```

### Параметры обучения

Вы можете изменить параметры обучения, отредактировав соответствующие скрипты или передав аргументы напрямую:

```bash
python train_encoder.py --data_path <путь_к_данным> --subject_id <id_субъекта> --mode <режим> --batch_size <размер_батча> --learning_rate <скорость_обучения> --num_epochs <число_эпох> --latent_dim <размерность_латентного_пространства> --kl_weight <вес_kl_дивергенции> --save_dir <директория_для_сохранения> --gpu <id_gpu>
```

```bash
python train_decoder.py --data_path <путь_к_данным> --subject_id <id_субъекта> --mode <режим> --batch_size <размер_батча> --learning_rate <скорость_обучения> --num_epochs <число_эпох> --latent_dim <размерность_латентного_пространства> --encoder_path <путь_к_энкодеру> --save_dir <директория_для_сохранения> --gpu <id_gpu>
```

## Ссылки

Реализация основана на следующих репозиториях:
- [WeizmannVision/SelfSuperReconst](https://github.com/WeizmannVision/SelfSuperReconst)
- [MedARC-AI/fMRI-reconstruction-NSD](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)