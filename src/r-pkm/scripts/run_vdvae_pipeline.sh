#!/bin/bash

# Скрипт для запуска полного пайплайна обработки и тренировки моделей VDVAE
# Аргументы:
# $1 - Номер субъекта (1-8)
# $2 - GPU ID
# $3 - Путь к данным NSD
# $4 - Путь для сохранения обработанных данных
# $5 - Путь к весам VDVAE

# Проверка аргументов
if [ $# -lt 3 ]; then
    echo "Использование: $0 <субъект> <gpu_id> <nsd_data_path> [processed_data_path] [vdvae_weights_path]"
    exit 1
fi

sbj_num=$1
gpu=$2
nsd_data_path=$3
processed_data_path=${4:-"./processed_data"}  # По умолчанию "./processed_data"
vdvae_model_dir=${5:-"./vdvae/model"}  # По умолчанию "./vdvae/model"
vdvae_weights_path="$vdvae_model_dir/imagenet64-iter-1600000-model-ema.th"

# Создаем директории
mkdir -p "$processed_data_path"
mkdir -p "$vdvae_model_dir"
mkdir -p "./logs/vdvae/enc"
mkdir -p "./logs/vdvae/dec"
mkdir -p "./data"  # Для моделей MiDaS

# Проверяем и загружаем веса MiDaS, если они отсутствуют
midas_small_path="./data/model-small-70d6b9c8.pt"
if [ ! -f "$midas_small_path" ]; then
    echo "Загрузка модели MiDaS small..."
    wget https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small-70d6b9c8.pt -O "$midas_small_path"
fi

# Проверяем и загружаем веса VDVAE, если они отсутствуют
if [ ! -f "$vdvae_weights_path" ]; then
    echo "Загрузка весов VDVAE..."
    wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th -O "$vdvae_weights_path"
fi

echo "=== Шаг 1: Подготовка данных с MiDaS ==="
python src/r-pkm/scripts/prepare_nsd_data.py \
    --nsd_dir "$nsd_data_path" \
    --output_dir "$processed_data_path" \
    --subject "$sbj_num" \
    --img_size 64 \
    --midas_type small

# Проверяем результат
if [ $? -ne 0 ]; then
    echo "Ошибка при подготовке данных. Проверьте логи."
    exit 1
fi

echo "=== Шаг 2: Обучение энкодера VDVAE RGBD ==="
CUDA_VISIBLE_DEVICES=$gpu python src/r-pkm/self_super_reconst/train_encoder.py \
    --exp_prefix "sub${sbj_num}_vdvae_rgbd" \
    --encoder_type VDVAEEncoderRGBD \
    --n_epochs 50 \
    --learning_rate 1e-3 \
    --cos_loss 0.3 \
    --random_crop_pad_percent 3 \
    --scheduler 10 \
    --gamma 0.2 \
    --fc_gl 1 \
    --fc_mom2 10 \
    --l1_convs 1e-4 \
    --is_rgbd 1 \
    --norm_within_img 1 \
    --may_save 1 \
    --sbj_num "$sbj_num" \
    --tensorboard_log_dir "./logs/vdvae/enc" \
    --vdvae_weights "$vdvae_weights_path" \
    --data_dir "$processed_data_path"

# Проверяем результат
if [ $? -ne 0 ]; then
    echo "Ошибка при обучении энкодера. Проверьте логи."
    exit 1
fi

echo "=== Шаг 3: Обучение декодера VDVAE RGBD ==="
CUDA_VISIBLE_DEVICES=$gpu python src/r-pkm/self_super_reconst/train_decoder.py \
    --exp_prefix "sub${sbj_num}_vdvae_rgbd" \
    --enc_cpt_name "sub${sbj_num}_vdvae_rgbd_best" \
    --decoder_type VDVAEDecoderRGBD \
    --test_avg_qntl 1 \
    --learning_rate 5e-3 \
    --loss_weights 1,1,0 \
    --fc_gl 1 \
    --gl_l1 40 \
    --gl_gl 400 \
    --fc_mom2 0 \
    --l1_convs 1e-4 \
    --tv_reg 3e-1 \
    --n_epochs 150 \
    --batch_size_list 24,16,48,50 \
    --scheduler 12345 \
    --mslr 100,140 \
    --sched_gamma 0.2 \
    --percept_w 10,10,10,10,2 \
    --is_rgbd 1 \
    --norm_within_img 1 \
    --sbj_num "$sbj_num" \
    --tensorboard_log_dir "./logs/vdvae/dec" \
    --vdvae_weights "$vdvae_weights_path" \
    --data_dir "$processed_data_path"

echo "=== Процесс завершен ==="
echo "Результаты обучения сохранены в директории ./results/" 