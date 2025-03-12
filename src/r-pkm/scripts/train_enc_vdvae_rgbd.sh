#!/bin/bash
gpu=0  # GPU ID
sbj_num=1  # Можно изменить на 1, 2, 5, 7
tensorboard_log_dir=./logs/vdvae/enc
data_dir=./processed_data  # Директория с подготовленными данными
vdvae_weights_path=./vdvae/model/imagenet64-iter-1600000-model-ema.th  # Путь к весам VDVAE

# Создаем директорию для логов, если она не существует
mkdir -p $tensorboard_log_dir

# Проверка наличия директории для весов VDVAE
vdvae_model_dir=$(dirname "$vdvae_weights_path")
mkdir -p "$vdvae_model_dir"

# Проверка наличия весов VDVAE и при необходимости их загрузка
if [ ! -f "$vdvae_weights_path" ]; then
    echo "Веса VDVAE не найдены по пути $vdvae_weights_path. Загружаем..."
    wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th -O "$vdvae_weights_path"
fi

# Запускаем обучение энкодера с VDVAE для RGBD данных
python src/r-pkm/self_super_reconst/train_encoder.py \
--exp_prefix sub${sbj_num}_vdvae_rgbd \
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
--sbj_num $sbj_num \
--tensorboard_log_dir $tensorboard_log_dir \
--gpu $gpu \
--vdvae_weights $vdvae_weights_path \
--data_dir $data_dir 