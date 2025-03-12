#!/bin/bash
gpu=0  # GPU ID
sbj_num=1  # Можно изменить на 1, 2, 5, 7
tensorboard_log_dir=./logs/vdvae/dec
data_dir=./processed_data  # Директория с подготовленными данными
vdvae_weights_path=./vdvae/model/imagenet64-iter-1600000-model-ema.th  # Путь к весам VDVAE

# Создаем директорию для логов, если она не существует
mkdir -p $tensorboard_log_dir

# Проверяем наличие файла с весами VDVAE
if [ ! -f "$vdvae_weights_path" ]; then
    echo "Ошибка: Файл с весами VDVAE не найден: $vdvae_weights_path"
    echo "Скачайте веса с помощью следующей команды:"
    echo "wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th -O $vdvae_weights_path"
    exit 1
fi

# Запускаем обучение декодера с VDVAE для RGBD данных
python src/r-pkm/self_super_reconst/train_decoder.py \
--exp_prefix sub${sbj_num}_vdvae_rgbd \
--enc_cpt_name sub${sbj_num}_vdvae_rgbd_best \
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
--sbj_num $sbj_num \
--tensorboard_log_dir $tensorboard_log_dir \
--gpu $gpu \
--vdvae_weights $vdvae_weights_path \
--data_dir $data_dir 