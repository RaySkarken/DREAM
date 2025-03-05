#!/bin/bash

# Скрипт для запуска обучения энкодера в режиме RGB

# Переходим в корневую директорию проекта
cd $(dirname $0)/..

# Запускаем обучение энкодера
python train_encoder.py \
    --data_path data/processed_depth_data \
    --subject_id 1 \
    --mode rgb \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --latent_dim 512 \
    --kl_weight 0.5 \
    --log_interval 10 \
    --save_interval 10 \
    --save_dir checkpoints \
    --gpu 0 