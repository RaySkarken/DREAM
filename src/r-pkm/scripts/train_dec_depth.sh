#!/bin/bash

# Скрипт для запуска обучения декодера в режиме Depth

# Переходим в корневую директорию проекта
cd $(dirname $0)/..

# Запускаем обучение декодера
python train_decoder.py \
    --data_path data/processed_depth_data \
    --subject_id 1 \
    --mode depth \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --latent_dim 512 \
    --log_interval 10 \
    --save_interval 10 \
    --save_dir checkpoints \
    --encoder_path checkpoints/encoder_depth_final.pt \
    --gpu 0 