#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для запуска всего процесса подготовки данных и обучения модели R-PKM.
Этот скрипт выполняет последовательно все шаги, необходимые для обучения модели.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def run_command(command, description=None):
    """Запустить команду и вывести результат."""
    if description:
        print(f"\n=== {description} ===\n")
    
    print(f"Выполнение команды: {command}")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Вывод результатов в режиме реального времени
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            print(f"Ошибка: команда вернула код {process.returncode}")
            return False
        
        end_time = time.time()
        print(f"\nКоманда выполнена успешно за {end_time - start_time:.2f} секунд")
        return True
    
    except Exception as e:
        print(f"Ошибка при выполнении команды: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Запуск процесса обучения R-PKM')
    parser.add_argument('--data_path', type=str, default='nsd_data',
                        help='Путь к директории с исходными данными NSD')
    parser.add_argument('--output_path', type=str, default='data/processed_depth_data',
                        help='Путь для сохранения обработанных данных')
    parser.add_argument('--subject_id', type=int, default=1,
                        help='ID субъекта (1, 2, 5, или 7)')
    parser.add_argument('--mode', type=str, choices=['rgb', 'depth', 'rgbd'], default='rgbd',
                        help='Режим обучения (rgb, depth, rgbd)')
    parser.add_argument('--skip_env_setup', action='store_true',
                        help='Пропустить настройку окружения')
    parser.add_argument('--skip_data_prep', action='store_true',
                        help='Пропустить подготовку данных')
    parser.add_argument('--skip_encoder', action='store_true',
                        help='Пропустить обучение энкодера')
    parser.add_argument('--skip_decoder', action='store_true',
                        help='Пропустить обучение декодера')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Устройство для обучения (cuda, cpu)')
    
    args = parser.parse_args()
    
    print("=== Запуск процесса обучения модели R-PKM ===")
    
    current_dir = Path.cwd()
    print(f"Рабочая директория: {current_dir}")
    
    # Убедимся, что директория src/r-pkm существует
    if not os.path.exists("src/r-pkm"):
        print("Ошибка: директория src/r-pkm не найдена. Убедитесь, что вы находитесь в корневой директории проекта.")
        sys.exit(1)
    
    # 1. Подготовка окружения
    if not args.skip_env_setup:
        env_setup_success = run_command(
            "python prepare_rpkm_environment.py --download_midas",
            "Подготовка окружения"
        )
        if not env_setup_success:
            print("Ошибка при подготовке окружения. Процесс остановлен.")
            sys.exit(1)
    else:
        print("\n=== Пропуск подготовки окружения ===\n")
    
    # 2. Подготовка данных
    if not args.skip_data_prep:
        data_prep_success = run_command(
            f"python prepare_rpkm_data.py --data_path {args.data_path} --output_path {args.output_path} --subject_id {args.subject_id}",
            "Подготовка данных"
        )
        if not data_prep_success:
            print("Ошибка при подготовке данных. Процесс остановлен.")
            sys.exit(1)
    else:
        print("\n=== Пропуск подготовки данных ===\n")
    
    # 3. Обучение энкодера
    if not args.skip_encoder:
        encoder_training_success = run_command(
            f"python src/r-pkm/train_encoder.py --mode {args.mode} --device {args.device}",
            "Обучение энкодера"
        )
        if not encoder_training_success:
            print("Ошибка при обучении энкодера. Обучение декодера может быть проблематичным.")
            # Не останавливаем процесс, так как декодер может использовать предварительно обученный энкодер
    else:
        print("\n=== Пропуск обучения энкодера ===\n")
    
    # 4. Обучение декодера
    if not args.skip_decoder:
        decoder_training_success = run_command(
            f"python src/r-pkm/train_decoder.py --mode {args.mode} --device {args.device}",
            "Обучение декодера"
        )
        if not decoder_training_success:
            print("Ошибка при обучении декодера.")
    else:
        print("\n=== Пропуск обучения декодера ===\n")
    
    print("\n=== Процесс обучения R-PKM завершен ===\n")
    print("Проверьте директорию results/ для результатов обучения.")

if __name__ == "__main__":
    main() 