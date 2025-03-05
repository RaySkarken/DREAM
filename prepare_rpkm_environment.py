#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для подготовки окружения для обработки данных и обучения модели R-PKM.
Устанавливает все необходимые зависимости и проверяет доступность CUDA.
"""

import os
import sys
import subprocess
import importlib
import argparse

def check_package(package_name):
    """Проверить, установлен ли пакет."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name, pip_args=""):
    """Установить пакет с помощью pip."""
    print(f"Установка {package_name}...")
    cmd = f"{sys.executable} -m pip install {package_name} {pip_args}"
    subprocess.check_call(cmd, shell=True)
    print(f"Пакет {package_name} успешно установлен")

def check_and_install_packages():
    """Проверить и установить необходимые пакеты."""
    required_packages = {
        "numpy": "numpy",
        "torch": "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "pillow": "pillow",
        "tqdm": "tqdm",
        "opencv-python": "opencv-python",
        "scikit-learn": "scikit-learn",
        "matplotlib": "matplotlib",
        "h5py": "h5py",
        "pandas": "pandas",
        "timm": "timm"
    }
    
    # Проверяем и устанавливаем пакеты
    for package, pip_name in required_packages.items():
        if not check_package(package):
            install_package(pip_name)
        else:
            print(f"Пакет {package} уже установлен")

def check_cuda():
    """Проверить доступность CUDA."""
    if not check_package("torch"):
        print("PyTorch не установлен. Установка...")
        install_package("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA доступен! Версия: {torch.version.cuda}")
            print(f"Количество доступных GPU: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA не доступен. Обучение будет выполняться на CPU.")
        return cuda_available
    except Exception as e:
        print(f"Ошибка при проверке CUDA: {e}")
        return False

def download_midas():
    """Скачать модель MiDaS для оценки глубины."""
    try:
        import torch
        print("Скачивание модели MiDaS для оценки глубины...")
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        print("Модель MiDaS успешно скачана.")
        return True
    except Exception as e:
        print(f"Ошибка при скачивании MiDaS: {e}")
        return False

def check_directory_structure():
    """Проверить и создать структуру директорий для данных и результатов."""
    directories = [
        "data",
        "data/processed_depth_data",
        "data/processed_depth_data/subj01",
        "results",
        "results/encoder",
        "results/decoder"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Создана директория: {directory}")
        else:
            print(f"Директория {directory} уже существует")

def main():
    parser = argparse.ArgumentParser(description='Подготовка окружения для R-PKM')
    parser.add_argument('--skip_deps', action='store_true', 
                        help='Пропустить установку зависимостей')
    parser.add_argument('--cuda_check', action='store_true',
                        help='Проверить доступность CUDA')
    parser.add_argument('--download_midas', action='store_true',
                        help='Скачать модель MiDaS для оценки глубины')
    args = parser.parse_args()
    
    print("Подготовка окружения для обработки данных и обучения модели R-PKM...")
    
    # Установка зависимостей
    if not args.skip_deps:
        check_and_install_packages()
    
    # Проверка CUDA
    if args.cuda_check or not args.skip_deps:
        check_cuda()
    
    # Скачивание MiDaS
    if args.download_midas:
        download_midas()
    
    # Проверка структуры директорий
    check_directory_structure()
    
    print("Подготовка окружения завершена.")
    print("Теперь вы можете запустить скрипт для обработки данных:")
    print("  python prepare_rpkm_data.py --data_path nsd_data --output_path data/processed_depth_data --subject_id 1")
    print("После обработки данных вы можете запустить скрипт обучения:")
    print("  python src/r-pkm/train_encoder.py --mode rgbd")
    print("  python src/r-pkm/train_decoder.py --mode rgbd")

if __name__ == "__main__":
    main() 