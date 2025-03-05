#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для подготовки данных NSD для обучения модели R-PKM.
Преобразует данные из директории nsd_data в формат, необходимый для модели R-PKM.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import shutil
import glob
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

# Пытаемся импортировать MiDaS для оценки глубины
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    import cv2
    
    # Проверяем, установлен ли MiDaS
    midas_available = False
    try:
        import torch
        from torch.hub import load_state_dict_from_url
        midas_available = True
    except ImportError:
        print("MiDaS не установлен. Карты глубины не будут сгенерированы.")
except ImportError:
    print("PyTorch или OpenCV не установлены. Пожалуйста, установите их для полной функциональности.")
    midas_available = False

def ensure_dir(directory):
    """Создать директорию, если она не существует."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Создана директория: {directory}")

def load_webdataset(data_path, subject_id=1):
    """
    Загрузить данные из формата webdataset.
    
    Args:
        data_path: Путь к директории с данными webdataset
        subject_id: ID субъекта
        
    Returns:
        fmri_data, image_paths
    """
    print(f"Загрузка webdataset из {data_path}")
    webdataset_path = os.path.join(data_path, 'webdataset_avg_split')
    
    # Путь к данным субъекта
    subject_path = os.path.join(webdataset_path, f'subj{subject_id:02d}')
    if not os.path.exists(subject_path):
        print(f"Данные для субъекта {subject_id} не найдены в {subject_path}")
        # Проверяем, есть ли директории для других субъектов
        subj_dirs = glob.glob(os.path.join(webdataset_path, 'subj*'))
        if subj_dirs:
            subject_path = subj_dirs[0]
            subject_id = int(os.path.basename(subject_path)[4:6])
            print(f"Будут использованы данные субъекта {subject_id} из {subject_path}")
        else:
            print("Не найдено данных ни для одного субъекта. Проверьте структуру директорий.")
            return None, None
    
    # Ищем fMRI данные
    fmri_files = glob.glob(os.path.join(subject_path, '*fmri*.npy')) + \
                 glob.glob(os.path.join(subject_path, '*fmri*.npz')) + \
                 glob.glob(os.path.join(subject_path, '*voxels*.npy')) + \
                 glob.glob(os.path.join(subject_path, '*voxels*.npz'))
    
    if not fmri_files:
        print(f"fMRI данные не найдены в {subject_path}")
        return None, None
    
    print(f"Найдены fMRI файлы: {fmri_files}")
    
    # Загружаем первый найденный файл с fMRI данными
    fmri_path = fmri_files[0]
    print(f"Загрузка fMRI данных из {fmri_path}")
    
    try:
        if fmri_path.endswith('.npz'):
            fmri_data = np.load(fmri_path)
            # NPZ файл может содержать несколько массивов, берем первый
            for key in fmri_data.files:
                fmri_array = fmri_data[key]
                break
        else:
            fmri_array = np.load(fmri_path)
        
        print(f"Форма fMRI данных: {fmri_array.shape}")
    except Exception as e:
        print(f"Ошибка при загрузке fMRI данных: {e}")
        return None, None
    
    # Ищем директорию с изображениями
    image_dirs = glob.glob(os.path.join(subject_path, '*images*')) + \
                 glob.glob(os.path.join(subject_path, '*stimuli*'))
    
    if not image_dirs:
        # Ищем изображения в корневой директории
        image_files = glob.glob(os.path.join(subject_path, '*.png')) + \
                      glob.glob(os.path.join(subject_path, '*.jpg')) + \
                      glob.glob(os.path.join(subject_path, '*.jpeg'))
        
        if image_files:
            print(f"Найдено {len(image_files)} изображений в {subject_path}")
            return fmri_array, image_files
        else:
            print(f"Изображения не найдены в {subject_path}")
            return fmri_array, None
    
    image_dir = image_dirs[0]
    print(f"Найдена директория с изображениями: {image_dir}")
    
    # Ищем все изображения
    image_files = glob.glob(os.path.join(image_dir, '*.png')) + \
                  glob.glob(os.path.join(image_dir, '*.jpg')) + \
                  glob.glob(os.path.join(image_dir, '*.jpeg'))
    
    if not image_files:
        print(f"Изображения не найдены в {image_dir}")
        return fmri_array, None
    
    print(f"Найдено {len(image_files)} изображений в {image_dir}")
    
    return fmri_array, image_files

def check_midas_and_download():
    """Проверить наличие MiDaS и скачать его при необходимости."""
    if not midas_available:
        print("MiDaS не доступен. Убедитесь, что PyTorch установлен.")
        return None
    
    try:
        # Скачиваем модель MiDaS
        print("Скачивание модели MiDaS...")
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        return midas
    except Exception as e:
        print(f"Не удалось загрузить MiDaS: {e}")
        return None

def generate_depth_maps(image_files, output_path, batch_size=16):
    """
    Генерировать карты глубины для изображений с помощью MiDaS.
    
    Args:
        image_files: Список путей к изображениям
        output_path: Путь для сохранения карт глубины
        batch_size: Размер батча для обработки
        
    Returns:
        depth_maps: Массив карт глубины
    """
    if not midas_available:
        print("MiDaS не доступен. Карты глубины не будут сгенерированы.")
        return None
    
    # Загружаем модель MiDaS
    print("Инициализация модели MiDaS...")
    midas = check_midas_and_download()
    
    if midas is None:
        print("Не удалось инициализировать MiDaS. Карты глубины не будут сгенерированы.")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    
    print(f"MiDaS запущен на устройстве: {device}")
    
    # Трансформации для MiDaS
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    
    # Подготовка директории для временного сохранения карт глубины
    temp_depth_dir = os.path.join(output_path, "temp_depth")
    ensure_dir(temp_depth_dir)
    
    # Обработка изображений и генерация карт глубины
    depth_maps = []
    print(f"Генерация карт глубины для {len(image_files)} изображений...")
    
    # Класс датасета для загрузки изображений
    class ImageDataset(Dataset):
        def __init__(self, image_files, transform=None):
            self.image_files = image_files
            self.transform = transform
            
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            image_path = self.image_files[idx]
            try:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if self.transform:
                    img = self.transform({"image": img})["image"]
                
                return img, idx
            except Exception as e:
                print(f"Ошибка при загрузке изображения {image_path}: {e}")
                # Возвращаем пустое изображение
                dummy_img = np.zeros((3, 384, 384), dtype=np.float32)
                return torch.from_numpy(dummy_img), idx
    
    # Создаем датасет и даталоадер
    dataset = ImageDataset(image_files, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Массивы для хранения карт глубины
    depth_maps = np.zeros((len(image_files), 384, 384), dtype=np.float32)
    
    # Обработка изображений
    for batch_images, batch_indices in tqdm(dataloader, desc="Генерация карт глубины"):
        batch_images = batch_images.to(device)
        
        with torch.no_grad():
            batch_depth_maps = midas(batch_images)
            batch_depth_maps = torch.nn.functional.interpolate(
                batch_depth_maps.unsqueeze(1),
                size=(384, 384),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        # Конвертируем в numpy и нормализуем
        batch_depth_maps_np = batch_depth_maps.cpu().numpy()
        
        # Сохраняем карты глубины
        for i, idx in enumerate(batch_indices):
            depth_map = batch_depth_maps_np[i]
            
            # Нормализация в диапазон [0, 1]
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max > depth_min:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            
            # Сохраняем в общий массив
            depth_maps[idx] = depth_map
            
            # Сохраняем карту глубины как изображение (опционально)
            depth_image = (depth_map * 255).astype(np.uint8)
            depth_path = os.path.join(temp_depth_dir, f"depth_{idx:05d}.png")
            cv2.imwrite(depth_path, depth_image)
    
    print(f"Карты глубины сгенерированы и сохранены в {temp_depth_dir}")
    return depth_maps

def load_coco_annotations(data_path):
    """
    Загрузить аннотации COCO.
    
    Args:
        data_path: Путь к директории с данными
        
    Returns:
        annotations: Данные аннотаций
    """
    annotations_path = os.path.join(data_path, 'COCO_73k_annots_curated.npy')
    if not os.path.exists(annotations_path):
        print(f"Файл аннотаций COCO не найден: {annotations_path}")
        return None
    
    print(f"Загрузка аннотаций COCO из {annotations_path}")
    try:
        annotations = np.load(annotations_path, allow_pickle=True)
        return annotations
    except Exception as e:
        print(f"Ошибка при загрузке аннотаций COCO: {e}")
        return None

def prepare_data_for_rpkm(data_path, output_path, subject_id=1):
    """
    Подготовить данные для обучения модели R-PKM.
    
    Args:
        data_path: Путь к директории с исходными данными
        output_path: Путь для сохранения обработанных данных
        subject_id: ID субъекта
    """
    print("Начало подготовки данных для R-PKM...")
    
    # Создаем директории для сохранения результатов
    subject_output_path = os.path.join(output_path, f'subj{subject_id:02d}')
    ensure_dir(subject_output_path)
    
    # Загружаем данные webdataset
    fmri_data, image_files = load_webdataset(data_path, subject_id)
    
    if fmri_data is None:
        print("Не удалось загрузить fMRI данные. Процесс подготовки данных прерван.")
        return
    
    # Сохраняем fMRI данные
    fmri_output_path = os.path.join(subject_output_path, 'fmri_data.npy')
    np.save(fmri_output_path, fmri_data)
    print(f"fMRI данные сохранены в {fmri_output_path}")
    
    # Проверяем, что у нас есть изображения
    if image_files is None or len(image_files) == 0:
        print("Изображения не найдены. Невозможно продолжить подготовку данных.")
        return
    
    # Проверяем, что количество fMRI данных соответствует количеству изображений
    if len(fmri_data) < len(image_files):
        print(f"Предупреждение: количество fMRI данных ({len(fmri_data)}) меньше количества изображений ({len(image_files)})")
        print(f"Будут использованы только первые {len(fmri_data)} изображений")
        image_files = image_files[:len(fmri_data)]
    elif len(fmri_data) > len(image_files):
        print(f"Предупреждение: количество fMRI данных ({len(fmri_data)}) больше количества изображений ({len(image_files)})")
        print(f"Будут использованы только первые {len(image_files)} fMRI данных")
        fmri_data = fmri_data[:len(image_files)]
        # Обновляем сохраненный файл fMRI данных
        np.save(fmri_output_path, fmri_data)
        print(f"Обновленные fMRI данные сохранены в {fmri_output_path}")
    
    # Обрабатываем изображения
    print(f"Обработка {len(image_files)} изображений...")
    image_array = []
    
    for i, image_path in tqdm(enumerate(image_files), desc="Обработка изображений", total=len(image_files)):
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((64, 64), Image.LANCZOS)  # изменяем размер до 64x64
            img_array = np.array(img)
            image_array.append(img_array)
        except Exception as e:
            print(f"Ошибка при обработке изображения {image_path}: {e}")
            # Используем пустое изображение вместо ошибочного
            img_array = np.zeros((64, 64, 3), dtype=np.uint8)
            image_array.append(img_array)
    
    # Сохраняем изображения в формат .npy
    image_array = np.array(image_array)
    image_output_path = os.path.join(subject_output_path, f'nsd_train_stim_sub{subject_id}.npy')
    np.save(image_output_path, image_array)
    print(f"Изображения сохранены в {image_output_path}")
    
    # Генерируем карты глубины
    depth_maps = generate_depth_maps(image_files, subject_output_path)
    
    if depth_maps is not None:
        # Изменяем размер карт глубины до 64x64
        resized_depth_maps = np.zeros((len(depth_maps), 64, 64), dtype=np.float32)
        
        for i, depth_map in enumerate(depth_maps):
            # Используем PIL для изменения размера
            depth_img = Image.fromarray((depth_map * 255).astype(np.uint8))
            depth_img = depth_img.resize((64, 64), Image.LANCZOS)
            resized_depth_maps[i] = np.array(depth_img) / 255.0
        
        # Сохраняем карты глубины
        depth_output_path = os.path.join(subject_output_path, f'nsd_train_depth_sub{subject_id}.npy')
        np.save(depth_output_path, resized_depth_maps)
        print(f"Карты глубины сохранены в {depth_output_path}")
    else:
        print("Карты глубины не были сгенерированы. Будут использованы случайные данные.")
        # Создаем случайные карты глубины для демонстрации
        random_depth_maps = np.random.rand(len(image_array), 64, 64).astype(np.float32)
        depth_output_path = os.path.join(subject_output_path, f'nsd_train_depth_sub{subject_id}.npy')
        np.save(depth_output_path, random_depth_maps)
        print(f"Случайные карты глубины сохранены в {depth_output_path}")
    
    print("Подготовка данных завершена. Данные готовы для обучения модели R-PKM.")
    print(f"Директория с данными: {subject_output_path}")
    print(f"fMRI данные: {fmri_output_path}")
    print(f"Изображения: {image_output_path}")
    print(f"Карты глубины: {depth_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Подготовка данных для R-PKM')
    parser.add_argument('--data_path', type=str, default='nsd_data',
                        help='Путь к директории с исходными данными NSD')
    parser.add_argument('--output_path', type=str, default='data/processed_depth_data',
                        help='Путь для сохранения обработанных данных')
    parser.add_argument('--subject_id', type=int, default=1,
                        help='ID субъекта (1, 2, 5, или 7)')
    
    args = parser.parse_args()
    
    # Подготавливаем данные
    prepare_data_for_rpkm(args.data_path, args.output_path, args.subject_id)

if __name__ == "__main__":
    main() 