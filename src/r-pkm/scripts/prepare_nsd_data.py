"""
    Скрипт для подготовки данных NSD для обучения VDVAE.
    Создает необходимые директории и преобразует данные в нужный формат.
"""

import os
import numpy as np
import h5py
import argparse
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import scipy.io as sio
import nibabel as nib  # Добавляем библиотеку для работы с NIfTI файлами

# Добавляем пути к необходимым модулям
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'self_super_reconst'))
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small

def create_directories():
    """Создает необходимые директории для хранения данных"""
    directories = [
        'data/processed_data',
        'data/processed_depth_data',
        'data/extracted_depth_features',
        'data/predicted_depth_features',
        'data/regression_depth_weights',
        'logs/vdvae/enc',
        'logs/vdvae/dec',
        'checkpoints'
    ]
    
    for subj in [1, 2, 5, 7]:
        directories.extend([
            f'data/processed_data/subj{subj:02d}',
            f'data/processed_depth_data/subj{subj:02d}',
            f'data/extracted_depth_features/subj{subj:02d}',
            f'data/predicted_depth_features/subj{subj:02d}',
            f'data/regression_depth_weights/subj{subj:02d}'
        ])
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Создана директория: {directory}")

def load_midas_model(model_type='small'):
    """Загрузка модели MiDaS для оценки глубины"""
    if model_type == "large":
        model_path = './data/model-f6b98070.pt'  # Путь к большой модели
        model = MidasNet(model_path, non_negative=True)
        input_size = 384
    else:
        model_path = './data/model-small-70d6b9c8.pt'  # Путь к малой модели
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", 
                              exportable=True, non_negative=True, blocks={'expand': True})
        input_size = 256
    
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    return model, input_size

def generate_depth_maps(image_paths, target_size=(64, 64), model_type='small'):
    """Генерация карт глубины для списка изображений с помощью MiDaS"""
    model, input_size = load_midas_model(model_type)
    
    # Подготовка трансформаций для изображений
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Подготовка для выходных карт глубины
    depth_maps = np.zeros((len(image_paths), target_size[0], target_size[1]), dtype=np.float32)
    
    for i, img_path in enumerate(tqdm(image_paths, desc="Генерация карт глубины")):
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        
        with torch.no_grad():
            depth = model(input_tensor)
            
        # Преобразование тензора к numpy и смена размера
        depth = depth.squeeze().cpu().numpy()
        depth_pil = Image.fromarray(depth)
        depth_resized = depth_pil.resize(target_size, Image.BICUBIC)
        
        # Нормализация глубины для сохранения в диапазоне 0-255
        depth_np = np.array(depth_resized)
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8) * 255.0
        
        depth_maps[i] = depth_np
    
    return depth_maps.astype(np.uint8)

def load_nifti_data(file_path):
    """Загрузка данных из NIfTI файла (возможно сжатого)"""
    print(f"Загрузка данных из {file_path}")
    # nibabel автоматически определяет и распаковывает gzip
    nifti_img = nib.load(file_path)
    
    # Извлекаем данные и преобразуем их в numpy массив
    data = nifti_img.get_fdata()
    
    return data

def prepare_nsd_data(nsd_dir, output_dir, subject_num, img_size=64, model_type='small'):
    """Основная функция подготовки данных NSD с генерацией карт глубины"""
    # Создание выходной директории, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Пути к данным NSD
    stim_dir = os.path.join(nsd_dir, 'nsddata_stimuli', 'stimuli')
    beta_dir = os.path.join(nsd_dir, 'nsddata_betas', 'ppdata')
    
    # Загрузка fMRI данных для субъекта
    # Путь к директории с beta-значениями
    beta_files_dir = os.path.join(beta_dir, f'subj{subject_num:02d}', 'func1pt8mm', 'betas_fithrf_GLMdenoise_RR')
    print(f"Директория с beta-файлами: {beta_files_dir}")
    
    # Перечисляем все файлы с beta-значениями
    beta_files = sorted([f for f in os.listdir(beta_files_dir) if f.startswith('betas_session') and f.endswith('.nii.gz')])
    
    if not beta_files:
        print(f"Не найдены beta-файлы в {beta_files_dir}")
        return
        
    print(f"Найдено {len(beta_files)} beta-файлов")
    
    # Загружаем и объединяем данные из всех сессий
    all_betas = []
    for beta_file in tqdm(beta_files, desc="Загрузка beta-файлов"):
        beta_path = os.path.join(beta_files_dir, beta_file)
        session_data = load_nifti_data(beta_path)
        all_betas.append(session_data)
    
    # Объединяем данные всех сессий
    betas = np.concatenate(all_betas, axis=-1)
    print(f"Форма объединенных beta-данных: {betas.shape}")
    
    # Загрузка информации о стимулах
    stim_info_file = os.path.join(nsd_dir, 'nsddata', 'experiments', 'nsd', 'nsd_stim_info_merged.mat')
    print(f"Загрузка информации о стимулах из {stim_info_file}")
    
    try:
        # Пробуем загрузить через h5py (для формата v7.3)
        with h5py.File(stim_info_file, 'r') as f:
            stim_info = {
                'subject': subject_num,
                'session': f['subjectim'][subject_num-1, :],
                'stimuli': f['stimuli'][:],
            }
    except:
        # Если не удалось, используем scipy.io для более старых форматов
        stim_data = sio.loadmat(stim_info_file)
        stim_info = {
            'subject': subject_num,
            'session': stim_data['subjectim'][subject_num-1, :] if 'subjectim' in stim_data else [],
            'stimuli': stim_data['stimuli'] if 'stimuli' in stim_data else [],
        }
    
    # Собираем пути к изображениям
    image_paths = []
    for stim_id in stim_info['stimuli']:
        if stim_id != 0:  # Стимул с ID 0 обычно означает пустой стимул
            stim_path = os.path.join(stim_dir, f'image{stim_id}.jpg')
            if os.path.exists(stim_path):
                image_paths.append(stim_path)
            else:
                print(f"Предупреждение: файл {stim_path} не найден")
    
    # Генерация карт глубины
    print("Генерация карт глубины с помощью MiDaS...")
    depth_maps = generate_depth_maps(image_paths, target_size=(img_size, img_size), model_type=model_type)
    
    # Загрузка и обработка RGB изображений
    print("Обработка RGB изображений...")
    rgb_images = np.zeros((len(image_paths), img_size, img_size, 3), dtype=np.uint8)
    
    for i, img_path in enumerate(tqdm(image_paths, desc="Обработка RGB")):
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((img_size, img_size), Image.BICUBIC)
        rgb_images[i] = np.array(img_resized)
    
    # Разделение на обучающие и тестовые данные
    # Для простоты: первые 80% для обучения, остальные для теста
    train_count = int(len(rgb_images) * 0.8)
    
    train_rgb = rgb_images[:train_count]
    train_depth = depth_maps[:train_count]
    train_betas = betas[..., :train_count]
    
    test_rgb = rgb_images[train_count:]
    test_depth = depth_maps[train_count:]
    test_betas = betas[..., train_count:]
    
    # Сохранение данных
    output_file = os.path.join(output_dir, f'nsd_subj{subject_num:02d}_data.h5')
    print(f"Сохранение данных в {output_file}")
    
    with h5py.File(output_file, 'w') as f:
        # Обучающие данные
        f.create_dataset('train/rgb', data=train_rgb)
        f.create_dataset('train/depth', data=train_depth)
        f.create_dataset('train/fmri', data=train_betas)
        
        # Тестовые данные
        f.create_dataset('test/rgb', data=test_rgb)
        f.create_dataset('test/depth', data=test_depth)
        f.create_dataset('test/fmri', data=test_betas)
        
    print("Подготовка данных завершена!")

def main():
    parser = argparse.ArgumentParser(description='Подготовка данных NSD для обучения VDVAE')
    parser.add_argument('--nsd_dir', type=str, default='./', help='Корневая директория данных NSD')
    parser.add_argument('--output_dir', type=str, default='./processed_data', help='Директория для выходных данных')
    parser.add_argument('--subject', type=int, default=1, help='Номер субъекта (1-8)')
    parser.add_argument('--img_size', type=int, default=64, help='Размер выходных изображений')
    parser.add_argument('--midas_type', type=str, default='small', choices=['small', 'large'], 
                        help='Тип модели MiDaS (small или large)')
    args = parser.parse_args()
    
    create_directories()
    prepare_nsd_data(args.nsd_dir, args.output_dir, args.subject, args.img_size, args.midas_type)

if __name__ == '__main__':
    main() 