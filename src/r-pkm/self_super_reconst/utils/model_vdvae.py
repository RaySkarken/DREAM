"""
    VDVAE модели для работы с fMRI -> Image реконструкцией.
"""

__author__ = "DREAM Project Team"

import torch
from torch import nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from collections import defaultdict

# Добавляем путь к VDVAE
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'vdvae'))

from vae import VAE
from hps import Hyperparams
from utils import logger, local_mpi_rank, mpi_size, maybe_download, mpi_rank
from train_helpers import restore_params
from image_utils import *
from model_utils import *

class VDVAEEncoderBase(nn.Module):
    """Базовый класс энкодера VDVAE для fMRI -> Latent"""
    
    def __init__(self, n_voxels, random_crop_pad_percent=0, drop_rate=0.5):
        super().__init__()
        self.random_crop_pad_percent = random_crop_pad_percent
        self.n_voxels = n_voxels
        self.drop_rate = drop_rate
        
        # Инициализация VDVAE гиперпараметров
        self.H = Hyperparams()
        self.H.update({
            'image_size': 64, 
            'image_channels': 3,
            'dataset': 'imagenet64',
            'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5',
            'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12',
            'zdim': 16,
            'width': 512,
            'bottleneck_multiple': 0.25
        })
        
        # Создаем VDVAE модель
        self.vdvae = VAE(self.H)
        
        # Определим размер латентного пространства
        # Для VDVAE это будет суммарный размер всех z
        # Определяется автоматически при первом проходе
        self._latent_dim = None
        
        # FC слои для преобразования латентных переменных в вокселы
        # Инициализируем только после вычисления размера латентного пространства
        self.final_layer = None
        self.dropout = nn.Dropout(drop_rate)
        
        # Отмечаем, какие части модели тренируемые
        self.trainable = []
    
    def _init_final_layer(self, latent_dim):
        """Инициализирует финальный слой после определения размерности латентного пространства"""
        self.final_layer = nn.Linear(latent_dim, self.n_voxels)
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)
        self.trainable = [self.final_layer, self.dropout]
        return self.final_layer
    
    def get_latent_dim(self):
        """Возвращает размер латентного пространства VDVAE"""
        return self._latent_dim
    
    def extract_latents(self, x):
        """Извлекает латентные переменные из изображения"""
        # Подготовка изображения в нужный формат для VDVAE
        if x.shape[1] in [3, 4]:
            # Формат NCHW -> NHWC для VDVAE
            x_vdvae = x.permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError(f"Неподдерживаемый формат входных данных: {x.shape}")
            
        # Получение активаций из энкодера VDVAE
        activations = self.vdvae.encoder.forward(x_vdvae)
        
        # Извлечение латентных переменных (z) из декодера
        _, stats = self.vdvae.decoder.forward(activations, get_latents=True)
        
        # Собираем все латентные переменные в один вектор
        batch_latent = []
        for stat in stats:
            if 'z' in stat:  # Некоторые блоки могут не иметь z
                z = stat['z']
                # Изменяем форму z, чтобы сделать его плоским вектором
                batch_latent.append(z.reshape(len(x), -1))
        
        # Конкатенация всех векторов z
        latent_vector = torch.cat(batch_latent, dim=1)
        
        # Если это первый проход, инициализируем размер латентного пространства
        if self._latent_dim is None:
            self._latent_dim = latent_vector.shape[1]
            self._init_final_layer(self._latent_dim)
            print(f"Инициализирован размер латентного пространства VDVAE: {self._latent_dim}")
        
        return latent_vector
    
    def forward(self, x):
        """Прямой проход через модель"""
        # Экстракция латентных переменных
        latents = self.extract_latents(x)
        
        # Применение dropout и финального слоя
        latents = self.dropout(latents)
        return self.final_layer(latents)
    
    def load_vdvae_weights(self, weights_path):
        """Загрузка предобученных весов VDVAE"""
        print(f"Загрузка предобученных весов VDVAE из {weights_path}")
        restore_params(self.vdvae, weights_path)
        print("Веса VDVAE успешно загружены")


class VDVAEDecoderBase(nn.Module):
    """Базовый класс декодера VDVAE для fMRI -> Image"""
    
    def __init__(self, n_voxels, n_channels=3, random_crop_pad_percent=0):
        super().__init__()
        self.n_voxels = n_voxels
        self.n_channels = n_channels
        self.random_crop_pad_percent = random_crop_pad_percent
        
        # Инициализация VDVAE гиперпараметров
        self.H = Hyperparams()
        self.H.update({
            'image_size': 64, 
            'image_channels': n_channels,
            'dataset': 'imagenet64',
            'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5',
            'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12',
            'zdim': 16,
            'width': 512,
            'bottleneck_multiple': 0.25
        })
        
        # Создаем VDVAE модель
        self.vdvae = VAE(self.H)
        
        # FC слои для преобразования вокселов в латентные переменные
        # Вместо фиксированной цепочки FC слоев, создадим модули для каждого уровня иерархии VDVAE
        self.n_hierarchy_levels = 31  # По умолчанию в VDVAE примерно 31 уровень иерархии
        
        # Создаем словарь модулей для генерации латентных переменных
        self.z_predictors = nn.ModuleDict()
        self.z_dims = {}  # Словарь для хранения размерностей латентных переменных
        
        # Промежуточные слои для обработки fMRI данных
        self.fc1 = nn.Linear(n_voxels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.dropout = nn.Dropout(0.5)
        
        # Инициализируем веса
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
        # Тренируемые параметры
        self.trainable = [self.fc1, self.fc2, self.dropout]
        
        # Флаг, указывающий, были ли инициализированы предикторы z
        self.z_predictors_initialized = False
    
    def _init_z_predictors(self, test_batch_size=1):
        """Инициализирует предикторы z с фиксированной структурой"""
        # Определим фиксированную структуру латентных переменных на основе 
        # архитектуры VDVAE для ImageNet 64x64
        
        # Размерности z для различных блоков (приблизительно)
        z_dims_config = [
            (16, 1, 1),    # Самый верхний уровень (1x1)
            (16, 4, 4),    # Уровни 4x4
            (16, 8, 8),    # Уровни 8x8
            (16, 16, 16),  # Уровни 16x16
            (16, 32, 32),  # Уровни 32x32
            (16, 64, 64),  # Уровни 64x64
        ]
        
        # Количество блоков на каждом уровне 
        # (из config.dec_blocks в VDVAE)
        block_counts = [2, 3, 7, 15, 31, 12]
        
        # Распределяем блоки по уровням
        z_dims = []
        i = 0
        for level_dims, count in zip(z_dims_config, block_counts):
            for _ in range(count):
                self.z_dims[i] = level_dims
                i += 1
        
        # Создаем предикторы для каждого уровня иерархии
        hidden_dim = 4096
        for i in range(self.n_hierarchy_levels):
            if i >= len(self.z_dims):
                break  # Не создаем лишних предикторов
            
            z_dim, z_h, z_w = self.z_dims[i]
            z_flat_dim = z_dim * z_h * z_w
            
            # Создаем предиктор для данного уровня
            predictor = nn.Sequential(
                nn.Linear(hidden_dim, z_flat_dim),
                nn.LayerNorm(z_flat_dim),
                nn.LeakyReLU()
            )
            
            # Инициализация весов
            nn.init.xavier_uniform_(predictor[0].weight)
            nn.init.zeros_(predictor[0].bias)
            
            # Добавляем в словарь
            self.z_predictors[str(i)] = predictor
            
            # Добавляем к тренируемым параметрам
            self.trainable.append(predictor)
        
        self.z_predictors_initialized = True
        print(f"Инициализировано {len(self.z_predictors)} предикторов z для декодера VDVAE")
    
    def forward(self, x):
        """Прямой проход через модель: из fMRI в изображение"""
        # Убедимся, что предикторы z инициализированы
        if not self.z_predictors_initialized:
            self._init_z_predictors(test_batch_size=x.size(0))
        
        # Применяем начальные FC слои к fMRI данным
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        
        # Генерируем латентные переменные для каждого уровня иерархии
        latents = []
        for i in range(self.n_hierarchy_levels):
            # Применяем предиктор для данного уровня
            z_flat = self.z_predictors[str(i)](h)
            
            # Преобразуем плоский вектор в нужную форму для VDVAE
            z_dim, z_h, z_w = self.z_dims[i]
            z = z_flat.view(-1, z_dim, z_h, z_w)
            
            latents.append(z)
        
        # Генерируем изображение из латентных переменных
        with torch.no_grad():
            self.vdvae.eval()  # Устанавливаем в режим оценки
            image = self.vdvae.forward_samples_set_latents(x.size(0), latents)
        
        # Преобразование из NHWC в NCHW формат
        image = image.permute(0, 3, 1, 2).contiguous()
        
        return image
    
    def load_vdvae_weights(self, weights_path):
        """Загрузка предобученных весов VDVAE"""
        print(f"Загрузка предобученных весов VDVAE из {weights_path}")
        restore_params(self.vdvae, weights_path)
        print("Веса VDVAE успешно загружены")


class VDVAEEncoderRGBD(VDVAEEncoderBase):
    """Класс энкодера VDVAE для работы с RGBD изображениями"""
    
    def __init__(self, n_voxels, random_crop_pad_percent=0, drop_rate=0.5):
        super().__init__(n_voxels, random_crop_pad_percent, drop_rate)
        # Модифицируем модель для работы с 4-канальными изображениями
        self.H.image_channels = 4
        # Пересоздаем VDVAE модель
        self.vdvae = VAE(self.H)
        
        # Корректная инициализация первой свёртки для 4 каналов
        # Сохраняем веса для RGB каналов и добавляем веса для канала глубины
        with torch.no_grad():
            old_weights = self.vdvae.encoder.in_conv.weight.data.clone()
            in_channels = old_weights.size(1)
            if in_channels != 4:  # Если первая свертка еще не настроена на 4 канала
                # Инициализируем новый слой с 4 входными каналами
                new_conv = nn.Conv2d(4, old_weights.size(0), 
                                    kernel_size=3, padding=1, 
                                    bias=self.vdvae.encoder.in_conv.bias is not None)
                
                # Копируем старые веса для RGB каналов
                new_conv.weight.data[:, :3, :, :] = old_weights
                
                # Инициализируем веса для канала глубины как среднее RGB весов
                new_conv.weight.data[:, 3:4, :, :] = old_weights.mean(dim=1, keepdim=True)
                
                # Копируем bias, если он есть
                if self.vdvae.encoder.in_conv.bias is not None:
                    new_conv.bias.data = self.vdvae.encoder.in_conv.bias.data.clone()
                
                # Заменяем слой в энкодере
                self.vdvae.encoder.in_conv = new_conv


class VDVAEDecoderRGBD(VDVAEDecoderBase):
    """Класс декодера VDVAE для генерации RGBD изображений"""
    
    def __init__(self, n_voxels, random_crop_pad_percent=0):
        super().__init__(n_voxels, n_channels=4, random_crop_pad_percent=random_crop_pad_percent) 