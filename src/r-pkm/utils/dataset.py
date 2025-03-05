import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class FMRIImageDataset(Dataset):
    """
    Датасет для обучения модели на парах fMRI-изображение
    """
    def __init__(self, fmri_data, image_data, transform=None):
        self.fmri_data = fmri_data
        self.image_data = image_data
        self.transform = transform
        
    def __len__(self):
        return len(self.fmri_data)
    
    def __getitem__(self, idx):
        fmri = self.fmri_data[idx]
        image = self.image_data[idx]
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        fmri = torch.tensor(fmri, dtype=torch.float32)
        
        return fmri, image

class FMRIRGBDDataset(Dataset):
    """
    Датасет для обучения модели на парах fMRI-RGBD (RGB + глубина)
    """
    def __init__(self, fmri_data, rgb_data, depth_data, transform=None):
        self.fmri_data = fmri_data
        self.rgb_data = rgb_data
        self.depth_data = depth_data
        self.transform = transform
        
    def __len__(self):
        return len(self.fmri_data)
    
    def __getitem__(self, idx):
        fmri = self.fmri_data[idx]
        rgb = self.rgb_data[idx]
        depth = self.depth_data[idx]
        
        if isinstance(rgb, np.ndarray):
            rgb = Image.fromarray(rgb)
        
        if isinstance(depth, np.ndarray):
            if depth.ndim == 2:
                depth = Image.fromarray(depth)
            elif depth.ndim == 3 and depth.shape[2] == 1:
                depth = Image.fromarray(depth[:, :, 0])
            else:
                depth = Image.fromarray(depth)
        
        if self.transform:
            rgb = self.transform(rgb)
            depth = self.transform(depth)
        else:
            rgb = transforms.ToTensor()(rgb)
            depth = transforms.ToTensor()(depth)
            
        fmri = torch.tensor(fmri, dtype=torch.float32)
        
        return fmri, rgb, depth

def load_data(data_path, subject_id=1, split_ratio=0.8):
    """
    Загрузка данных для обучения и тестирования
    
    Args:
        data_path: путь к директории с данными
        subject_id: ID субъекта
        split_ratio: соотношение обучающей и тестовой выборок
        
    Returns:
        train_data: кортеж (train_fmri, train_images, train_depth)
        test_data: кортеж (test_fmri, test_images, test_depth)
    """
    # Загрузка fMRI данных
    fmri_path = os.path.join(data_path, f'subj{subject_id:02d}', 'fmri_data.npy')
    fmri_data = np.load(fmri_path)
    
    # Загрузка изображений
    image_path = os.path.join(data_path, f'subj{subject_id:02d}', 'nsd_train_stim_sub{subject_id}.npy')
    image_data = np.load(image_path)
    
    # Загрузка глубинных карт
    depth_path = os.path.join(data_path, f'subj{subject_id:02d}', 'nsd_train_depth_sub{subject_id}.npy')
    depth_data = np.load(depth_path)
    
    # Разделение на обучающую и тестовую выборки
    train_size = int(split_ratio * len(fmri_data))
    
    train_fmri = fmri_data[:train_size]
    train_images = image_data[:train_size]
    train_depth = depth_data[:train_size]
    
    test_fmri = fmri_data[train_size:]
    test_images = image_data[train_size:]
    test_depth = depth_data[train_size:]
    
    return (train_fmri, train_images, train_depth), (test_fmri, test_images, test_depth) 