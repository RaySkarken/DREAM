import h5py
import numpy as np
import torch

class RGBD_fMRI_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        
        # Загрузка данных
        with h5py.File(data_path, 'r') as f:
            # Загружаем RGB изображения
            self.rgb = np.array(f[f'{split}/rgb'][:])
            # Загружаем карты глубины
            self.depth = np.array(f[f'{split}/depth'][:])
            # Загружаем fMRI данные
            self.fmri = np.array(f[f'{split}/fmri'][:])
        
        print(f"Загружено {len(self.rgb)} примеров из {split} набора")
        
    def __len__(self):
        return len(self.rgb)
        
    def __getitem__(self, idx):
        # RGB изображение
        rgb = self.rgb[idx]
        
        # Карта глубины
        depth = self.depth[idx]
        
        # Объединяем в RGBD изображение
        rgbd = np.concatenate([rgb, depth[..., np.newaxis]], axis=-1)
        
        # fMRI данные
        fmri = self.fmri[idx]
        
        # Преобразование в тензоры
        rgbd_tensor = torch.from_numpy(rgbd).float() / 255.0
        # Меняем формат с NHWC на NCHW
        rgbd_tensor = rgbd_tensor.permute(2, 0, 1)
        
        fmri_tensor = torch.from_numpy(fmri).float()
        
        # Применяем трансформации, если указаны
        if self.transform:
            rgbd_tensor = self.transform(rgbd_tensor)
        
        return rgbd_tensor, fmri_tensor 