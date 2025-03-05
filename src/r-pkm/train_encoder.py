import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import argparse
import time
from tqdm import tqdm
import logging
import sys
import torchvision.transforms as transforms

from models import Encoder
from utils import FMRIImageDataset, FMRIRGBDDataset, load_data

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("encoder_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Функция потерь для обучения энкодера
def encoder_loss_function(recon_x, x, mu, logvar, kl_weight=0.5):
    # Реконструкционная потеря (MSE)
    mse_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    
    # KL дивергенция
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Общая потеря
    total_loss = mse_loss + kl_weight * kl_loss
    
    return total_loss, mse_loss, kl_loss

# Основная функция обучения
def train_encoder(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    logger.info(f"Using device: {device}")
    
    # Загрузка данных
    logger.info("Loading data...")
    (train_fmri, train_images, train_depth), (test_fmri, test_images, test_depth) = load_data(
        args.data_path, args.subject_id
    )
    
    # Преобразования для изображений
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Создание датасетов
    if args.mode == 'rgb':
        input_channels = 3
        train_dataset = FMRIImageDataset(train_fmri, train_images, transform)
        test_dataset = FMRIImageDataset(test_fmri, test_images, transform)
    elif args.mode == 'depth':
        input_channels = 1
        train_dataset = FMRIImageDataset(train_fmri, train_depth, transform)
        test_dataset = FMRIImageDataset(test_fmri, test_depth, transform)
    elif args.mode == 'rgbd':
        input_channels = 4
        # Для RGBD используем специальный датасет
        train_dataset = FMRIRGBDDataset(train_fmri, train_images, train_depth, transform)
        test_dataset = FMRIRGBDDataset(test_fmri, test_images, test_depth, transform)
    else:
        raise ValueError("Mode must be one of 'rgb', 'depth', or 'rgbd'")
    
    # Создание даталоадеров
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Инициализация модели
    encoder = Encoder(input_channels=input_channels, latent_dim=args.latent_dim).to(device)
    
    # Оптимизатор
    optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
    
    # Обучение
    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        encoder.train()
        train_loss = 0
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader)):
            if args.mode == 'rgbd':
                fmri, rgb, depth = batch_data
                # Объединяем RGB и глубину
                images = torch.cat([rgb, depth], dim=1).to(device)
            else:
                fmri, images = batch_data
                images = images.to(device)
                
            fmri = fmri.to(device)
            
            optimizer.zero_grad()
            
            # Прямой проход
            mu, logvar = encoder(images)
            z = encoder.reparameterize(mu, logvar)
            
            # Вычисление потери
            loss, mse, kl = encoder_loss_function(z, fmri, mu, logvar, args.kl_weight)
            
            # Обратное распространение
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % args.log_interval == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(images):.6f}')
        
        # Оценка на тестовой выборке
        encoder.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_data in test_loader:
                if args.mode == 'rgbd':
                    fmri, rgb, depth = batch_data
                    # Объединяем RGB и глубину
                    images = torch.cat([rgb, depth], dim=1).to(device)
                else:
                    fmri, images = batch_data
                    images = images.to(device)
                    
                fmri = fmri.to(device)
                
                mu, logvar = encoder(images)
                z = encoder.reparameterize(mu, logvar)
                
                loss, _, _ = encoder_loss_function(z, fmri, mu, logvar, args.kl_weight)
                test_loss += loss.item()
        
        test_loss /= len(test_loader.dataset)
        logger.info(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}, '
              f'Test loss: {test_loss:.4f}')
        
        # Сохранение модели
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f'encoder_{args.mode}_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / len(train_loader.dataset),
            }, save_path)
            logger.info(f"Model saved to {save_path}")
    
    # Сохранение финальной модели
    final_save_path = os.path.join(args.save_dir, f'encoder_{args.mode}_final.pt')
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss / len(train_loader.dataset),
    }, final_save_path)
    logger.info(f"Final model saved to {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Encoder for fMRI-to-Image')
    parser.add_argument('--data_path', type=str, default='data/processed_depth_data',
                        help='path to the data directory')
    parser.add_argument('--subject_id', type=int, default=1,
                        help='subject ID (1, 2, 5, or 7)')
    parser.add_argument('--mode', type=str, default='rgbd', choices=['rgb', 'depth', 'rgbd'],
                        help='training mode: rgb, depth, or rgbd')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='dimension of latent space')
    parser.add_argument('--kl_weight', type=float, default=0.5,
                        help='weight for KL divergence loss')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='how many epochs to wait before saving model')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='directory to save model checkpoints')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    
    args = parser.parse_args()
    
    # Создание директории для сохранения моделей, если она не существует
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_encoder(args) 