import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim=512, output_channels=3):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        
        # Проекция из латентного пространства
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)
        
        # Основная архитектура декодера
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 2, 2)
        x = self.decoder(x)
        return x

class RGBDDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(RGBDDecoder, self).__init__()
        # Общий декодер для начальных слоев
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)
        self.initial_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Ветвь для RGB
        self.rgb_branch = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        # Ветвь для глубины
        self.depth_branch = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Для глубины используем Sigmoid, чтобы значения были в диапазоне [0, 1]
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 2, 2)
        x = self.initial_layers(x)
        
        rgb = self.rgb_branch(x)
        depth = self.depth_branch(x)
        
        return rgb, depth 