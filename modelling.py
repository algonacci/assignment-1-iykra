import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, image_size=(128, 128)):
        super(Generator, self).__init__()
        self.image_size = image_size
        initial_size = image_size[0] // 16

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024 * initial_size * initial_size),
            nn.BatchNorm1d(1024 * initial_size * initial_size),
            nn.ReLU(inplace=True)
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
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

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 1024, self.image_size[0] // 16, self.image_size[1] // 16)  # Reshape for convolutions
        img = self.conv_layers(x)
        return img

class Discriminator(nn.Module):
    def __init__(self, image_size=(128, 128)):
        super(Discriminator, self).__init__()
        self.image_size = image_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(1024 * (image_size[0] // 32) * (image_size[1] // 32), 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = self.conv_layers(img)
        x = x.view(x.size(0), -1)
        validity = self.fc(x)
        return validity
