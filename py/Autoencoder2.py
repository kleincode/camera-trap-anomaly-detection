# This is the preferred autoencoder architecture.
# Fully convolutional with 7 layer encoder and decoder.
# Dropout, relu on hidden layers, tanh on output layer
# Allows multiples of 16 as number of latent features

from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, dropout=0.1, latent_features=512):
        super(Autoencoder, self).__init__()
        if latent_features % 16 != 0:
            raise ValueError("latent_features must be a multiple of 16 in this architecture.")
        latent_channels = latent_features // 16
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(128, latent_channels, kernel_size=3, padding="same"),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(latent_channels, 128, kernel_size=3, padding="same"),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=2),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(32, 16, kernel_size=8, stride=2, padding=3),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding="same"),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x