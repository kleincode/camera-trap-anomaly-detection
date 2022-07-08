from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, dropout=0.1, latent_features=512):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(1024, latent_features),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 4, 4)),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2, padding=2),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, 64, kernel_size=8, stride=2, padding=3),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding="same"),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x