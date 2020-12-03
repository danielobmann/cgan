import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=16, dim=32):
        super(Generator, self).__init__()
        self.dim = dim
        self.feature_sizes = [4, 4]
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.ReLU()
        )
        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, 1, 1),
            nn.ReLU()
        )

    def forward(self, z):
        z = self.latent_to_features(z)
        z = z.view(-1, 8*self.dim, *self.feature_sizes)
        z = self.features_to_image(z)
        return z


class Discriminator(nn.Module):
    def __init__(self, filters=16, img_size=(32, 32)):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, filters, 2, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, 2*filters, 2, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*filters)
        )
        self.classify = nn.Sequential(
            nn.Linear(2 * filters * img_size[0] // 4 * img_size[1] // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.features(x)
        b = feat.size(0)
        feat = feat.view(b, -1)
        return self.classify(feat)
