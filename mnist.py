from torchvision import datasets, transforms
import torch
from networks.mnist_nets import Generator, Discriminator
from networks.mnist_train import Training


latent_dim = 64
batch_size = 128
epoochs = 200
lam_gp = 1
D_steps = 7
lr = 1e-5

transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor()])

dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)

G = Generator(latent_dim=latent_dim)
D = Discriminator()
Trainer = Training(G, D, latent_dim=latent_dim, lr=lr)

Trainer.train(train_loader=train_loader, val_loader=test_loader, epochs=epoochs, lam_gp=lam_gp, D_steps=D_steps)
