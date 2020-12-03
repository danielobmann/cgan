import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from networks.discriminator import PatchDiscriminator
from networks.generator import Generator
from utils import get_data
from tqdm import trange

latentdim = 16**2
steps = 4
train_loader, val_loader = get_data()


def prepare_batch(batch, latentdim=latentdim, sigma=0.1):
    z = torch.rand(batch.size(0), latentdim, device=batch.device)
    inp = batch.clone() + sigma*torch.randn(batch.shape, device=batch.device)
    inp[..., [1, 4, 32, 33, 35, 36], :] = 0
    out = batch.clone()
    return inp, z, out


G = Generator(latentdim=latentdim, steps=steps, filters=64, zsteps=3).cuda()
D = PatchDiscriminator(steps=3).cuda()

trainable_G = [p for p in G.parameters() if p.requires_grad]
trainable_D = [p for p in D.parameters() if p.requires_grad]

total_G = sum(p.numel() for p in trainable_G)
total_D = sum(p.numel() for p in trainable_D)
print("Number of parameters: %d" % total_G)
print("Number of discriminator parameters: %d" % total_D)

epochs = 3000
plotting = 50
D_steps = 10
eps = 1e-6
lr = 2e-4
opt_G = torch.optim.Adam(trainable_G, lr=lr)
opt_D = torch.optim.Adam(trainable_D, lr=lr)

crit = nn.L1Loss()
pbar = trange(epochs, unit="epoch", disable=epochs <= 0)
LOSS = []
VAL_LOSS = []
DISC_LOSS = []

for epoch in range(epochs):
    train_loss = []
    disc_loss = []
    val_loss = []

    for dstep in range(D_steps):
        for batch in train_loader:
            opt_D.zero_grad()
            batch = batch.cuda()
            inp, z, out = prepare_batch(batch=batch)
            interp = out.clone()
            interp.requires_grad = True
            grads = D(interp, inp).mean()
            grads = torch.autograd.grad(grads, interp)[0].pow(2).mean()
            pred = G(z, inp)
            loss = D(out, inp) - D(pred, inp) + grads
            loss = loss.mean()
            disc_loss.append(loss.item())
            loss.backward()
            opt_D.step()

    for batch in train_loader:
        opt_G.zero_grad()
        batch = batch.cuda()
        inp, z, out = prepare_batch(batch=batch)
        pred = G(z, inp)
        loss = D(pred, inp).mean() + crit(pred, out)
        train_loss.append(loss.item())

        loss.backward()
        opt_G.step()

    for batch in val_loader:
        batch = batch.cuda()
        inp, z, out = prepare_batch(batch=batch)
        pred = G(z, inp)
        loss = D(pred, inp).mean() + crit(pred, out)
        val_loss.append(loss.item())

    pbar.set_postfix({"loss": np.mean(train_loss),
                      "disc_loss": np.mean(disc_loss),
                      "val_loss": np.mean(val_loss)})
    pbar.update(1)
    LOSS.append(np.mean(train_loss))
    VAL_LOSS.append(np.mean(val_loss))
    DISC_LOSS.append(np.mean(disc_loss))

    if (epoch+1) % plotting == 0:
        batch = next(iter(val_loader))
        batch = batch.cuda()
        inp, z, out = prepare_batch(batch)
        pred = G(z, inp)
        probs = D(pred, inp)
        for i in range(batch.size(0)):
            plt.subplot(221)
            plt.imshow(inp.cpu().detach().numpy()[i, 0, ...])
            plt.colorbar()
            plt.title("Input")
            plt.axis('off')

            plt.subplot(222)
            plt.imshow(pred.cpu().detach().numpy()[i, 0, ...])
            plt.colorbar()
            plt.title("Prediction")
            plt.axis('off')

            plt.subplot(223)
            plt.imshow(out.cpu().detach().numpy()[i, 0, ...])
            plt.colorbar()
            plt.title("True")
            plt.axis('off')

            plt.subplot(224)
            plt.imshow(probs.cpu().detach().numpy()[i, 0, ...], vmin=0, vmax=1, cmap='magma')
            plt.colorbar()
            plt.title("Discriminator")
            plt.axis('off')

            plt.savefig("images/epoch%d_batch%d.pdf" % (epoch+1, i+1), dpi=1000)
            plt.clf()

        plt.plot(LOSS, label="train")
        plt.plot(VAL_LOSS, label="val")
        plt.plot(DISC_LOSS, label="disc")
        plt.ylim(-1.1, max(LOSS) + 1)
        plt.legend()
        plt.savefig("images/loss.pdf", dpi=1000)
        plt.clf()
