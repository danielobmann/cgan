import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


class Training:
    def __init__(self, generator, discriminator, latent_dim, use_cuda=True, lr=1e-4):
        self.G = generator
        self.D = discriminator
        self.latent_dim = latent_dim
        self.use_cuda = use_cuda
        self.opt_G, self.opt_D = self._get_opt(lr=lr)
        if use_cuda:
            self.G = generator.cuda()
            self.D = discriminator.cuda()

    def _sample_generator(self, num_samples):
        z = torch.rand(num_samples, self.latent_dim)
        if self.use_cuda:
            z = z.cuda()
        return self.G(z)

    def _get_opt(self, lr=2e-4):
        opt_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, .99))
        opt_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, .99))
        return opt_G, opt_D

    def _gradient_penalty(self, generated, real):
        bs = generated.size(0)
        alpha = torch.rand(bs, 1, 1, 1, device=generated.device)
        interp = alpha*generated.data + (1.-alpha)*real.data
        interp.requires_grad = True
        grads = self.D(interp).mean()
        grads = torch.autograd.grad(grads, interp)[0]
        grads = grads.view(bs, -1)
        grads = (grads.norm(2, dim=1) - 1.).pow(2).mean()
        return grads

    def _critic_step(self, generated, real, lam_gp=10):
        self.opt_D.zero_grad()
        loss = (self.D(real) - self.D(generated)).mean() + lam_gp*self._gradient_penalty(generated, real)
        loss.backward()
        self.opt_D.step()
        return loss.item()

    def _generator_step(self, num_samples):
        self.opt_G.zero_grad()
        gen = self._sample_generator(num_samples)
        loss = self.D(gen).mean()
        loss.backward()
        self.opt_G.step()
        return loss.item()

    def train(self, train_loader, val_loader, epochs=100, D_steps=5, plotting=5, lam_gp=10):
        hist = {"gen_loss": [],
                "disc_loss": []}

        pbar = trange(epochs, unit="epoch", disable=epochs <= 0)

        for epoch in range(epochs):
            gen_loss = []
            disc_loss = []
            for i, batch in enumerate(train_loader):
                batch = batch[0]
                if self.use_cuda:
                    batch = batch.cuda()
                bs = batch.size(0)
                generated = self._sample_generator(bs)
                ld = self._critic_step(generated, batch, lam_gp=lam_gp)
                disc_loss.append(ld)

                if (i+1) % D_steps == 0:
                    lg = self._generator_step(bs)
                    gen_loss.append(lg)

            hist["gen_loss"].append(np.mean(gen_loss))
            hist["disc_loss"].append(np.mean(disc_loss))

            pbar.set_postfix({"gen_loss": np.mean(gen_loss),
                              "disc_loss": np.mean(disc_loss)})
            pbar.update(1)

            # TODO: Validation and plotting
            if (epoch+1) % plotting == 0:
                gen = self._sample_generator(4)
                if self.use_cuda:
                    gen = gen.cpu().detach().data

                for i in range(4):
                    plt.subplot(2, 2, i+1)
                    plt.imshow(gen[i, 0, ...], cmap='gray', vmin=0, vmax=1)
                    plt.colorbar()
                plt.savefig("images/mnist_epoch%d.pdf" % (epoch+1))
                plt.clf()
