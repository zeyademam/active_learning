import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

CROP_H = 64
CROP_W = 64

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""

    def __init__(self, z_dim=32, nc=3, latent_scale=None):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        ls = int(latent_scale)
        self.encoder = nn.Sequential(nn.Conv2d(nc, 128, 4, 2, 1, bias=False),  # B,  128, 32, 32
                                     nn.BatchNorm2d(128), nn.ReLU(True),
                                     nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # B,  256, 16, 16
                                     nn.BatchNorm2d(256), nn.ReLU(True),
                                     nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # B,  512,  8,  8
                                     nn.BatchNorm2d(512), nn.ReLU(True),
                                     nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # B, 1024,  4,  4
                                     nn.BatchNorm2d(1024), nn.ReLU(True), View((-1, 1024 * 2 * 2 * ls * ls)),
                                     # B, 1024*4*4
                                     )

        self.fc_mu = nn.Linear(1024 * 2 * 2 * ls * ls , z_dim)  # B, z_dim
        self.fc_logvar = nn.Linear(1024 * 2 * 2 * ls * ls, z_dim)  # B, z_dim
        self.decoder = nn.Sequential(nn.Linear(z_dim, 1024 * 4 * 4 * ls * ls),  # B, 1024*8*8
                                     View((-1, 1024, 4 * ls, 4 * ls)),  # B, 1024,  8,  8
                                     nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
                                     # B,  512, 16, 16
                                     nn.BatchNorm2d(512), nn.ReLU(True),
                                     nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                                     # B,  256, 32, 32
                                     nn.BatchNorm2d(256), nn.ReLU(True),
                                     nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                                     # B,  128, 64, 64
                                     nn.BatchNorm2d(128), nn.ReLU(True),
                                     nn.ConvTranspose2d(128, nc, 1),  # B,   nc, 64, 64
                                     )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def set_crop_seed(self, seed):
        self.crop_seed = seed

    def _gen_random_crop_index(self, h, w):
        np.random.seed(self.crop_seed)
        if w < CROP_W and h < CROP_H:
            w_start, w_end = 0, CROP_W
            h_start, h_end = 0, CROP_H
        elif w >= CROP_W and h >= CROP_H:
            w_start = np.random.randint(w - CROP_W + 1)
            w_end = w_start + CROP_W
            h_start = np.random.randint(h - CROP_H + 1)
            h_end = h_start + CROP_H
        else:
            raise ValueError("Unimplemented architecture for current image size.")

        return h_start, h_end, w_start, w_end, 

    def forward(self, x):
        crop_indxs = self._gen_random_crop_index(x.size(2), x.size(3))
        x_random_crop = x[:, :, crop_indxs[0]:crop_indxs[1], crop_indxs[2]:crop_indxs[3]]
        z = self._encode(x_random_crop)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return x_random_crop, x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
