import torch.nn as nn
import torch.nn.init as init


class Discriminator(nn.Module):
    """Adversary architecture (Discriminator) for WAE-GAN."""

    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(nn.Linear(z_dim, 512), nn.ReLU(True), nn.Linear(512, 512),
                                 nn.ReLU(True), nn.Linear(512, 1), nn.Sigmoid())
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
