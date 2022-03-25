import torch
import torch.nn as nn
from networks.spectral import  SpectralNorm,Self_Attn



class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, image_size=256, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(2, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 256:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        # self.attn1 = Self_Attn(256, 'relu')
        # self.attn2 = Self_Attn(512, 'relu')


    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        # out,p1 = self.attn1(out)
        out=self.l4(out)
        # out,p2 = self.attn2(out)
        out=self.last(out)

        return out.squeeze()


##################################################################################
# Basic Functions
##################################################################################


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)


if __name__ == '__main__':
    pass

