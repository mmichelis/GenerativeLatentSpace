# ------------------------------------------------------------------------------
# Some utility functions used in the models.
# ------------------------------------------------------------------------------

import torch as pt
from torch import nn

from argparse import ArgumentParser


class LinBlock (nn.Sequential):
    def __init__ (self, in_c, out_c):
        super().__init__()
        self.add_module('Linear', nn.utils.spectral_norm(nn.Linear(in_c, out_c)))
        self.add_module('BatchNorm', nn.BatchNorm1d(out_c, affine=True))
        self.add_module('Activation', nn.ELU())

class ConvBlock (nn.Sequential):
    def __init__ (self, in_c, out_c, kernel_size, stride=1):
        super().__init__()
        self.add_module('Convolution', nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size, stride)))
        self.add_module('BatchNorm', nn.BatchNorm2d(out_c, affine=True))
        self.add_module('Activation', nn.ELU())

class ConvTransposeBlock (nn.Sequential):
    def __init__ (self, in_c, out_c, kernel_size, stride=1):
        super().__init__()
        self.add_module('ConvTranspose', nn.utils.spectral_norm(nn.ConvTranspose2d(in_c, out_c, kernel_size, stride)))
        self.add_module('BatchNorm', nn.BatchNorm2d(out_c, affine=True))
        self.add_module('Activation', nn.ELU())

# Lambda module does not like to be torch.save 'd.
class Lambda (nn.Module):
    def __init__ (self, f):
        super().__init__()
        self.f = f
    def forward (self, x):
        return self.f(x)


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--lr', help="Learning to start training with.", type=float, default=5e-3)
    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--digits', help='Comma-separated list of MNIST digits to use.', type=str, default=None)

    args = parser.parse_args()
    
    if args.digits is not None:
        args.digits = [int(item) for item in args.digits.split(',')]

    return args
