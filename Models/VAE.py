# ------------------------------------------------------------------------------
# PyTorch implementation of a convolutional Variational Autoencoder (VAE).
# ------------------------------------------------------------------------------

import torch 
import numpy as np

from torch import nn


class ConvBlock (nn.Sequential):
    def __init__ (self, in_c, out_c, kernel_size, stride=1, padding=1):
        super().__init__()
        self.add_module('Convolution', nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size, stride, padding)))
        self.add_module('BatchNorm', nn.BatchNorm2d(out_c, affine=True))
        self.add_module('Activation', nn.Tanh())

class ConvTransposeBlock (nn.Sequential):
    def __init__ (self, in_c, out_c, kernel_size, stride=1):
        super().__init__()
        self.add_module('ConvTranspose', nn.utils.spectral_norm(nn.ConvTranspose2d(in_c, out_c, kernel_size, stride)))
        self.add_module('BatchNorm', nn.BatchNorm2d(out_c, affine=True))
        self.add_module('Activation', nn.Tanh())

class Lambda (nn.Module):
    def __init__ (self, f):
        super().__init__()
        self.f = f
    def forward (self, x):
        return self.f(x)


class Encoder (nn.Module):
    """
    Encoder of 2D VAE, producing latent multivariate normal distributions from input images. We return logarithmic variances as they have the real numbers as domain.

    Forward pass:
        1 Input: 
            i)  Image of shape [N, C, H, W] (by default 1x28x28 MNIST images)
        2 Outputs: 
            i)  Means of latent distributions of shape [N, latent_dim]
            ii) Logarithmic variances of latent distribution of shape [N, latent_dim] (Approximation of multivariate Gaussian, covariance is strictly diagonal, i.e. [N, d, d] is now [N, d])

    Arguments:
        X_dim (list) : dimensions of input 2D image, in the form of [Channels, Height, Width]
        latent_dim (int) : dimension of latent space.
    """
    def __init__(self, X_dim=[1,28,28], latent_dim=2):
        super(Encoder, self).__init__()
        
        conv1_outchannels = 16
        conv2_outchannels = 32

        # How the convolutions change the shape
        conv_outputshape = (
            conv2_outchannels 
            * int(((X_dim[1]-3)/2 - 2)/2 + 1) 
            * int(((X_dim[2]-3)/2 - 2)/2 + 1)
        )

        self.enc = torch.nn.Sequential(
                ConvBlock(X_dim[0], conv1_outchannels, kernel_size=3, stride=2),
                ConvBlock(conv1_outchannels, conv2_outchannels, kernel_size=3, stride=2),
                Lambda(lambda x: x.view(-1, conv_outputshape))
        )
        
        self.zmean = nn.Linear(conv_outputshape, latent_dim)
        self.zlogvar = nn.Linear(conv_outputshape, latent_dim)
        
        
    def forward (self, X, L):
        result = self.enc(X)
        mean = self.zmean(result)
        logvar = self.zlogvar(result)

        # z_list = []
        # for i in range(L):         
        #     xi = torch.normal(torch.zeros_like(mean))
        #     z = mean + torch.exp(logvar/2) * xi
        #     z_list.append(z)
        
        return mean, logvar


class Decoder (nn.Module):
    """
    Decoder of 2D VAE, producing output multivariate normal distributions from latent vectors. We return logarithmic variances as they have the real numbers as domain.

    Forward pass:
        1 Input: 
            i)   Latent vector of shape [N, latent_dim]
        2 Outputs: 
            i)  Means of output distributions of shape [N, C, H, W]
            ii) Variances of output distribution of shape [N, C, H, W] (Approximation of multivariate Gaussian, covariance is strictly diagonal). We assume constant variance 

    Arguments:
        X_dim (list) : dimensions of input 2D image, in the form of [Channels, Height, Width]
        latent_dim (int) : dimension of latent space.
    """
    def __init__(self, X_dim=[1,28,28], latent_dim=2):
        super(Decoder, self).__init__()
        
        conv1_outchannels = 16
        conv2_outchannels = 32

        # How the convolutions change the shape
        conv_outputshape = (
            int(((X_dim[1]-3)/2 - 2)/2 + 1),
            int(((X_dim[2]-3)/2 - 2)/2 + 1)
        )

        self.dec = nn.Sequential(
                nn.Linear(latent_dim, conv2_outchannels*conv_outputshape[0]*conv_outputshape[1]),
                Lambda(lambda x: x.view(-1, conv2_outchannels, conv_outputshape[0], conv_outputshape[1])),
                ConvTransposeBlock(conv2_outchannels, conv1_outchannels, kernel_size=3, stride=2),
                # Need kernel of 4 instead of 3 as the shape is 27*27 here otherwise.
                ConvTransposeBlock(conv1_outchannels, 3, kernel_size=4, stride=2)
        )

        self.Xmean = nn.Sequential(
                nn.Conv2d(3, X_dim[0], kernel_size=1),
                # Output is grayscale between 0 and 1
                nn.Sigmoid()
        )

    
    def forward (self, z):
        result = self.dec(z)
        
        mean = self.Xmean(result)
        # We freeze the variance as constant 0.5
        logvar = torch.log(torch.ones_like(mean) * 0.5)
        
        return mean, logvar

