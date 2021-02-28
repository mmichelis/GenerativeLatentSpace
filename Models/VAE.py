# ------------------------------------------------------------------------------
# PyTorch implementation of a convolutional Variational Autoencoder (2014 D.
# Kingma "Auto-Encoding Variational Bayes" in https://arxiv.org/abs/1312.6114)
# ------------------------------------------------------------------------------

import os

import matplotlib.pyplot as plt
import numpy as np
import torch as pt

from torch import nn
from torchsummary import summary
from torchvision.utils import save_image

import sys
sys.path.append('./')
from Models.base import ConvBlock, ConvTransposeBlock, get_args
from Models.utility.RBF import trainRBF, RBF
from dataloader import MNISTDigits, FashionMNIST, EMNIST


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
    def __init__(self, X_dim=[1,28,28], latent_dim=16):
        super(Encoder, self).__init__()

        conv1_outchannels = 32
        conv2_outchannels = 32

        # How the convolutions change the shape
        conv_outputshape = (
            conv2_outchannels
            * int(((X_dim[1]-4)/2 - 2)/2 + 1)
            * int(((X_dim[2]-4)/2 - 2)/2 + 1)
        )

        self.enc = nn.Sequential(
            ConvBlock(X_dim[0], conv1_outchannels, kernel_size=4, stride=2),
            ConvBlock(conv1_outchannels, conv2_outchannels, kernel_size=3, stride=2)
        )

        self.zmean = nn.Linear(conv_outputshape, latent_dim)
        self.zlogvar = nn.Linear(conv_outputshape, latent_dim)


    def forward (self, X):
        x = self.enc(X)
        x = x.view(X.shape[0], -1)
        mean = self.zmean(x)
        logvar = self.zlogvar(x)

        return mean, logvar


class Decoder (nn.Module):
    """
    Decoder of 2D VAE, producing output multivariate normal distributions from latent vectors. We return logarithmic variances as they have the real numbers as domain.

    Forward pass:
        1 Input:
            i)  Latent vector of shape [N, latent_dim]
        2 Outputs:
            i)  Means of output distributions of shape [N, C, H, W]
            ii) Variances of output distribution of shape [N, C, H, W] (Approximation of multivariate Gaussian, covariance is strictly diagonal). We assume constant variance during VAE training.

    Arguments:
        X_dim (list) : dimensions of input 2D image, in the form of [Channels, Height, Width]
        latent_dim (int) : dimension of latent space.
    """
    def __init__(self, X_dim=[1,28,28], latent_dim=16):
        super(Decoder, self).__init__()
        
        # Currently number of clusters is set to 4*latent_dim as a good rule of thumb, can be changed, but then also change it later during training!
        self.improved_variance = True
        k = 4*latent_dim
        self.rbfNN = RBF(centers=pt.zeros(k,latent_dim), bandwidth=pt.zeros(k), X_dim=np.prod(X_dim))

        conv1_outchannels = 32
        conv2_outchannels = 32

        # How the convolutions change the shape
        self.conv_outputshape = (
            int(((X_dim[1]-4)/2 - 2)/2 + 1),
            int(((X_dim[2]-4)/2 - 2)/2 + 1)
        )

        self.lin = nn.Linear(latent_dim, conv2_outchannels*self.conv_outputshape[0]*self.conv_outputshape[1])

        self.conv = nn.Sequential(
            ConvTransposeBlock(conv2_outchannels, conv1_outchannels, kernel_size=3, stride=2),
            ConvTransposeBlock(conv1_outchannels, 32, kernel_size=4, stride=2)
        )

        self.Xmean = nn.Sequential(
            nn.Conv2d(32, X_dim[0], kernel_size=1),
            # Output is grayscale between -1 and 1
            nn.Tanh()
        )


    def forward (self, z):
        """
        When improved_variance is set to True, we use a trained RBF to return a better variance estimate as described in "Arvanitidis et al. (2018): Latent Space Oddity". Of course the RBF has to be assigned to the Decoder first.
        """
        x = self.lin(z)
        x = x.view(z.shape[0], -1, self.conv_outputshape[0], self.conv_outputshape[1])
        x = self.conv(x)
        mean = self.Xmean(x)

        if not self.improved_variance:
            # We freeze the variance as constant 0.5. Requires grad so metric computation goes smoothly (i.e. returns no gradient)
            var = pt.ones_like(mean, requires_grad=True) * 0.5
        else:
            var = 1/self.rbfNN(z)

        return mean, var


def train (dataloader, latent_dim=2, lr=5e-3, max_epochs=100, device=None):
    """
    Trains the VAE on data presented in dataloader for max_epochs. By default trained using Adam optimizer with learning rate 5e-3 and weight decay 1e-4, and having a multiplicative learning rate scheduler (0.95 multiplier per epoch).

    Arguments:
        dataloader (nn.utils.data.Dataloader) : 2D images of shape [N, C, H, W]
            loaded in batches of N. Unsupervised, so no targets/labels needed.
        latent_dim (int) : latent dimension of autoencoder.

    Returns:
        modelE (Encoder: nn.Module) : trained encoder architecture.
        modelD (Decoder: nn.Module) : trained decoder architecture.
    """
    if not os.path.exists("Outputs"):
        os.makedirs("Outputs")
    if not os.path.exists("TrainedModels"):
        os.makedirs("TrainedModels")

    if device is None:
        device = pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu')

    X_dim = dataloader.dataset[0][0].shape

    ### Initialize encoder, decoder and optimizers
    modelE = Encoder(X_dim, latent_dim).to(device)
    modelD = Decoder(X_dim, latent_dim).to(device)
    modelE.train(), modelD.train()
    modelD.improved_variance = False    # During training of VAE no RBF

    # Show the network architectures
    summary(modelE, X_dim)
    summary(modelD, (latent_dim,))

    optimizer = pt.optim.Adam(
                list(modelE.parameters())+list(modelD.parameters()),
                lr=lr,
                weight_decay=1e-4
            )

    scheduler = pt.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.95)


    ### Loop over epochs
    loss_history = []
    for epoch in range(max_epochs):
        ### Store an evaluation output in every epoch
        with pt.set_grad_enabled(False):
            randidx = np.random.randint(len(dataloader.dataset))
            output = modelD(
                modelE(
                    dataloader.dataset[randidx][0].to(device).unsqueeze(dim=0)
                )[0]
            )[0].detach().cpu().squeeze().numpy()

        fig = plt.figure(figsize=(12, 12))
        # Image from [-1,1] to [0,1]
        plt.imshow((output+1)/2, cmap='gray')
        plt.savefig(f"Outputs/train_{epoch+1: 04d}.png")
        plt.close(fig)


        L = 4
        epoch_loss = 0
        for X_input in dataloader:
            with pt.set_grad_enabled(True):
                # No need for targets/labels
                X_input = X_input[0].to(device)
                modelE.zero_grad(), modelD.zero_grad()

                zmean, zlogvar = modelE(X_input)
                kl_loss = -0.5 * (1 + zlogvar - zmean**2 - zlogvar.exp()).sum(dim=1)

                rec_loss = 0.0
                for _ in range(L):
                    # Reparametrization trick
                    xi = pt.normal(pt.zeros_like(zmean))
                    z = zmean + pt.exp(zlogvar/2) * xi

                    Xmean, Xlogvar = modelD(z)
                    # rec_loss as the negative log likelihood. Constant log(2pi^k/2) keeps loss positive, but is optional.
                    rec_loss += 0.5 * Xlogvar.sum(dim=[1,2,3]) + 0.5 * ((X_input - Xmean)**2 / Xlogvar.exp()).sum(dim=[1,2,3])
                    rec_loss += (np.prod(X_input.shape[1:])/2) * np.log(6.283)
                rec_loss /= L

                loss = (kl_loss + rec_loss).mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        # Length of dataloader is the amount of batches, not the total number of data points
        epoch_loss /= len(dataloader.dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{max_epochs}]: Loss {epoch_loss:.4e}")


        ### See how the loss evolves
        fig = plt.figure(figsize=(12,9))
        plt.plot(loss_history, label='Loss History')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(0, epoch+1)
        plt.legend()
        plt.grid(True)
        fig.savefig("Outputs/LossHistory.png", bbox_inches='tight')
        plt.close(fig)

        pt.save(modelE.state_dict(), "TrainedModels/trainedVAE_E.pth")
        pt.save(modelD.state_dict(), "TrainedModels/trainedVAE_D.pth")


    ### After training has finished, create better variance estimate with RBF
    # Currently parameters are just set with default values, work well in general setting.
    rbfNN = trainRBF(modelE, modelD, dataloader, latent_dim, X_dim, k=4*latent_dim, zeta=1e-2, curveMetric=1, max_epochs=50, batch_size=dataloader.batch_size)
    # Set the better estimate in the decoder
    modelD.rbfNN = rbfNN
    modelD.improved_variance = True

    pt.save(modelD.state_dict(), "TrainedModels/trainedVAE_D.pth")

    return modelE, modelD


if __name__ == "__main__":
    args = get_args()

    print("Starting VAE training on MNIST data...")

    dataset = MNISTDigits(
        list(range(10)) if args.digits is None else args.digits, 
        number_of_samples=3000,
        train=True
    )
    dataloader = pt.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    modelE, modelD = train(dataloader, latent_dim=args.latent_dim, lr=args.lr, max_epochs=args.epochs)

    print("Creating 10x10 grid of samples...")
    N = 10
    # Plot standard normal Gaussian z
    z = pt.randn((N*N, args.latent_dim)).to(next(modelD.parameters()).device)

    with pt.set_grad_enabled(False):
        X_pred = modelD(z)[0].cpu()

    save_image(((X_pred+1)/2), "Outputs/VAE_samples.png", nrow=N)
