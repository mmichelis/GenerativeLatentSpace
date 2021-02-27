# ------------------------------------------------------------------------------
# PyTorch implementation of convolutional BiGAN (2016 J. Donahue "Adversarial
# Feature Learning" in https://arxiv.org/abs/1605.09782)
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
from Models.base import LinBlock, ConvBlock, ConvTransposeBlock, get_args
from dataloader import MNISTDigits


class Discriminator (nn.Module):
    """
    Discriminator of 2D GAN, producing a score for input images.

    Forward pass:
        1 Input:
            i) Image of shape [N, C, H, W] (by default 1x28x28 MNIST images)
        1 Output:
            i) Score of shape [N, 1]

    Arguments:
        X_dim (list) : dimensions of input 2D image, in the form of [Channels, Height, Width]
        latent_dim (int) : dimension of latent vector input.
    """
    def __init__(self, X_dim=[1,28,28], latent_dim=16):
        super(Discriminator, self).__init__()

        conv1_outchannels = 16
        conv2_outchannels = 32

        # How the convolutions change the shape
        conv_outputshape = (
            conv2_outchannels
            * int(((X_dim[1]-4)/2 - 2)/2 + 1)
            * int(((X_dim[2]-4)/2 - 2)/2 + 1)
        )

        self.Xconv = nn.Sequential(
            ConvBlock(X_dim[0], conv1_outchannels, kernel_size=4, stride=2),
            ConvBlock(conv1_outchannels, conv2_outchannels, kernel_size=3, stride=2)
        )

        lin1_outchannels = 16
        lin2_outchannels = 32

        self.zlin = nn.Sequential(
            LinBlock(latent_dim, lin1_outchannels),
            LinBlock(lin1_outchannels, lin2_outchannels)
        )

        cat_outchannels = 32

        self.post_cat = nn.Sequential(
            LinBlock(conv_outputshape+lin2_outchannels, cat_outchannels),
            nn.Linear(cat_outchannels, 1),
            nn.Sigmoid()
        )

    def forward (self, X, z):
        x1 = self.Xconv(X)
        x1 = x1.view(X.shape[0], -1)
        x2 = self.zlin(z)
        cat_input = pt.cat([x1, x2], dim=-1)
        x = self.post_cat(cat_input)

        return x


class Generator (nn.Module):
    """
    Generator of 2D GAN, producing image from latent vector.

    Forward pass:
        1 Input:
            i) Latent vector of shape [N, latent_dim]
        1 Output:
            i) Image of shape [N, C, H, W] (by default 1x28x28 MNIST images)

    Arguments:
        X_dim (list) : dimensions of input 2D image, in the form of [Channels, Height, Width]
        latent_dim (int) : dimension of latent vector input.
    """
    def __init__(self, X_dim=[1,28,28], latent_dim=16):
        super(Generator, self).__init__()

        conv1_outchannels = 16
        conv2_outchannels = 32

        # How the convolutions change the shape
        self.conv_outputshape = (
            int(((X_dim[1]-4)/2 - 2)/2 + 1),
            int(((X_dim[2]-4)/2 - 2)/2 + 1)
        )

        self.lin = nn.Linear(latent_dim, conv2_outchannels*self.conv_outputshape[0]*self.conv_outputshape[1])

        self.conv = nn.Sequential(
            ConvTransposeBlock(conv2_outchannels, conv1_outchannels, kernel_size=3, stride=2),
            ConvTransposeBlock(conv1_outchannels, 3, kernel_size=4, stride=2),
            nn.Conv2d(3, X_dim[0], kernel_size=1),
            nn.Tanh()
        )

    def forward (self, z):
        x = self.lin(z)
        x = x.view(z.shape[0], -1, self.conv_outputshape[0], self.conv_outputshape[1])
        x = self.conv(x)

        return x


class Encoder (nn.Module):
    """
    Encoder of 2D BiGAN, mapping an image back to latent space.

    Forward pass:
        1 Input:
            i) Image of shape [N, C, H, W] (by default 1x28x28 MNIST images)
        1 Output:
            i) Latent vector of shape [N, latent_dim]

    Arguments:
        X_dim (list) : dimensions of input 2D image, in the form of [Channels, Height, Width]
        latent_dim (int) : dimension of latent vector input.
    """
    def __init__(self, X_dim=[1,28,28], latent_dim=16):
        super(Encoder, self).__init__()

        conv1_outchannels = 16
        conv2_outchannels = 32

        # How the convolutions change the shape
        conv_outputshape = (
            conv2_outchannels
            * int(((X_dim[1]-4)/2 - 2)/2 + 1)
            * int(((X_dim[2]-4)/2 - 2)/2 + 1)
        )

        self.conv = nn.Sequential(
            ConvBlock(X_dim[0], conv1_outchannels, kernel_size=4, stride=2),
            ConvBlock(conv1_outchannels, conv2_outchannels, kernel_size=3, stride=2)
        )

        self.lin = nn.Linear(conv_outputshape, latent_dim)

    def forward (self, X):
        x = self.conv(X)
        x = x.view(X.shape[0], -1)
        x = self.lin(x)

        return x


### Training of a BiGAN
def train (dataloader, latent_dim=2, max_epochs=100, device=None):
    """
    Trains the BiGAN on data presented in dataloader for max_epochs. By default trained using Adam optimizer with learning rate 6e-5 and weight decay 1e-4, and having a multiplicative learning rate scheduler (0.95 multiplier per epoch).

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

    modelE = Encoder(X_dim, latent_dim).to(device)
    modelG = Generator(X_dim, latent_dim).to(device)
    modelD = Discriminator(X_dim, latent_dim).to(device)
    modelE.train(), modelG.train(), modelD.train()

    # Show the network architectures
    summary(modelE, X_dim)
    summary(modelG, (latent_dim,))
    # There is still a bug in torchsummary for multiple inputs
    try:
        summary(modelD, [X_dim, (latent_dim,)])
    except:
        pass

    optimGE = pt.optim.Adam(list(modelE.parameters())+list(modelG.parameters()), lr=5e-3, weight_decay=1e-4)
    optimD = pt.optim.Adam(modelD.parameters(), lr=3e-3, weight_decay=1e-4)

    schedulerGE = pt.optim.lr_scheduler.MultiplicativeLR(optimGE, lambda epoch: 0.95)
    schedulerD = pt.optim.lr_scheduler.MultiplicativeLR(optimD, lambda epoch: 0.95)

    ### Loop over epochs
    crit = nn.BCELoss()
    D_history = []
    G_history = []
    for epoch in range(max_epochs):
        ### Store an evaluation output in every epoch
        with pt.set_grad_enabled(False):
            randidx = np.random.randint(len(dataloader.dataset))
            output = modelG(
                modelE(
                    dataloader.dataset[randidx][0].to(device).unsqueeze(dim=0)
                )
            )[0].detach().cpu().squeeze().numpy()

            fig = plt.figure(figsize=(12, 12))
            # Image from [-1,1] to [0,1]
            plt.imshow((output+1)/2, cmap='gray')
            plt.savefig(f"Outputs/train_{epoch+1: 04d}.png")
            plt.close(fig)

        D_loss = 0
        G_loss = 0
        for X_input, _ in dataloader:
            X_input = X_input.to(device)
            real_scores = pt.ones([X_input.shape[0], 1]).to(device)
            fake_scores = pt.zeros([X_input.shape[0], 1]).to(device)

            with pt.set_grad_enabled(True):
                ### Train discriminator
                z = pt.randn([X_input.shape[0], latent_dim]).to(device)

                # Real images with real embedded latent vectors
                # Fake images with sample latent vector
                real_out = modelD(X_input, modelE(X_input))
                fake_out = modelD(modelG(z), z)

                lossD = crit(fake_out, fake_scores) + crit(real_out, real_scores)
                modelD.zero_grad()
                lossD.backward()
                optimD.step()

                ### Train encoder and generator
                z = pt.randn([X_input.shape[0], latent_dim]).to(device)
                real_out = modelD(X_input, modelE(X_input))
                fake_out = modelD(modelG(z), z)

                lossGE = crit(fake_out, real_scores) + crit(real_out, fake_scores)
                modelE.zero_grad(), modelG.zero_grad()
                lossGE.backward()
                optimGE.step()


            D_loss += lossD.item()
            G_loss += lossGE.item()

        D_loss /= len(dataloader)
        G_loss /= len(dataloader)

        schedulerD.step()
        schedulerGE.step()

        print(f"Epoch [{epoch+1}/{max_epochs}]: D Loss {D_loss:.4e}, G Loss {G_loss:.4e}")
        D_history.append(D_loss)
        G_history.append(G_loss)


        ### See how the loss evolved
        fig = plt.figure(figsize=(12,9))
        plt.plot(D_history, label='D_loss')
        plt.plot(G_history, label='G_loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(0, epoch+1)
        plt.legend()
        plt.grid(True)
        fig.savefig("Outputs/LossHistory.png", bbox_inches='tight')
        plt.close(fig)

        pt.save(modelD.state_dict(), "TrainedModels/trainedBiGAN_D.pth")
        pt.save(modelG.state_dict(), "TrainedModels/trainedBiGAN_G.pth")
        pt.save(modelE.state_dict(), "TrainedModels/trainedBiGAN_E.pth")

    return modelE, modelG


if __name__ == "__main__":
    args = get_args()

    print("Starting BiGAN training on MNIST data...")

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

    modelE, modelG = train(dataloader, latent_dim=args.latent_dim, max_epochs=args.epochs)

    print("Creating 10x10 grid of samples...")
    N = 10
    # Plot standard normal Gaussian z
    z = pt.randn((N*N, args.latent_dim)).to(next(modelG.parameters()).device)

    with pt.set_grad_enabled(False):
        X_pred = modelG(z).cpu()

    save_image(((X_pred+1)/2), "Outputs/BiGAN_samples.png", nrow=N)
