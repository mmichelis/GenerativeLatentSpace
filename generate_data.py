# ------------------------------------------------------------------------------
# Generates random normal data from given generator. Saves 2D images for now.
# If shape of input images are different from MNIST 28x28, need to change here
# manually.
# ------------------------------------------------------------------------------

import os
import torch as pt
import matplotlib.pyplot as plt

from argparse import ArgumentParser

import sys
sys.path.append('./')
import Models.base
from dataloader import MNISTDigits
from Models.VAE import Decoder
from Models.BiGAN import Generator


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('gen', help="Name of generator architecture in 'Models/' directory. None if you wish to generate MNIST test data.", choices=[None, 'VAE', 'BiGAN'], default=None)
    parser.add_argument('--trained_gen', help="Name of trained generator in 'TrainedModels/' directory.", type=str, default=None)
    parser.add_argument('--latent_dim', help="Dimension of latent space.", type=int, default=16)
    parser.add_argument('--N', help="Number of samples.", type=int, default=500)
    args = parser.parse_args()

    X_dim = [1,28,28]
    device = pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu')

    if args.gen is None:
        ### For now we just do MNIST digits dataset, can be changed to whatever you wish.
        # Check that N is divided by number of digits!
        num_labels = 10
        dataset = MNISTDigits(list(range(num_labels)), number_of_samples=int(args.N/num_labels), train=False)

        if not os.path.exists("Outputs/MNISTdigits"):
            os.makedirs("Outputs/MNISTdigits")

        for i in range(args.N):
            if i%50 == 0:
                print(f"Storing image [{i}/{args.N}]")
            fig = plt.figure(figsize=(12, 12))
            image = (dataset[i][0]+1)/2
            plt.imshow(image.squeeze(), cmap='gray')
            plt.savefig(f"Outputs/MNISTdigits/{i:04d}.png")
            plt.close(fig)

    else:
        if args.gen == "VAE":
            modelG = Decoder(X_dim, args.latent_dim)
            args.trained_gen = args.trained_gen if args.trained_gen is not None else "trainedVAE_D.pth"
        elif args.gen == "BiGAN":
            modelG = Generator(X_dim, args.latent_dim)
            args.trained_gen = args.trained_gen if args.trained_gen is not None else "trainedBiGAN_G.pth"

        modelG.load_state_dict(pt.load(os.path.join("TrainedModels", args.trained_gen)))
        modelG.to(device)
        modelG.eval()

        z = pt.randn((args.N, args.latent_dim)).to(device)

        with pt.set_grad_enabled(False):
            X_pred = modelG(z)
            # For VAE we get mean and logvar as output
            if isinstance(X_pred, tuple):
                X_pred = X_pred[0]
            X_pred = X_pred.detach().cpu().squeeze().numpy()

        if not os.path.exists(f"Outputs/{args.gen}"):
            os.makedirs(f"Outputs/{args.gen}")

        for i in range(args.N):
            if i%50 == 0:
                print(f"Storing image [{i}/{args.N}]")
            fig = plt.figure(figsize=(12, 12))
            plt.imshow((X_pred[i]+1)/2, cmap='gray')
            plt.savefig(f"Outputs/{args.gen}/{i:04d}.png")
            plt.close(fig)
