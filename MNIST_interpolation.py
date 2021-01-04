# ------------------------------------------------------------------------------
# Interpolation in latent space, by default for MNIST digits, but can be changed
# easily.
# ------------------------------------------------------------------------------

import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import os
import gc
import time

from argparse import ArgumentParser

import sys
sys.path.append('./')
import Models.base 
from dataloader import MNISTDigits
from evaluation import create_sequence, create_crosscorrelation
from Models.VAE import Decoder
from Models.BiGAN import Generator
from Geometry.metric import InducedMetric
from Geometry.geodesic import trainGeodesic
from Geometry.curves import BezierCurve


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('gen', help="Name of generator architecture in 'Models/' directory.", choices=['VAE', 'BiGAN'])
    parser.add_argument('--trained_gen', help="Name of trained generator in 'Outputs/' directory.", type=str, default=None)
    #parser.add_argument('--latent_dim', help="Dimension of latent space.", type=int, default=16)
    parser.add_argument('--M_batch_size', help="Batchsize for computation of metric.", type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists("Outputs"):
        os.makedirs("Outputs")

    device = pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu')

    # Need to manually set bc anyways.
    X_dim = [1,28,28]
    latent_dim = 16
    
    # Margin for zmin zmax domain of latent space
    margin = 3
    # Discretization of geodesic curve
    N_t = 20
    bc0 = -pt.ones(latent_dim)
    bc1 = pt.ones(latent_dim)
    
    # MNIST data
    #dataset = MNISTDigits(list(range(num_labels)), number_of_samples=int(args.N/num_labels), train=False)

    if args.gen == "VAE":
        modelG = Decoder(X_dim, latent_dim)
    elif args.gen == "BiGAN":
        modelG = Generator(X_dim, latent_dim)

    modelG.load_state_dict(pt.load(os.path.join("TrainedModels", args.trained_gen)))
    modelG.to(device)
    modelG.eval()
    print("Generator loaded!")
    
    # print("Starting RBF training...")
    # rbfNN = trainRBF(model, dataloader, latent_dim, X_dim, k=30, zeta=1e-1, curveMetric=2, max_epochs=50, batch_size=args.batch_size)
    # rbfNN.eval()

    ### Create metric space for curvelengths
    metricSpace = InducedMetric(modelG, X_dim, latent_dim)


    ### Find shorter path than straight line
    print("Optimizing for shorter path...")
    start = time.time()
    best_gamma, length_history = trainGeodesic(bc0, bc1, N_t, metricSpace, M_batch_size=args.M_batch_size, max_epochs=100, val_epoch=5)
    print(f"Optimization took {time.time()-start:.1f}s.")

    fig, ax1 = plt.subplots(figsize=(12,9))
    ax1.plot(length_history[1:], linewidth=2)
    ax1.set_xlabel('Epoch (x100)')
    ax1.set_ylabel('Length', color='blue')
    fig.savefig("Outputs/Length_History.png", bbox_inches='tight')
    plt.close(fig)

    # Plot shorter curve
    t_plot = pt.linspace(0, 1, 2*N_t).to(device).view(-1,1)
    dt = 1 / (2*N_t - 1)

    with pt.set_grad_enabled(False):
        straight_plot = BezierCurve(pt.stack([bc0, bc1]).to(device))(t_plot)[0].cpu().numpy()
        curve_plot = best_gamma(t_plot)[0].detach().cpu().numpy()


    ### Evaluate interpolation curves
    create_sequence(modelG, straight_plot, curve_plot, seq_length=20)
    create_crosscorrelation(modelG, straight_plot, curve_plot)

