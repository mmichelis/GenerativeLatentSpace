# ------------------------------------------------------------------------------
# Relative improvements on Riemannian straight line curve length sampled in 
# generator latent space.
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
from Models.VAE import Decoder
from Models.BiGAN import Generator
from Geometry.metric import InducedMetric
from Geometry.geodesic import trainGeodesic
from Geometry.curves import BezierCurve


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('gen', help="Name of generator architecture in 'Models/' directory.", choices=['VAE', 'BiGAN'])
    parser.add_argument('--trained_gen', help="Name of trained generator in 'Outputs/' directory.", type=str, default=None)
    parser.add_argument('--latent_dim', help="Dimension of latent space.", type=int, default=16)
    parser.add_argument('--M_batch_size', help="Batchsize for computation of metric.", type=int, default=1)
    parser.add_argument('--N', help="Number of endpoint sampled pairs in latent space.", type=int, default=50)
    parser.add_argument('--step_size', help="Step size in direction of maximal eigenvector.", type=float, default=1.0)
    parser.add_argument('--epochs', help="Number of epochs to train the shorter curve.", type=int, default=20)
    args = parser.parse_args()

    if not os.path.exists("Outputs"):
        os.makedirs("Outputs")

    device = pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu')

    # Need to manually set bc anyways.
    X_dim = [1,28,28]
    
    # Discretization of geodesic curve
    N_t = 16

    # Histogram precision on x-axis
    divisions = 30
    
    if args.gen == "VAE":
        modelG = Decoder(X_dim, args.latent_dim)
        args.trained_gen = args.trained_gen if args.trained_gen is not None else "trainedVAE_D.pth"
    elif args.gen == "BiGAN":
        modelG = Generator(X_dim, args.latent_dim)
        args.trained_gen = args.trained_gen if args.trained_gen is not None else "trainedBiGAN_G.pth"

    modelG.load_state_dict(pt.load(os.path.join("TrainedModels", args.trained_gen)))
    modelG.to(device)
    modelG.eval()
    print("Generator loaded!")

    ### Create metric space for curvelengths
    metricSpace = InducedMetric(modelG, X_dim, args.latent_dim)


    ### Monte Carlo sampling in latent space
    point_pairs = []
    error_list = []
    relative_list = []

    start_time = time.time()
    for i in range(args.N):
        if i%2 == 0:
            print(f"\nProcessing sample [{i+1}/{args.N}]")
            print('-'*20)

        # Should be standard zero mean gaussian for VAE, but that's the ideal case.
        start_point = np.random.normal(np.zeros(args.latent_dim))

        with pt.set_grad_enabled(True):
            start = pt.tensor(start_point, dtype=pt.float32, device=device)

            ### Follow maximal eigenvector of metric
            eig, eigv = np.linalg.eig(metricSpace.M_valueAt(start, M_batch_size=args.M_batch_size)[0])

        # Don't go away from [0,0] origin, always towards (to prevent leaving the region)
        direction = eigv[:, np.argmax(eig)]
        sign = np.sign(np.dot(start_point, direction))
        end_point = start_point - args.step_size*sign*direction

        point_pairs.append([start_point, end_point])

        best_gamma, length_history = trainGeodesic(
            pt.tensor(start_point, dtype=pt.float32, device=device), 
            pt.tensor(end_point, dtype=pt.float32, device=device), 
            N_t, metricSpace, 
            M_batch_size=args.M_batch_size, 
            max_epochs=args.epochs, 
            val_epoch=5,
            verbose=1
        )

        straight_len = length_history[0]
        curve_len = min(length_history)
        error = (straight_len - curve_len).item()
        error_list.append(error)
        relative_improvement = (error / straight_len).item()
        relative_list.append(relative_improvement)

        
    point_pairs = np.stack(np.array(point_pairs), axis=0)

    timetaken = time.time() - start_time
    print(f"Took {timetaken:.4f} seconds (In minutes: {timetaken/60:.4f})")
    print(f"On average {timetaken/args.N:.4f} seconds per sample.")
    

    ### Plot histogram
    fig, ax = plt.subplots(figsize=(15,9))
    plt.hist([100*i for i in relative_list], divisions)
    plt.xlabel('Relative percentual improvement (%)')
    plt.ylabel('Number of occurences')
    fig.savefig(f"Outputs/relativeImprovements_{100*np.array(relative_list).mean():.2f}.png", bbox_inches='tight')


    ### Plot absolute length improvement
    fig, ax = plt.subplots(figsize=(12,12))
    plt.plot(error_list)
    plt.xlabel('Line number')
    plt.ylabel('Absolute Improvement')
    plt.plot([np.array(error_list).mean()]*args.N, c='gray', linestyle='-', markersize=1, label="Mean")
    plt.legend()
    fig.savefig("Outputs/absolute_improvements.png", bbox_inches='tight')