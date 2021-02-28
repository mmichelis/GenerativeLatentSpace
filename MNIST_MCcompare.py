# ------------------------------------------------------------------------------
# Comparing relative improvement of linear interpolations in two generative 
# models' latent space.
# ------------------------------------------------------------------------------

import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import os
import time
import math

from argparse import ArgumentParser

import sys
sys.path.append('./')
import Models.base 
from dataloader import MNISTDigits
from Geometry.metric import InducedMetric
from Geometry.geodesic import trainGeodesic
from Geometry.curves import BezierCurve

import Models.VAE as VAE
import Models.BiGAN as BiGAN


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('model1', help="Name of first model architecture in 'Models/' directory.", choices=['VAE', 'BiGAN'])
    parser.add_argument('--enc1', help="Name of first trained encoder in 'Outputs/' directory.", type=str, default=None)
    parser.add_argument('--gen1', help="Name of first trained generator in 'Outputs/' directory.", type=str, default=None)
    parser.add_argument('--latent_dim1', help="Dimension of latent space for first model.", type=int, default=16)
    
    parser.add_argument('model2', help="Name of second model architecture in 'Models/' directory.", choices=['VAE', 'BiGAN'])
    parser.add_argument('--enc2', help="Name of second trained encoder in 'Outputs/' directory.", type=str, default=None)
    parser.add_argument('--gen2', help="Name of second trained generator in 'Outputs/' directory.", type=str, default=None)
    parser.add_argument('--latent_dim2', help="Dimension of latent space for second model.", type=int, default=16)

    parser.add_argument('--N', help="Number of endpoint sampled pairs in latent space.", type=int, default=50)
    parser.add_argument('--step_size', help="Step size in direction of maximal eigenvector.", type=float, default=0.5)
    parser.add_argument('--epochs', help="Number of epochs to train the shorter curve.", type=int, default=20)
    parser.add_argument('--M_batch_size', help="Batchsize for computation of metric.", type=int, default=1)
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

    ### First Model
    if args.model1 == "VAE":
        modelG1 = VAE.Decoder(X_dim, args.latent_dim1)
        modelE1 = VAE.Encoder(X_dim, args.latent_dim1)
        args.gen1 = args.gen1 if args.gen1 is not None else "trainedVAE_D.pth"
        args.enc1 = args.enc1 if args.enc1 is not None else "trainedVAE_E.pth"
    elif args.model1 == "BiGAN":
        modelG1 = BiGAN.Generator(X_dim, args.latent_dim1)
        modelE1 = BiGAN.Encoder(X_dim, args.latent_dim1)
        args.gen1 = args.gen1 if args.gen1 is not None else "trainedBiGAN_G.pth"
        args.enc1 = args.enc1 if args.enc1 is not None else "trainedBiGAN_E.pth"

    modelG1.load_state_dict(pt.load(os.path.join("TrainedModels", args.gen1)))
    modelG1.to(device)
    modelG1.eval()
    modelE1.load_state_dict(pt.load(os.path.join("TrainedModels", args.enc1)))
    modelE1.to(device)
    modelE1.eval()
    print("Model 1 loaded!")

    ### Second Model
    if args.model2 == "VAE":
        modelG2 = VAE.Decoder(X_dim, args.latent_dim2)
        modelE2 = VAE.Encoder(X_dim, args.latent_dim2)
        args.gen2 = args.gen2 if args.gen2 is not None else "trainedVAE_D.pth"
        args.enc2 = args.enc2 if args.enc2 is not None else "trainedVAE_E.pth"
    elif args.model2 == "BiGAN":
        modelG2 = BiGAN.Generator(X_dim, args.latent_dim2)
        modelE2 = BiGAN.Encoder(X_dim, args.latent_dim2)
        args.gen2 = args.gen2 if args.gen2 is not None else "trainedBiGAN_G.pth"
        args.enc2 = args.enc2 if args.enc2 is not None else "trainedBiGAN_E.pth"
        
    modelG2.load_state_dict(pt.load(os.path.join("TrainedModels", args.gen2)))
    modelG2.to(device)
    modelG2.eval()
    modelE2.load_state_dict(pt.load(os.path.join("TrainedModels", args.enc2)))
    modelE2.to(device)
    modelE2.eval()
    print("Model 2 loaded!")


    ### Create metric space for curvelengths
    metricSpace1 = InducedMetric(modelG1, X_dim, args.latent_dim1)
    metricSpace2 = InducedMetric(modelG2, X_dim, args.latent_dim2)

    
    ### Image pairs
    endpoints = []
    print("Creating Endpoint Pairs...")

    # MNIST data
    N_test_samples = 500
    digits = list(range(10))    # In case we did not train on all MNIST digits
    test_data = MNISTDigits(digits, number_of_samples=N_test_samples, train=False)

    fig = plt.figure(figsize=(args.N, 2))
    for i in range(args.N):
        start = test_data[np.random.randint(N_test_samples*len(digits))][0]
        end = test_data[np.random.randint(N_test_samples*len(digits))][0]
        while (end == start).all():
            end = test_data[np.random.randint(N_test_samples*len(digits))]
        
        endpoints.append(np.stack([start, end], axis=0))

        # More than 20 images in the sequence aren't readable anymore
        if i < 20:
            ax0 = plt.subplot2grid((2, args.N), (0, i))
            ax0.axis('off')
            ax1 = plt.subplot2grid((2, args.N), (1, i))
            ax1.axis('off')
            if i == args.N-1:
                ax0.text(30, 13, "Start", fontsize=9)
                ax1.text(30, 13, "End", fontsize=9)

            ax1.imshow(start.reshape(28,28), cmap='gray')
            ax0.imshow(end.reshape(28,28), cmap='gray')

    fig.savefig("Outputs/startendsamples.png", bbox_inches='tight')
    plt.close(fig)

    # [N, 2, 1, 28, 28] shape array
    endpoints = np.stack(endpoints, axis=0)


    ### Latent points
    latent_pairs1 = modelE1(pt.Tensor(endpoints).to(device).view(-1, 1, 28, 28))
    latent_pairs1 = latent_pairs1[0] if isinstance(latent_pairs1, tuple) else latent_pairs1
    latent_pairs1 = latent_pairs1.view(args.N, 2, -1).detach()

    latent_pairs2 = modelE1(pt.Tensor(endpoints).to(device).view(-1, 1, 28, 28))
    latent_pairs2 = latent_pairs2[0] if isinstance(latent_pairs2, tuple) else latent_pairs2
    latent_pairs2 = latent_pairs2.view(args.N, 2, -1).detach()


    ### Monte Carlo sampling in latent space
    relative_list1 = []
    relative_list2 = []
    t_plot = pt.linspace(0, 1, 2*N_t).to(device).view(-1,1)

    start_time = time.time()
    for i in range(args.N):
        if i%2 == 0:
            print(f"\nProcessing sample [{i+1}/{args.N}]")
            print('-'*20)

        print("\nModel 1")
        start_point, end_point = latent_pairs1[i]

        best_gamma, length_history = trainGeodesic(
            start_point, 
            end_point, 
            N_t, metricSpace1, 
            M_batch_size=args.M_batch_size, 
            max_epochs=args.epochs, 
            val_epoch=5,
            verbose=0
        )

        straight_len1 = length_history[0]
        curve_len1 = min(length_history)
        relative_improvement = ((straight_len1 - curve_len1) / straight_len1).item()
        relative_list1.append(relative_improvement)

        straight_plot1 = BezierCurve(latent_pairs1[i])(t_plot)[0].detach()
        g_plot1 = best_gamma(t_plot)[0].detach()


        print("\nModel 2")
        start_point, end_point = latent_pairs2[i]

        best_gamma, length_history = trainGeodesic(
            start_point, 
            end_point, 
            N_t, metricSpace2, 
            M_batch_size=args.M_batch_size, 
            max_epochs=args.epochs, 
            val_epoch=5,
            verbose=0
        )

        straight_len2 = length_history[0]
        curve_len2 = min(length_history)
        relative_improvement = ((straight_len2 - curve_len2) / straight_len2).item()
        relative_list2.append(relative_improvement)

        straight_plot2 = BezierCurve(latent_pairs2[i])(t_plot)[0].detach()
        g_plot2 = best_gamma(t_plot)[0].detach()


        ### Plot the sequences
        seq_length = 16
        N = 2*N_t
        straight_plot1 = straight_plot1[::math.ceil(N/seq_length)]
        g_plot1 = g_plot1[::math.ceil(N/seq_length)]
        straight_plot2 = straight_plot2[::math.ceil(N/seq_length)]
        g_plot2 = g_plot2[::math.ceil(N/seq_length)]

        seq_length = straight_plot1.shape[0]

        # figsize is W x H
        fig = plt.figure(figsize=(seq_length, 5))

        for point_num in range(seq_length):
            ax0 = plt.subplot2grid((5, seq_length), (0, point_num))
            ax0.axis('off')
            ax1 = plt.subplot2grid((5, seq_length), (1, point_num))
            ax1.axis('off')
            ax2 = plt.subplot2grid((5, seq_length), (3, point_num))
            ax2.axis('off')
            ax3 = plt.subplot2grid((5, seq_length), (4, point_num))
            ax3.axis('off')
            
            if point_num == seq_length-1:
                ax0.text(30, 13, f"Straight Curve 1 {straight_len1:.2f}", fontsize=9)
                ax1.text(30, 13, f"Straight Curve 2 {straight_len2:.2f}", fontsize=9)
                ax2.text(30, 13, f"Shorter Curve 1 {curve_len1:.2f}", fontsize=9)
                ax3.text(30, 13, f"Shorter Curve 2 {curve_len2:.2f}", fontsize=9)

            with pt.set_grad_enabled(False):
                out_curve1 = modelG1(g_plot1[point_num].view(1,-1))
                out_curve1 = out_curve1[0] if isinstance(out_curve1, tuple) else out_curve1
                out_curve1 = out_curve1.detach().cpu().numpy()
                out_straight1 = modelG1(straight_plot1[point_num].view(1,-1))
                out_straight1 = out_straight1[0] if isinstance(out_straight1, tuple) else out_straight1
                out_straight1 = out_straight1.detach().cpu().numpy()

                out_curve2 = modelG2(g_plot2[point_num].view(1,-1))
                out_curve2 = out_curve2[0] if isinstance(out_curve2, tuple) else out_curve2
                out_curve2 = out_curve2.detach().cpu().numpy()
                out_straight2 = modelG2(straight_plot2[point_num].view(1,-1))
                out_straight2 = out_straight2[0] if isinstance(out_straight2, tuple) else out_straight2
                out_straight2 = out_straight2.detach().cpu().numpy()


            ax0.imshow(out_straight1.reshape(28,28), cmap='gray')
            ax1.imshow(out_straight2.reshape(28,28), cmap='gray')
            ax2.imshow(out_curve1.reshape(28,28), cmap='gray')
            ax3.imshow(out_curve2.reshape(28,28), cmap='gray')

        axl = plt.subplot2grid((5, seq_length), (2, 0))
        axl.axis('off')
        axl.imshow(endpoints[i,0,0], cmap='gray')
        
        axr = plt.subplot2grid((5, seq_length), (2, seq_length-1))
        axr.axis('off')
        axr.text(30, 13, "Sampled Endpoints", fontsize=9)
        axr.imshow(endpoints[i,1,0], cmap='gray')

        fig.savefig(f"Outputs/interpolation_sequence_{i}.png", bbox_inches='tight')
        plt.close(fig)


    timetaken = time.time() - start_time
    print(f"Took {timetaken:.4f} seconds (In minutes: {timetaken/60:.4f})")
    print(f"On average {timetaken/args.N:.4f} seconds per sample.")


    ### Plot Comparison
    fig, ax = plt.subplots(figsize=(12,9))
    x = np.arange(args.N)
    width = 0.4
    ax.bar(x - width/2, [100*i for i in relative_list1], width, color='blue', label='Model 1')
    ax.bar(x + width/2, [100*i for i in relative_list2], width, color='green', label='Model 2')

    ax.set_ylabel('Relative Improvement (%)')
    ax.set_title('Line Number')
    ax.set_xticks(x)
    ax.legend()

    fig.savefig("Outputs/relativeImprovementsComparison.png", bbox_inches='tight')