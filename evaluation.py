# ------------------------------------------------------------------------------
# Methods for evaluating generative models.
# ------------------------------------------------------------------------------

import math
import torch as pt
import numpy as np
import matplotlib.pyplot as plt


def create_sequence (model, straight_plot, curve_plot, seq_length=None):
    """
    Compares straight line vs shorter curve side by side. Rounds towards seq_length, round down. So if we wanted 10 sequence length, but our curves had 14 points, then we get 7 in the end (14/10 = 1.4 -> rounded up 2, stepsize 2 [0:14:2]).

    Arguments:
        model (nn.Module) : generative model creating images from latent vectors.
        straight_plot (np.ndarray [N, latent_dim]) : array of points on a 
            straight line in latent space.
        curve_plot (np.ndarray [N, latent_dim]) : array of points on a 
            shorter curve in latent space.
        seq_length (int) : length of image sequence to create, if None then it's 
            the length of straight_plot/curve_plot.

    Returns:
        None, creates a figure and stores it at "Outputs/interpolation_sequence.png".
    """
    print("Creating interpolation sequence...")

    if seq_length is not None:
        N = straight_plot.shape[0]
        straight_plot = straight_plot[::math.ceil(N/seq_length)]
        curve_plot = curve_plot[::math.ceil(N/seq_length)]

    seq_length = straight_plot.shape[0]
    device = next(model.parameters()).device

    # figsize is W x H
    fig = plt.figure(figsize=(seq_length, 2))

    for point_num in range(seq_length):
        ax0 = plt.subplot2grid((2, seq_length), (0, point_num))
        ax0.axis('off')
        ax1 = plt.subplot2grid((2, seq_length), (1, point_num))
        ax1.axis('off')
        if point_num == seq_length-1:
            ax0.text(30, 13, "Straight Curve", fontsize=9)
            ax1.text(30, 13, "Shorter Curve", fontsize=9)

        curve_point = curve_plot[point_num]
        straight_point = straight_plot[point_num]

        with pt.set_grad_enabled(False):
            out_straight = model(pt.Tensor(straight_point).to(device).view(1,-1))
            if isinstance(out_straight, tuple):
                out_straight = out_straight[0]
            out_straight = out_straight.detach().squeeze().cpu().numpy()

            out_curve = model(pt.Tensor(curve_point).to(device).view(1,-1))
            if isinstance(out_curve, tuple):
                out_curve = out_curve[0]
            out_curve = out_curve.detach().squeeze().cpu().numpy()

        ax0.imshow(out_straight, cmap='gray')
        ax1.imshow(out_curve, cmap='gray')

    fig.savefig("Outputs/interpolation_sequence.png", bbox_inches='tight')
    plt.close(fig)


def create_crosscorrelation (model, straight_plot, curve_plot, featureMapping=None):
    """
    Cross correlation of outputs from both curves. Currently implemented as elementwise dot product between images.

    Arguments:
        model (nn.Module) : generative model creating images from latent vectors.
        straight_plot (np.ndarray [N, latent_dim]) : array of points on a 
            straight line in latent space.
        curve_plot (np.ndarray [N, latent_dim]) : array of points on a 
            shorter curve in latent space.

    Returns:
        None, creates a figure and stores it at "Outputs/cross_correlation.png".
    """
    print("Creating cross-correlation...")
    device = next(model.parameters()).device
    N = straight_plot.shape[0]

    # figsize is W x H
    curve_list = []
    straight_list = []
    for point_num in range(N):
        curve_point = curve_plot[point_num]
        straight_point = straight_plot[point_num]

        with pt.set_grad_enabled(False):
            out_curve = model(pt.Tensor(curve_point).to(device).view(1,-1))
            out_straight = model(pt.Tensor(straight_point).to(device).view(1,-1))

            if isinstance(out_straight, tuple):
                out_curve = out_curve[0]
                out_straight = out_straight[0]
                
            if featureMapping is not None:
                out_curve = featureMapping(out_curve)
                out_straight = featureMapping(out_straight)

            out_curve = out_curve.detach().squeeze().cpu().numpy().reshape(-1)
            out_straight = out_straight.detach().squeeze().cpu().numpy().reshape(-1)

        curve_list.append(out_curve)
        straight_list.append(out_straight)

    # Maybe we wanna convolve the whole lists
    curve_vec = np.stack(curve_list, axis=0)
    straight_vec = np.stack(straight_list, axis=0)
    # Normalize
    curve_vec /= np.linalg.norm(curve_vec, axis=1, keepdims=True)
    straight_vec /= np.linalg.norm(straight_vec, axis=1, keepdims=True)
    curve_corr = np.dot(curve_vec, np.transpose(curve_vec))
    straight_corr = np.dot(straight_vec, np.transpose(straight_vec))
    #signal.convolve2d(out_curve.reshape(28,28), out_straight.reshape(28,28))

    ### Plotting
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6, 12))
    vmin = min(np.amin(curve_corr), np.amin(straight_corr))
    vmax = max(np.amax(curve_corr), np.amax(straight_corr))
    # Could stick to [0,1] so we can compare images with one another
    levels = np.linspace(vmin, vmax, 15)

    im0 = ax0.contourf(np.linspace(1,N,N), np.linspace(1,N,N), straight_corr, levels=levels, cmap='jet')
    fig.colorbar(im0, ax=ax0)
    ax0.set_title("Straight Curve")

    im1 = ax1.contourf(np.linspace(1,N,N), np.linspace(1,N,N), curve_corr, levels=levels, cmap='jet')
    fig.colorbar(im1, ax=ax1)
    ax1.set_title("Shorter Curve")

    fig.savefig("Outputs/cross_correlation.png", bbox_inches='tight')
    plt.close(fig)

    ### Compute a measure based on cross-correlation
    straight_var = np.var(straight_corr)
    curve_var = np.var(curve_corr)
    print(f"Straight variance: {straight_var:.3f}")
    print(f"Curve variance: {curve_var:.3f}")
