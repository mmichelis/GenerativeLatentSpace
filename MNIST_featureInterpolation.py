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

from Models.utility.logisticRegression import LogisticRegression
from Models.utility.VGG import VGG


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('gen', help="Name of generator architecture in 'Models/' directory.", choices=['VAE', 'BiGAN'])
    parser.add_argument('--trained_gen', help="Name of trained generator in 'TrainedModels/' directory.", type=str, default=None)
    parser.add_argument('--latent_dim', help="Dimension of latent space.", type=int, default=16)
    parser.add_argument('--M_batch_size', help="Batchsize for computation of metric.", type=int, default=1)
    parser.add_argument('--epochs', help="Number of epochs to train the shorter curve.", type=int, default=50)

    parser.add_argument('--mapping', help="Feature mapping to use: logreg (logistic regression) or VGG (relu5_2).", choices=['logreg', 'VGG'], default='logreg')
    args = parser.parse_args()

    if not os.path.exists("Outputs"):
        os.makedirs("Outputs")

    device = pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu')

    # Need to manually set bc anyways.
    X_dim = [1,28,28]
    
    # Discretization of geodesic curve
    N_t = 20
    bc0 = -pt.ones(args.latent_dim)
    bc1 = pt.ones(args.latent_dim)
    

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


    ### Feature Mapping
    featureMapping = None

    if args.mapping == 'logreg':
        try:
            logreg = torch.load("TrainedModels/trainedLogReg.pth")
            logreg.to(device)
            logreg.eval()
            print("Logistic Regression loaded!")
        except:
            print("Training Logistic Regression...")
            digits = list(range(10))    # In case we did not train on all MNIST digits

            dataloader = pt.utils.data.DataLoader(
                MNISTDigits(digits, number_of_samples=3000, train=True),
                batch_size=64,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
            test_dataloader = pt.utils.data.DataLoader(
                MNISTDigits(digits, number_of_samples=500, train=False),
                batch_size=64,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )

            logreg = LogisticRegression(K=len(digits), D=np.prod(X_dim)).to(device)
            logreg.run('train', dataloader, lr=1e-1, max_epochs=50)
            logreg.eval()
            logreg.run('test', test_dataloader)
            pt.save(logreg, "TrainedModels/trainedLogReg.pth")

        featureMapping = logreg
        
    elif args.mapping == 'VGG':
        vgg = VGG()
        vgg.load_state_dict(pt.load("TrainedModels/VGG_pretrained.pth")).to(device)
        vgg.eval()
        print("VGG loaded!")

        class Vgg_score(pt.nn.Module):
            def __init__(self, vgg):
                super().__init__()
                self.vgg = vgg
                self.K = 512

            def forward (self, x):
                #upsample = torch.nn.functional.interpolate(pt.stack([x,x,x], axis=1).view(-1, 3, 28, 28), [224,224])
                ### Just input 28x28 image, but stacked RGB ish. VGG input is [0,1] range.
                upsample = (pt.stack([x,x,x], axis=1).view(-1, 3, 28, 28) + 1) / 2

                return self.vgg(upsample, ['relu5_2'])[0].view(x.shape[0], -1)

        featureMapping = Vgg_score(vgg)


    ### Create metric space for curvelengths
    metricSpace = InducedMetric(modelG, X_dim, args.latent_dim, featureMapping=featureMapping)


    ### Find shorter path than straight line
    print("Optimizing for shorter path...")
    start = time.time()
    best_gamma, length_history = trainGeodesic(
        bc0, bc1, N_t, metricSpace, 
        M_batch_size=args.M_batch_size, 
        max_epochs=args.epochs, 
        val_epoch=5
    )
    print(f"Optimization took {time.time()-start:.1f}s.")

    fig, ax1 = plt.subplots(figsize=(12,9))
    ax1.plot(np.arange(len(length_history)-1), length_history[1:], linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Length', color='blue')
    fig.savefig("Outputs/Length_History.png", bbox_inches='tight')
    plt.close(fig)


    ### Plot shorter curve
    t_plot = pt.linspace(0, 1, 2*N_t).to(device).view(-1,1)
    dt = 1 / (2*N_t - 1)

    with pt.set_grad_enabled(False):
        straight_plot = BezierCurve(pt.stack([bc0, bc1]).to(device))(t_plot)[0].cpu().numpy()
        curve_plot = best_gamma(t_plot)[0].detach().cpu().numpy()


    ### Evaluate interpolation curves
    create_sequence(modelG, straight_plot, curve_plot, seq_length=20)
    create_crosscorrelation(modelG, straight_plot, curve_plot)

