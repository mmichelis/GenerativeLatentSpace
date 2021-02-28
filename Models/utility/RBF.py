# ------------------------------------------------------------------------------
# Special file just for the improved variance estimate for VAEs as presented in
# 2018 G. Arvanitidis "Latent Space Oddity: on the Curvature of Deep Generative 
# Models" (https://arxiv.org/abs/1710.11379)
# ------------------------------------------------------------------------------

import torch as pt
import numpy as np


class RBF (pt.nn.Module):
    """
    Class to improve the variance of VAE
    """
    def __init__(self, centers, bandwidth, X_dim, zeta=1e-1):
        super().__init__()
        
        self.k = centers.shape[0]
        self.centers = pt.nn.Parameter(pt.Tensor(centers), requires_grad=False)
        self.bandwidth = pt.nn.Parameter(pt.Tensor(bandwidth), requires_grad=False)
        
        self.W = pt.nn.Linear(centers.shape[0], X_dim, bias=False)
        self.zeta = pt.nn.Parameter(pt.Tensor([zeta]), requires_grad=False)
        
        
    def forward(self, z_input):
        N = z_input.shape[0]
        latent_dim = z_input.shape[1]
        
        v = pt.exp(-self.bandwidth * pt.sum(
                (z_input.view(N, 1, latent_dim) - self.centers.view(1, self.k, latent_dim))**2, 
                axis=-1
                )
            )
        beta = self.W(v) + self.zeta
        
        return beta


def trainRBF (modelE, modelD, dataloader, latent_dim, X_dim, k, zeta=1e-3, curveMetric=1, max_epochs=100, batch_size=16):
    """
    Using trained VAE we now fit more accurate variance estimates using a RBF network.
    
    Arguments:
        modelE (torch.nn.Module) : Encoder of VAE.
        modelD (torch.nn.Module) : Decoder of VAE.
        dataloader (torch.nn.DataLoader) : training data used for VAE.
        latent_dim (int) : dimension of latent space.
        X_dim (np.ndarray) [C, H, W] : shape of input/output space.
        k (int) : number of clusters for k-means.
        zeta (float) : minimal precision (zeta > 0, at 0 the variance goes to infinity)
        curveMetric (float) : RBF bandwidth parameter, higher values create smoother variances, lower values make the RBF stick closer to the data (less smooth)
        max_epochs (int) : training of RBF

    Returns:
        Trained RBF (torch.nn.Module)
    """
    N = len(dataloader.dataset)
    # Keep a copy of the X_dim
    input_dim = X_dim
    device = next(modelD.parameters()).device
    modelE.eval(), modelD.eval()

    ### Compute embedded vectors
    with pt.no_grad():
        z_input = []
        kl_loss = []
        for X_input, _ in dataloader:
            X_input = X_input.to(device)
            zmean, zlogvar = modelE(X_input)
            z_input.append(zmean.detach().cpu().numpy())
            kl_loss.append((-0.5 * (1 + zlogvar - zmean**2 - zlogvar.exp()).sum(dim=1)).detach())
        
        z_input = np.concatenate(z_input, axis=0)
        kl_loss = pt.cat(kl_loss, dim=0).mean()

    ### Initialize the centers randomly between zmin and zmax with margin 1
    z_min = np.min(z_input.T, axis=1) - 1
    z_max = np.max(z_input.T, axis=1) + 1
    centers = np.random.uniform(z_min, z_max, size=[k, latent_dim])

    ### Generate the sets that belong to the same center, S shape [N]
    # These shaped subtractions: 
    # [N, 1, latent_dim] - [1, k, latent_dim] -> [N, k, latent_dim]
    # S contains the index of which set each point belongs to
    Sidx = np.argmin(np.sum((z_input.reshape(N, 1, latent_dim) - centers.reshape(1, k, latent_dim))**2, axis=-1), axis=-1)

    ### Repeat center assignment until S_idx does not change anymore
    print("Starting k-means: ")
    iterc = 0
    while (True):
        iterc+=1
        if (iterc % 10 == 0):
            print(f"Iteration {iterc}")
            
        for i in range(k):
            S_i = z_input[Sidx==i]
            if S_i.shape[0] == 0:
                # We don't want empty centers, randomize until we find non-empty
                centers[i] = np.random.uniform(z_min, z_max, size=latent_dim)
            else:
                centers[i] = np.sum(S_i, axis=0) / S_i.shape[0]
            
        Sidx_new = np.argmin(np.sum((z_input.reshape(N, 1, latent_dim) - centers.reshape(1, k, latent_dim))**2, axis=-1), axis=-1)
        
        if (Sidx == Sidx_new).all():
            break
        else:
            Sidx = Sidx_new

    S_shapes = []
    for i in range(k):
        S_i = z_input[Sidx==i]
        S_shapes.append(S_i.shape[0])
        
    # print(f"Number of points within each center_set: {S_shapes}")

    bandwidth = np.ones(k)

    for i in range(k):
        S_i = z_input[Sidx==i]
        if S_i.shape[0] == 0:
            # If there are no points in this center, it should have minimal influence 
            bandwidth[i] = 1e-3
        else:
            # Prevent that all S_i are exactly on the center
            eps = 1e-6
            bandwidth[i] = 0.5 * ( curveMetric / S_i.shape[0] * np.sum(np.sqrt(np.sum((S_i - centers[i])**2, axis=-1))+eps) )**-2

    print("Training RBF...")
    ### Start building network and clipper
    class PosClipper (object):
        def __call__(self, module):
            if hasattr(module, 'weight'):
                w = module.weight.data
                w.clamp_(0)

    if not isinstance(X_dim, int) and len(X_dim) > 1:
        # Not a scalar value
        input_dim = np.asarray(input_dim).prod()

    rbfNN = RBF(centers, bandwidth, input_dim, zeta)
    rbfNN.to(device)
    clipper = PosClipper()

    optimizerRBF = pt.optim.Adam(
                    rbfNN.parameters(),
                    lr=1e0,
                    weight_decay=1e-4
                )

    rbfNN.train()
    rbfNN.apply(clipper)
    for epoch in range(max_epochs):
        epoch_loss = 0
        shuffledIdx = pt.randperm(N)

        for i in range(0, N, batch_size):
            with pt.set_grad_enabled(True):
                modelD.eval()
                rbfNN.zero_grad()
                idx = shuffledIdx[i:i+batch_size]
                z = pt.Tensor(z_input[idx]).to(device)
                X = dataloader.dataset.data[idx].to(device)
            
                rbfVar = 1 / rbfNN(z)
                Xmean, Xlogvar = modelD(z)
                if not isinstance(X_dim, (int, np.int32, np.int64)) and len(X_dim) > 1:
                    Xmean = Xmean.view(-1, input_dim)
                    X = X.view(-1, input_dim)
                rec_loss = 0.5 * pt.log(rbfVar).sum(dim=1) + 0.5 * ((X - Xmean)**2 / rbfVar).sum(dim=1)
                rec_loss += (input_dim/2) * np.log(2*np.pi)

                loss = rec_loss.sum()
            
            loss.backward()
            optimizerRBF.step()
            rbfNN.apply(clipper)

            epoch_loss += loss.item()
        epoch_loss /= N
        # Add this constant value so it's the complete ELBO loss
        epoch_loss += kl_loss

        if (epoch % 10 == 0):
            print(f"Epoch [{epoch+1}/{max_epochs}]: Loss {epoch_loss:.4f}")
            
    rbfNN.eval()
    return rbfNN