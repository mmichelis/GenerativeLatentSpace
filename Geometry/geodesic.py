# ------------------------------------------------------------------------------
# Training shorter curves than straight line.
# ------------------------------------------------------------------------------

import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import copy

import sys
sys.path.append('./')
from Geometry.curves import trainableCurve, BezierCurve


def trainGeodesic (bc0, bc1, N, metricSpace, M_batch_size=4, max_epochs=1000, val_epoch=10, verbose=2):
    """
    Finds a shorter curve from bc0 to bc1 (attention, this may not be symmetric) in the metricSpace. 

    Arguments:
        bc0 (torch.Tensor [latent_dim]) : starting point for interpolation.
        bc1 (torch.Tensor [latent_dim]) : end point of interpolation.
        N (int) : discretization of shorter curve.
        metricSpace (Geometry.metric.InducedMetric) : contains generator model 
            with jacobian computation.
        M_batch_size (int) : batchsize for computation of metric.
        val_epoch (int) : Defines when the curve is reset to optimal.
        verbose (int) : 0 is no plots nor prints, 1 is no plots but print outputs, 2 is both.

    Returns:
        best_gamma (func: [b, 1] -> [b, latent_dim]) : curve function mapping 
            scalar parameter to vector points in latent space. 
        length_history (list) : list of lengths during training of shorter curve.
    """
    ### Parameters for training
    lr_init = 1e0
    lr_gamma = 0.9
    max_nodecount = 10
    max_hardschedules = 5
    hardschedule_factor = 0.3

    # Have a validation set of points to use for validation. Let's use half of N while training.
    t_val = pt.linspace(0, 1, N)

    gamma = trainableCurve(bc0, bc1, max_nodes=max_nodecount)
    gamma.to(metricSpace.device)

    # Start with straight line
    best_gamma = copy.deepcopy(gamma)
    with pt.set_grad_enabled(False):
        res, diff = best_gamma(t_val.to(metricSpace.device))
        g = res.detach().cpu().numpy()
        dg = diff.detach().cpu().numpy()

    dt = t_val[1] - t_val[0]
    best_length = metricSpace.curveLength(dt, g, dg, M_batch_size=M_batch_size)
    straight_measure = metricSpace.curve_measure(g, dg, M_batch_size=M_batch_size)

    # Let tolerance depend on the length of straight line.
    length_tol = best_length/200.

    print(f"Straight curve length: {best_length:.3f}")
    print(f"Straight curve measure: {straight_measure:.3f}")

    optimizer = pt.optim.Adam(gamma.parameters(), lr=lr_init, weight_decay=1e-4)

    # Multiplies the given lr with lambda every call to the scheduler
    scheduler = pt.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: lr_gamma)

    hardSchedules = 0
    length_history = [best_length]
    for epoch in range(max_epochs):
        if (epoch+1) % val_epoch:
            # Training: best_gamma unchanged
            runGammaEpoch(gamma, optimizer, scheduler, t_val, metricSpace, M_batch_size=M_batch_size, train=True)
        else:
            # Validation
            length = runGammaEpoch(gamma, None, None, t_val, metricSpace, M_batch_size=M_batch_size, train=False)
            length_history.append(length)
            
            if verbose >= 1:
                print('-'*10)
                print(f"Learning rate: {optimizer.param_groups[0]['lr']:.5e}")
                print(f"Epoch[{epoch+1:04d}/{max_epochs}]: Length: {length:.3f}")

            length_improvement = best_length - length
            if length < best_length:
                # Store current best network for minimal length
                if verbose >= 1:
                    print("Found better curve!")
                best_gamma = copy.deepcopy(gamma)
                best_length = length
                
            if length_improvement < length_tol: 
                # In case the loss increases, we first wanna rapidly decrease lr before we add nodes. 
                # We restart from the best solution when adding nodes or decreasing LR
                if hardSchedules >= max_hardschedules:
                    ### New Node
                    if (best_gamma.nodecount >= max_nodecount):
                        print("Node limit reached!")
                        break
                    if verbose >= 1:
                        print("*** Adding node ***")
                    best_gamma.add_node()

                ### Set gamma, and Reset best_gamma so it isn't trained
                gamma = best_gamma
                best_gamma = copy.deepcopy(best_gamma)

                # Re-initialize the optimizer to only the gamma parameters with gradients
                optimizer = pt.optim.Adam(filter(lambda p: p.requires_grad, gamma.parameters()), lr=lr_init, weight_decay=1e-4)
                scheduler = pt.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: lr_gamma)

                if hardSchedules < max_hardschedules:
                    if verbose >= 1:
                        print("* Decreasing LR *")
                    hardSchedules += 1
                    optimizer.param_groups[0]['lr'] *= hardschedule_factor**hardSchedules
                else:
                    # Reset hardSchedules when adding new node
                    hardSchedules = 0
                
    
    with pt.set_grad_enabled(False):
        res, diff = best_gamma(t_val.to(metricSpace.device))

    curve_measure = metricSpace.curve_measure(res.detach().cpu().numpy(), diff.detach().cpu().numpy(), M_batch_size=M_batch_size)
    del res, diff

    print(f"New curve length: {best_length:.3f}")
    print(f"New curve measure: {curve_measure:.3f}")

    return best_gamma, length_history


def runGammaEpoch(gamma, optimizer, scheduler, t_val, metricSpace, M_batch_size=4, train=True):
    """
    During validation we do not perturb the curve parameter.
    """ 
    eps = 1e-6
    dt = (t_val[1] - t_val[0]).to(metricSpace.device)
    t = t_val.to(metricSpace.device)

    if train:
        gamma.train()

        # Slightly perturb all the t values, such that it isn't sampled at the same points every time.
        perturb = pt.normal(pt.zeros_like(t), 0.1*dt).to(metricSpace.device)
        t = pt.min(pt.max(t+perturb, 0*t), 0*t+1)
    else:
        gamma.eval()

    length = 0 
    gamma.zero_grad()
    for batch in range(0, t_val.shape[0], M_batch_size):
        # Grad necessary during validation as well for M computation
        with pt.set_grad_enabled(True):
            res_batch, diff_batch = gamma(t[batch:batch+M_batch_size])
            N = res_batch.shape[0]
            M = metricSpace.M_valueAt(res_batch)
            # Length minimized
            norm = pt.matmul(pt.matmul(diff_batch.view(N, 1, -1), M), diff_batch.view(N, -1, 1)).view(-1)

            loss = (dt**2) * norm.sum() 
            length += dt * pt.sqrt(norm.detach().cpu()+eps).sum()
            # When we don't backward during evaluation too, M clogs up the GPU memory.
            loss.backward()
            
    if train:
        optimizer.step()
        scheduler.step()

    return length.item()
    