# ------------------------------------------------------------------------------
# Training shorter curves than straight line.
# ------------------------------------------------------------------------------

import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import copy

from curves import trainableCurve, BezierCurve


def trainGeodesic (bc0, bc1, N, metricSpace, M_batch_size=4, max_epochs=1000, val_epoch=10, verbose=2):
    """
    Finds a shorter curve from bc0 to bc1 (attention, this may not be symmetric) in the metricSpace. 

    Arguments:
        bc0 (torch.Tensor) : starting point for interpolation.
        bc1 (torch.Tensor) : end point of interpolation.
        N (int) : discretization of shorter curve.
        metricSpace (Geometry.metric.InducedMetric) : contains generator model 
            with jacobian computation.
        M_batch_size (int) : batchsize for computation of metric.
        verbose (int) : 0 is no plots nor prints, 1 is no plots but print outputs, 2 is both.

    Returns:
        best_gamma (func: [b, 1] -> [b, latent_dim]) : curve function mapping 
            scalar parameter to vector points in latent space. 
        length_history (list) : list of lengths during training of shorter curve.
    """
    ### Parameters for training
    lr_init = 5e-1
    lr_gamma = 0.999
    length_tol = 1e-1
    max_nodecount = 5
    max_hardschedules = 2
    hardschedule_factor = 0.2
    MAX_PATIENCE = 20

    # Have a validation set of points to use for validation. Let's use half of N while training.
    t_val = pt.linspace(0, 1, N)

    gamma = trainableCurve(bc0, bc1, max_nodes=max_nodecount)
    gamma.to(metricSpace.device)

    # Start with straight line
    best_gamma = BezierCurve(pt.stack([bc0, bc1]).to(metricSpace.device))
    with pt.set_grad_enabled(False):
        res, diff = best_gamma(t_val.to(metricSpace.device))
        g = res.detach().cpu().numpy()
        dg = diff.detach().cpu().numpy()
        del res, diff
    dt = t_val[1] - t_val[0]
    best_length = metricSpace.curveLength(dt, g, dg, M_batch_size=M_batch_size)
    straight_measure = metricSpace.curve_measure(g, dg, M_batch_size=M_batch_size)
    

    print(f"Straight curve length: {best_length:.3f}")
    print(f"Straight curve measure: {straight_measure:.3f}")

    optimizer = pt.optim.Adam(gamma.parameters(), lr=lr_init, weight_decay=1e-4)

    # Multiplies the given lr with lambda every call to the scheduler
    scheduler = pt.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: lr_gamma)

    hardSchedules = 0
    patience = 0
    length_history = [best_length]
    for epoch in range(max_epochs):
        if epoch and ((epoch+1) % val_epoch):
            # Training: best_gamma unchanged
            length = runGammaEpoch(gamma, optimizer, scheduler, t_val, metricSpace, M_batch_size=M_batch_size, train=True)
        else:
            # Validation
            length = runGammaEpoch(gamma, None, None, t_val, metricSpace, M_batch_size=M_batch_size, train=False)
            length_history.append(length)
            
            if length < best_length:
                # Store current best network for minimal length
                if verbose >= 1:
                    print("Found better curve!")
                best_gamma = copy.deepcopy(gamma)
                best_length = length
                patience = 0
            else:
                patience += 1
                if patience > MAX_PATIENCE:
                    print("Got no patience no more")
                    break

            if verbose >= 1:
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                print(f"Epoch[{epoch+1:04d}/{max_epochs}]: Length: {length:.3f}")
                print('-'*10)


            ### Update control points in curve
            if len(length_history) > 1 and ((length_history[-2] - length) < length_tol): 
                # We first wanna rapidly decrease lr before we add nodes
                # Also when length increases          
                if hardSchedules < max_hardschedules:
                    if verbose >= 1:
                        print("* Decreasing LR *")
                    optimizer.param_groups[0]['lr'] *= hardschedule_factor
                    hardSchedules += 1
                
                else:
                    hardSchedules = 0
                    if (gamma.nodecount >= max_nodecount):
                        print("Node limit reached!")
                        break
                    if verbose >= 1:
                        print("*** Adding node ***")
                    gamma.add_node()
                    # Disable the next to last node
                    gamma.points[-3].requires_grad = False
                    # Re-initialize the optimizer to only the parameters with gradients
                    # Maybe not have as large of a learning rate as before.
                    optimizer = pt.optim.Adam(filter(lambda p: p.requires_grad, gamma.parameters()), lr=lr_init, weight_decay=1e-4)
                    scheduler = pt.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: lr_gamma)

    
    with pt.set_grad_enabled(False):
        res, diff = best_gamma(t_val.to(metricSpace.device))

    curve_measure = metricSpace.curve_measure(res.detach().cpu().numpy(), diff.detach().cpu().numpy(), M_batch_size=M_batch_size)
    del res, diff

    print(f"New curve length: {best_length:.3f}")
    print(f"New curve measure: {curve_measure:.3f}")

    return best_gamma, length_history


def runGammaEpoch(gamma, optimizer, scheduler, t_val, metricSpace, M_batch_size=4, train=True):
    """
    During validation the length is returned instead of loss (sqrt not taken for loss).
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

    # Grad necessary during validation as well for M computation
    with pt.set_grad_enabled(True):
        res, diff = gamma(t_val)

        length = 0  
        for batch in range(0, t_val.shape[0], M_batch_size):
            gamma.zero_grad()
            ### We want to have evaluated all t_val with old model, now update the model but use old res values.
            res_batch = res[batch:batch+M_batch_size]
            diff_batch = diff[batch:batch+M_batch_size]
            N = res_batch.shape[0]
            # Length minimized
            M = metricSpace.M_valueAt(res_batch)
            norm = pt.matmul(pt.matmul(diff_batch.view(N, 1, -1), M), diff_batch.view(N, -1, 1)).view(-1)
            if (norm==0).all():
                # Issue with backprop through 0 and pt.var
                variance=0
            else:
                variance = pt.var(norm)

            loss = (dt**2) * norm.sum() #+ 1e-5 * variance
            length += dt * pt.sqrt(norm.detach().cpu()+eps).sum()

            if train:
                # Gradients through gamma need to stay
                loss.backward(retain_graph=True)
                optimizer.step()

    if train:
        scheduler.step()

    return length
    