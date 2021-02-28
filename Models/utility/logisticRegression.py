# ------------------------------------------------------------------------------
# PyTorch implementation of logistic regression.
# ------------------------------------------------------------------------------

import torch as pt
import numpy as np

import sys
sys.path.append('./')
from Models.utility.feature import FeatureMap
from dataloader import CustomData


class LogisticRegression (FeatureMap):
    """
    Simple logistic regression on data. Supervised discriminative linear classification.

    Arguments:
        K (int) : number of classes
        D (int) : dimension of data space x_i

    Trainable parameters:
        weight [K, D] : weight for each class.
        bias [K] : scalar bias for each class.
    """

    def __init__ (self, K, D):
        super().__init__(K)
        self.K = K
        self.D = D

        requires_grad = True
        self.weight = pt.nn.Parameter(pt.randn(K, D), requires_grad=requires_grad)
        self.bias = pt.nn.Parameter(pt.randn(K), requires_grad=requires_grad)


    def run (self, mode, X, y=None, max_epochs=1000, lr=1e-3, batch_size=32):
        """
        Runs a certain mode on the current model. 
        - Train: train the weights to fit the data.
        - Test: Check accuracy on test data.
        - Predict: Test the current model on some unknown data.

        Arguments:
            mode (String) : one of ['train', 'test', 'predict']
            X (torch.Tensor [N, D] or torch.utils.data.DataLoader) : if input data tensor, 
                    then y must be given. Otherwise it must be a dataloader containing the target labels as well.
            y (torch.Tensor [N]) : (opt) target labels for classification. Classes are 
                    ints in range [0, K-1]! Make sure it's mapped to this range and has dtype Long.

        Returns:
            loss_history (List) in TRAIN mode : History of every loss during 
                    every validation epoch.
            OR
            y_pred (torch.Tensor [N]) in TEST/PREDICT mode : all predicted 
                    class labels in range [0, K-1].
        """
        assert mode in ['train', 'test', 'predict'], "Invalid mode!"

        device = self.weight.device
        train = (mode=='train')

        ### Create dataloader and optimizers
        if isinstance(X, pt.Tensor):
            dataloader = pt.utils.data.DataLoader(CustomData(X, y), batch_size=batch_size, shuffle=(not train), num_workers=1, pin_memory=True)
        else:
            dataloader = X

        optimizer = pt.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = pt.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.9)

        tot_correct = 0.0
        y_pred = []
        loss_history = []

        ### Loop over all epochs
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            # Work in batches
            for X_batch, y_batch in dataloader:
                # Reshape any tensor into batch of 1D vectors.
                X_batch = X_batch.view(-1, self.D).to(device)
                # y_batch is empty list for predictions
                if len(y_batch) > 0:
                    y_batch = y_batch.to(device)

                with pt.set_grad_enabled(train):
                    optimizer.zero_grad()

                    ### Compute log posterior probability of every class for every input point
                    log_joint = pt.matmul(X_batch, self.weight.transpose(0,1)) + self.bias.view(-1, self.K)  # [Nb, K]
                    # May consider using exponential normalization, but logsumexp should have stable gradients
                    log_posterior = log_joint - pt.logsumexp(log_joint, dim=1, keepdim=True)

                    if train:
                        # Only take the posteriors from the target class
                        nll = -pt.gather(log_posterior, dim=1, index=y_batch.view(-1,1)).sum()

                        # Prevent gradients scaling with batch size.
                        loss = nll/X_batch.shape[0]

                        loss.backward()
                        optimizer.step()
                        epoch_loss += nll.item()
                    else:
                        pred = pt.argmax(log_posterior, dim=1)
                        if mode == 'test':
                            tot_correct += (pred == y_batch).sum()
                        y_pred.append(pred)

            if train:
                scheduler.step()
                epoch_loss /= len(dataloader.dataset)
                if epoch == 0 or (epoch+1) % 10 == 0:
                    print(f"Epoch [{epoch+1:03d}/{max_epochs}]: Loss {epoch_loss:.5f}")
                loss_history.append(epoch_loss)

            else:
                ### We don't need to loop over any epochs if not training.
                y_pred = pt.cat(y_pred, dim=0)

                if mode == 'test':
                    ### Compute metric
                    accuracy = tot_correct / len(dataloader.dataset)
                    print(f"Accuracy: {accuracy*100:.1f}")

                return y_pred

        # Return loss history for training mode
        return loss_history


    def forward (self, X):
        """
        Returns log posterior probabilities of shape [N, K] for each class in X of shape [N, D].
        """
        log_joint = pt.matmul(X.view(-1, self.D), self.weight.transpose(0,1)) + self.bias.view(-1, self.K)  # [Nb, K]
        log_posterior = log_joint - pt.logsumexp(log_joint, dim=1, keepdim=True)

        return log_posterior

