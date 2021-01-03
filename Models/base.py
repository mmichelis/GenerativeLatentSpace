# ------------------------------------------------------------------------------
# Some utility functions used in the models.
# ------------------------------------------------------------------------------

import torch as pt
from torch import nn


class LinBlock (nn.Sequential):
    def __init__ (self, in_c, out_c):
        super().__init__()
        self.add_module('Linear', nn.utils.spectral_norm(nn.Linear(in_c, out_c)))
        self.add_module('BatchNorm', nn.BatchNorm1d(out_c, affine=True))
        self.add_module('Activation', nn.ELU())

class ConvBlock (nn.Sequential):
    def __init__ (self, in_c, out_c, kernel_size, stride=1):
        super().__init__()
        self.add_module('Convolution', nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size, stride)))
        self.add_module('BatchNorm', nn.BatchNorm2d(out_c, affine=True))
        self.add_module('Activation', nn.ELU())

class ConvTransposeBlock (nn.Sequential):
    def __init__ (self, in_c, out_c, kernel_size, stride=1):
        super().__init__()
        self.add_module('ConvTranspose', nn.utils.spectral_norm(nn.ConvTranspose2d(in_c, out_c, kernel_size, stride)))
        self.add_module('BatchNorm', nn.BatchNorm2d(out_c, affine=True))
        self.add_module('Activation', nn.ELU())

# Lambda module does not like to be torch.save 'd.
class Lambda (nn.Module):
    def __init__ (self, f):
        super().__init__()
        self.f = f
    def forward (self, x):
        return self.f(x)

        
def runEpoch (lossFunc, dataloader, optimizer, scheduler=None):
    """
    Default training loop that minimizes lossFunc for one epoch of data in dataloader.

    Arguments:
        lossFunc (func: inputdata -> loss) : operations on inputdata with gradients enabled,
            producing a loss to use in backward call. Inputdata should be cast to cuda in lossFunc explicitly.
    """
    epoch_loss = 0

    for X_input in dataloader:
        with pt.set_grad_enabled(True):
            loss = lossFunc(X_input)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    
    # Length of dataloader is the amount of batches, not the total number of data points
    epoch_loss /= len(dataloader.dataset) 

    return epoch_loss