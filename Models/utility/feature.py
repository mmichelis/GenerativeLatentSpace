# ------------------------------------------------------------------------------
# Base class for feature mappings
# ------------------------------------------------------------------------------

import torch as pt

class FeatureMap (pt.nn.Module):
    # Only really necessary attribute is the out_dim currently.
    def __init__ (self, out_dim):
        super().__init__()
        self.out_dim = out_dim
