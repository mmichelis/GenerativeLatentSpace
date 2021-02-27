# ------------------------------------------------------------------------------
# Computing metric tensor and metric derivative at points in latent space. Also 
# includes computations that require the induced metric (e.g. curve length).
# ------------------------------------------------------------------------------

import torch as pt
import numpy as np
import gc


class InducedMetric:
    """
    Class combining the functionality for the metric tensor. 
    """
    def __init__(self, modelG, X_dim, latent_dim):
        self.modelG = modelG
        self.modelG.eval()
        self.device = next(modelG.parameters()).device

        self.X_dim = X_dim
        # Input_dim is the scalar dimension of the input, i.e. all dims multiplied if input is multi-dimensional.
        self.input_dim = X_dim
        if not isinstance(self.X_dim, int) and len(self.X_dim) > 1:
            self.input_dim = np.asarray(self.input_dim).prod()
        self.latent_dim = latent_dim


    def curveLength (self, dt, curve_points, curve_derivatives=None, M_batch_size=4):
        """
        For a discretized curve defined by N points we find the curve length. If analytic curve_derivatives can be computed, we use those, otherwise we use finite difference on the curve_points.
        
        Arguments:
            dt (float or np.ndarray) : Time difference. Scalar if the t are 
                uniformly distributed, otherwise vector of time differences (should have one padded value such that shape is [N])
            curve_points (np.ndarray) : Shape (N, d) where d is the coordinate 
                dimension of the curve.
            curve_derivatives (np.ndarray) : Shape (N, d) containing the 
                derivatives of the curve at each point.
            
        Returns:
            Scalar value representing the length. No gradients enabled by default.
        """
        N, d = curve_points.shape

        if curve_derivatives is None:
            z_upper = curve_points[1:]
            z_lower = curve_points[:-1]
            z_diff = z_upper - z_lower
            M = self.M_valueAt(pt.Tensor((z_upper + z_lower)/2).to(self.device), M_batch_size=M_batch_size)
            
            length = np.sqrt(np.matmul(np.matmul(z_diff.reshape(N-1, 1, d), M), z_diff.reshape(N-1, d, 1)).reshape(-1) + 1e-6).sum()
        
        else:
            M = self.M_valueAt(pt.Tensor(curve_points).to(self.device), M_batch_size=M_batch_size)
            length = dt * np.sqrt(np.matmul(np.matmul(curve_derivatives.reshape(N, 1, d), M), curve_derivatives.reshape(N, d, 1)).reshape(-1) + 1e-6).sum()

        return length


    def curve_measure (self, curve_points, curve_derivatives, M_batch_size=4):
        """
        Computes a measure to describe how much a curve follows the minimal eigenvectors of the induced metric tensor. Can show the improvement that is possible for a certain curve.
        """
        N = curve_points.shape[0]
        M = self.M_valueAt(pt.Tensor(curve_points).to(self.device), M_batch_size)
        derivative_norm = np.sqrt(np.sum(curve_derivatives**2, axis=-1, keepdims=True))

        eig, eigv = np.linalg.eig(M)
        eigS = np.min(eig, axis=-1)
        eigL = np.max(eig, axis=-1)
        eigSIdx = np.argmin(eig, axis=-1)
        # Shape of eigv is [N, d, d] where the COLUMNS of the [d,d] matrix are the eigenvectors of the corresponding eigenvalue.
        eigvS = np.take_along_axis(eigv, eigSIdx.reshape(-1,1,1), axis=-1).reshape(N,-1)

        condition_number = (eigL/eigS).reshape(N)
        scalar_prod = np.einsum('ij, ij -> i', curve_derivatives/(derivative_norm+1e-6), eigvS)
        # Would normally multiply with dt, but cancels out with normalization
        measure = np.sum(condition_number*abs(scalar_prod))
        normalized_measure = measure / np.sum(condition_number)

        return normalized_measure


    def M_valueAt(self, z, M_batch_size=None):
        """ 
        Computes the M = J.T * J value at a certain position in the latent space z.
        
        Arguments:
            z (torch.Tensor) : position in latent space. Shape [N, d]
            M_batch_size (int) : as computing the Jacobian and Hessian are very memory intensive, we may wish to use small batches instead. But this can only be used when no gradients are required!
            
        Returns:
            M (torch.Tensor or np.ndarray) : M matrix at z. Shape [N, d, d]. Numpy output when we use M_batch_size.
        """
        N = 1 if len(z.shape)==1 else z.shape[0]
        z = z.view(N, -1)
        
        if M_batch_size is not None:
            ### Loop over ourselves in batches. Detach every output and move to CPU.
            M_values = []
            for batch in range(0, N, M_batch_size):
                M_values.append(
                    self.M_valueAt(
                        z[batch: batch+M_batch_size]
                    ).detach().cpu().numpy()
                )
                gc.collect()
            # Could output torch Tensor here, but NumPy will prevent confusion (torch Tensor with no gradients and different device)
            return np.concatenate(M_values, axis=0)

        z_J = z.repeat_interleave(self.input_dim, dim=0)
        z_J.requires_grad_(True)
        
        X_pred = self.modelG(z_J)
        # For VAE we get mean and logvar as output
        if isinstance(X_pred, tuple):
            X_pred = X_pred[0]

        grad_outputs = pt.eye(self.input_dim).repeat(N,1).to(self.device)
        
        # Generate gradients
        J = pt.autograd.grad(outputs=X_pred.view(-1, self.input_dim), inputs=z_J,
                    grad_outputs=grad_outputs, create_graph=True, retain_graph=True,
                    only_inputs=True)[0].reshape(N, self.input_dim, self.latent_dim)

        # Prevent singular M matrices
        eps = 1e-6
        M = pt.matmul(pt.transpose(J, 1, 2), J)

        #del J, grad_outputs, X_pred

        return M + eps*pt.eye(self.latent_dim).to(self.device)

