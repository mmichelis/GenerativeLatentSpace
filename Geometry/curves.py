# ------------------------------------------------------------------------------
# Bezier and Bspline curve definitions, including a trainable curve with 
# variable parameters.
# ------------------------------------------------------------------------------

import torch as pt
import numpy as np

from torch import nn
from scipy.special import binom
from scipy.integrate import solve_bvp

import warnings


def BezierCurve (control_points):
    """
    Creates a Bezier curve using all control points provided. Curve has the dimension of the input control_points.
    
    Arguments:
        control_points (torch.Tensor) [n, d] : n points to use for the curve, first element 
        should be start point and last element the end point.
        
    Returns:
        Python function taking t in [0,1] and returning a value on the curve [b, D] for batch size b.
        Python function returning the derivative at each point too.
        -- Not anymore needed // Python function for second derivative. Second derivative always exists if n>1 (which is the case for our endpoint interpolation).
    """   
    def bernstein (i, n):
        binomial = binom(n, i.cpu()).to(i.device)
        # If i has shape [n] and t has shape [b], result is [b, n]
        return (lambda t: binomial * t**i * (1-t)**(n-i))

    # Curve of order n has points P0 ... Pn, i.e. n+1 total points
    n = control_points.shape[0]-1
    
    b_0n = bernstein(pt.linspace(0, n, n+1).to(control_points.device), n)
    # Cannot be done using b_0n as case where t=1 would have divide by zeros. otherwise 0**0 is handled correctly.
    b_0n_1 = bernstein(pt.linspace(0, n-1, n).to(control_points.device), n-1)
    #b_0n_2 = bernstein(torch.linspace(0, n-2, n-1).to(control_points.device), n-2)
    
    # If computation too inefficient, could consider implementing explicit formula for Bezier.
    def curve(t):
        if isinstance(t, pt.Tensor):
            t = t.view(-1, 1)
        
        return (
            pt.matmul(b_0n(t), control_points), 
            n * pt.matmul(b_0n_1(t), control_points[1:] - control_points[:-1]),
            # n * (n-1) * torch.matmul(b_0n_2(t), control_points[2:] - 2*control_points[1:-1] + control_points[:-2])
        )
    
    return curve


def CubicBSpline (control_points, knot_vector=None):
    """
    Creates a Cubic B-spline using all control points provided. Order 4 B-spline. Requires at least 4 control points!
    
    Arguments:
        control_points (torch.Tensor) [n, d] : n control points to use for the curve, first element 
        should be start point and last element the end point.
        
    Returns:
        Python function taking t in [0,1] and returning a value on the curve [b, D] for batch size b.
        Python function returning the derivative at each point too.
        Python function for second derivative. Second derivative always exists if n>1 (which is the case for our endpoint interpolation).
    """  
    device = control_points.device
    num_control = control_points.shape[0]
    order = 4
    if num_control < order:
        assert False, "Not enough control points for cubic Bspline!"

    if knot_vector is None:
        # Cardinal Bspline if no knot vector is given.
        knot_vector = pt.Tensor(np.concatenate([[0]*3, np.linspace(0, 1, num_control-2), [1]*3]).reshape(1, -1)).to(device)
        # Total length of knot vector is (n+k). We want shape [1,n] such that subtraction will work later on.
    else:
        # Make sure it's a tensor on the right device and has the correct shape.
        knot_vector = pt.as_tensor(knot_vector).view(1, -1).to(device)

    def basis_func (knots, k, t):
        """
        Returns the basis function at control point i and of degree k and evaluated at t, a torch Tensor of shape [N,1].
        If this method is too slow, I can write down the whole formula for cubic Bsplines explicitly and implement it that way.
        
        Arguments:
            knots (torch.Tensor [1, n])
            k (int)
            t (torch.Tensor [N, 1])
        
        Returns:
            torch.Tensor [N, n]: the Bspline evaluated at given points for all knots
            torch.Tensor [N, n-1]: derivative of Bspline at given points for all knots
        """
        # Shape of knots is n+k, so n (active knots) is knots.shape - k
        n = knots.shape[1] - k
        
        if k == 1:
            return pt.where(pt.logical_and(t >= knots[:, :-1], t <= knots[:, 1:]), 
                            pt.ones([t.shape[0], n]).to(device), 
                            pt.zeros([t.shape[0], n]).to(device)
                        )
        else:
            # Such that total new knot_vector length is still n+k
            B_ik_1 = basis_func(knots[:, :-1], k-1, t)
            B_i1k_1 = basis_func(knots[:, 1:], k-1, t)

            # t - knots[:n] is a shape [N, n] matrix
            term1 = pt.where(knots[:, k-1:-1] - knots[:, :n] != 0, 
                                B_ik_1 * (t - knots[:, :n]) / (knots[:, k-1:-1] - knots[:, :n]), 
                                pt.zeros_like(B_ik_1))
            term2 = pt.where(knots[:, k:] - knots[:, 1:n+1] != 0, 
                                    B_i1k_1 * (knots[:, k:] - t) / (knots[:, k:] - knots[:, 1:n+1]), 
                                pt.zeros_like(B_i1k_1))
            
            res = term1+term2
            if knots.shape[1] == num_control+order:
                # We're in the highest loop
                return res, B_i1k_1[:,:-1]
         
            return res
        
        
    def curve(t):
        if not isinstance(t, pt.Tensor):
            t = pt.as_tensor(t)
        t = t.view(-1, 1).to(device)
        
        basis, dbasis = basis_func(knot_vector, order, t)
        #dbasis = basis_func(knot_vector[:, 1:-1], order-1, t)
        gamma = pt.matmul(basis, control_points)
        dgamma = (order-1) * pt.matmul(
            dbasis / (knot_vector[:, order:-1] - knot_vector[:, 1:num_control]), 
            (control_points[1:] - control_points[:-1])
        ) 

        return gamma, dgamma
    
    return curve
    

class trainableCurve (nn.Module):
    def __init__(self, start, end, max_nodes=10, bspline=True):
        """
        BSpline: Adding nodes to the curve happen in a binary fashion, we always add 2^n nodes such that knots stay the same, and we simply refine the curve by adding more points in between the previous knots.

        Arguments:
            max_nodes (int) : all node parameters are created at the start (such that 
                              they are trainable parameters of the module). Counts all nodes in between start and end (excluding both). For BSplines, it's bes that max_nodes is 2^n - 1, wastes no memory in that way.
        """
        super().__init__()
        # There is one UserWarning thrown for instantiating ParameterLists. Should be fixed by PyTorch soon?
        warnings.filterwarnings("ignore", category=UserWarning)

        self.start = nn.Parameter(start, requires_grad=False)
        self.end = nn.Parameter(end, requires_grad=False)
        self.bspline = bspline
        
        # Without ParameterList the entries within a Parameter are not moved to device.
        # Initialize first 2 points on a straight line, rest will be set later
        self.new_nodes = nn.ParameterList([nn.Parameter(self.start + (i+1)/3 * (self.end - self.start)) for i in range(max_nodes)])
        
        # Keep as list such that moving to device and adding nodes work as expected.
        # CubicBsplines require 2 new nodes, whereas Bezier can start with 1 node. 
        self.points = [self.start, self.new_nodes[0], self.new_nodes[1], self.end]
        self.nodecount = 2
        self.knot_vector = [0,0,0,0,1,1,1,1]
        
    def add_node(self):
        if (self.nodecount >= len(self.new_nodes)):
            assert False, "Not enough max_nodes to use for another node addition!"

        ### For Bezier -----------
        if not self.bspline:
            self.points = self.points[:-1] + [self.new_nodes[self.nodecount], self.end]
            self.nodecount += 1
        ### ----------------------
        
        ### For Bspline ----------
        else:
            # We change 4 control_points into 5 control_points on the knot interval that is largest (we halve it).
            # Find largest knot interval:
            k = np.argmax(np.array(self.knot_vector)[1:] - np.array(self.knot_vector)[:-1])
            new_knot = (self.knot_vector[k+1]+self.knot_vector[k])/2
            # Degree 3 curve
            p = 3

            # First we add the new control point
            if self.knot_vector[k+p] - self.knot_vector[k] == 0:
                import pdb; pdb.set_trace()
            with pt.set_grad_enabled(False):
                ratio = (new_knot - self.knot_vector[k]) / (self.knot_vector[k+p] - self.knot_vector[k])
                self.new_nodes[self.nodecount] *= 0
                self.new_nodes[self.nodecount] += (1-ratio)*self.points[k-1] + ratio*self.points[k]

                # We update the points from back to front in-place.
                for i in range(k-1, k-p, -1):
                    if self.knot_vector[i+p] - self.knot_vector[i] == 0:
                        import pdb; pdb.set_trace()
                    ratio = (new_knot - self.knot_vector[i]) / (self.knot_vector[i+p] - self.knot_vector[i])
                    self.points[i] *= ratio
                    self.points[i] += (1-ratio)*self.points[i-1]

            self.points.insert(k, self.new_nodes[self.nodecount])
            self.knot_vector.insert(k+1, new_knot)
            self.nodecount += 1
        ### ----------------------

 
    def forward(self, t):
        if not self.bspline:
            return BezierCurve(pt.stack(self.points))(t)
        else:
            return CubicBSpline(pt.stack(self.points), self.knot_vector)(t)
