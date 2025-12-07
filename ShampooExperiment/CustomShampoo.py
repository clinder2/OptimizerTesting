import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
from typing import List

"""Simple Shampoo implementation for testing Cholesky factorization approximations
to speed up inverse matrix power computations. Runs default update L^{-1/4}@G_t@R^{-1/4}, 
experimental Cholesky update L'^{-1/4}@G_t@R'^{-1/4}, L'@L'.T=L, R'@R'.T=R, or scaled
up approximation from Anil et al. G_t@R^{-1/2}"""
class CustomShampoo(Optimizer):
    def __init__(self, lr, W, chol=False, optimized=False):
        data=dict(lr=lr)
        super().__init__(W, data)
        self.L=torch.eye(W[0].shape[0])
        self.R=torch.eye(W[0].shape[0])
        self.chol=chol
        self.opt=optimized

    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                grad=p.grad
                self.L+=grad@grad.T #update left/right preconditioners
                self.R+=grad.T@grad
                if self.opt:
                    update=grad@self.mat_pow(self.R, 1/2) #optimized approx. with 1 preconditioner
                else:
                    if self.chol:
                        Lp=torch.linalg.cholesky_ex(self.L)
                        Rp=torch.linalg.cholesky_ex(self.R)
                        if Lp.info==0 and Rp.info==0:
                            print('CHOL')
                            Lp=self.mat_pow(Lp.L, 1/2)
                            Rp=self.mat_pow(Rp.L, 1/2)
                            Rp=Rp
                        else:
                            Lp=self.mat_pow(self.L, 1/4)
                            Rp=self.mat_pow(self.R, 1/4)
                        update=Lp@grad@Rp
                    else:
                        Lp=self.mat_pow(self.L, 1/4)
                        Rp=self.mat_pow(self.R, 1/4)
                        update=Lp@grad@Rp
                p.data-=g['lr']*update
                #print(torch.linalg.norm(grad))

    def mat_pow(self, X, p):
        if p==1/4:
            return sqrtm_newton_schulz(inverse_sqrtm_newton_schulz(X))
        elif p==1/2:
            return inverse_sqrtm_newton_schulz(X)

    def zero_grad(self, set_to_none = True):
        super().zero_grad(set_to_none)

def inverse_sqrtm_newton_schulz(matrix: torch.Tensor, num_iters: int = 100):
    """
    Approximate the inverse square root of a matrix using the Newton-Schulz method.
    Adapted from https://discuss.pytorch.org/t/pytorch-square-root-of-a-positive-semi-definite-matrix/100138
    """
    if matrix.dim() < 2 or matrix.size(-1) != matrix.size(-2):
        raise ValueError("Input must be a square matrix or a batch of square matrices.")
    
    dim = matrix.size(-1)
    
    # Calculate the Frobenius norm of the matrix (batched)
    norm_of_matrix = torch.norm(matrix, p='fro', dim=(-2, -1), keepdim=True)
    
    # Normalize the matrix and initialize Y and Z
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
    Z = I.clone()

    # Newton-Schulz iteration
    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.matmul(Y))
        Y = Y.matmul(T)
        Z = T.matmul(Z)
    
    # Rescale and return the approximate inverse square root
    result = Z.div(torch.sqrt(norm_of_matrix))

    return result

def sqrtm_newton_schulz(A, num_iters=10):
    """
    Computes the matrix square root of a positive semi-definite matrix using
    the Newton-Schulz iterative method in PyTorch.

    Args:
        A (torch.Tensor): A batch of square positive semi-definite matrices
                          of shape (*, n, n).
        num_iters (int): The number of iterations to run the algorithm.

    Returns:
        torch.Tensor: The matrix square root of A.
    """
    
    # Ensure input matrix has at least two dimensions
    if A.dim() < 2:
        raise ValueError(f"Input dimension equals {A.dim()}, expected at least 2")
    
    # Get the batch dimensions and matrix size
    batch_dims = A.shape[:-2]
    n = A.size(-1)
    
    # Check if the matrix is square
    if A.size(-1) != A.size(-2):
        raise ValueError("Input matrices must be square.")
    
    # Scale the matrices by their Frobenius norm for stable convergence
    # Frobenius norm is computed over the last two dimensions
    norm_A = torch.norm(A, p='fro', dim=(-2, -1), keepdim=True)
    
    # Handle the case where norm is close to zero to avoid division issues
    norm_A = torch.where(norm_A < 1e-8, 1e-8 * torch.ones_like(norm_A), norm_A)
    
    # Normalize the matrix and prepare the inverse estimate
    Y = A / norm_A
    I = torch.eye(n, n, dtype=A.dtype).expand(*batch_dims, n, n)
    Z = I.clone()
    
    # Perform Newton-Schulz iterations
    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z@Y)
        Y = Y@T
        Z = T@Z
    
    # Apply the final scaling to get the actual square root
    return Y * torch.sqrt(norm_A)