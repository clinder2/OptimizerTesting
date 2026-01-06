import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
from typing import List
from Grafting import AdagradGraft

"""Simple Shampoo implementation for testing Cholesky factorization approximations
to speed up inverse matrix power computations. Runs default update L^{-1/4}@G_t@R^{-1/4}, 
experimental Cholesky update L'^{-1/4}@G_t@R'^{-1/4}, L'@L'.T=L, R'@R'.T=R, or scaled
up approximation from Anil et al. G_t@R^{-1/2}"""
class CustomShampoo(Optimizer):
    def __init__(self, lr, W, p=4, chol=False, optimized=False, debug=False):
        data=dict(lr=lr)
        super().__init__(W, data)
        self.L=torch.eye(W[0].shape[0]) #left preconditioner
        self.R=torch.eye(W[0].shape[0]) #right preconditioner
        self.p=p #matrix power (4 for -1/4, 2 for -1/2, etc.)
        self.chol=chol #cholesky or not
        self.opt=optimized #Anil approximation or not
        self.state={}
        for g in self.param_groups:
            for p in g['params']:
                self.state[p]={} #init each parameter's state
                self.state[p]['graft']=AdagradGraft(None, p) #init Adagrad grafting
                self.state[p]['L']=torch.eye(p.shape[0])
                self.state[p]['R']=torch.eye(p.shape[1])
        self.debug=debug
        self.iter=0
        self.beta2=.9 #for L, R exp. decay

    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                graft=self.state[p]['graft']
                grad=p.grad
                L=self.state[p]['L']
                R=self.state[p]['R']
                L=self.beta2*L+(1-self.beta2)*grad@grad.T #update left/right preconditioners
                R=self.beta2*R+(1-self.beta2)*grad.T@grad
                self.state[p]['L']=L
                self.state[p]['R']=R
                if self.opt:
                    update=grad@self.mat_pow(self.state[p]['R'], 2) #optimized approx. with 1 preconditioner
                else:
                    if self.chol:
                        Lp=torch.linalg.cholesky_ex(self.state[p]['L']) #Cholesky decomp of L
                        Rp=torch.linalg.cholesky_ex(self.state[p]['R']) #Cholesky decomp of R
                        if Lp.info==0 and Rp.info==0: #successful Cholesky decomp
                            #print('CHOL', torch.linalg.norm(Lp.L))
                            Lp=self.mat_pow(Lp.L, 1/2*self.p)
                            Rp=self.mat_pow(Rp.L, 1/2*self.p) #.T better
                            Rp=Rp.T
                        else: #just standard Shampoo
                            Lp=self.mat_pow(self.state[p]['L'], self.p)
                            Rp=self.mat_pow(self.state[p]['R'], self.p)
                        update=Lp@grad@Rp
                    else: #just standard Shampoo
                        Lp=self.mat_pow(self.state[p]['L'], self.p)
                        Rp=self.mat_pow(self.state[p]['R'], self.p)
                        update=Lp@grad@Rp
                #p.data-=g['lr']*update
                graft.add_statistics(grad) #update grafting state
                graft_grad=graft.precondition_gradient(grad) #do grafting
                graft_n=torch.linalg.norm(graft_grad)
                shampoo_n=torch.linalg.norm(update)
                p.data-=g['lr']*(graft_n/(shampoo_n+1e-6))*update #param update with grafting
                if self.debug:
                   print(f"PRECONDITIONERS at {self.iter}:")
                   print("L: ", Lp.data)
                   print("R: ", Lp.data)
                   print(f"UPDATE at {self.iter}:")
                   print(update)
        self.iter+=1

    ### wrapper to call Scalable Shampoo ComputerPower for matrix inverses
    def mat_pow(self, X, p):
        return ComputePower(X, p)
        # if p==4:
        #     return sqrtm_newton_schulz(inverse_sqrtm_newton_schulz(X))
        # elif p==2:
        #     return inverse_sqrtm_newton_schulz(X)
        # else:
        #     return X

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

###
# Scalable Shampoo matrix power functions: 
# https://github.com/google-research/google-research/blob/master/scalable_shampoo/pytorch/matrix_functions.py 
###

@torch.no_grad()
def PowerIter(mat_g, error_tolerance=1e-6, num_iters=100):
  """Power iteration.

  Compute the maximum eigenvalue of mat, for scaling.
  v is a random vector with values in (-1, 1)

  Args:
    mat_g: the symmetric PSD matrix.
    error_tolerance: Iterative exit condition.
    num_iters: Number of iterations.

  Returns:
    eigen vector, eigen value, num_iters
  """
  v = torch.rand(list(mat_g.shape)[0]) * 2 - 1
  error = 1
  iters = 0
  singular_val = 0
  while error > error_tolerance and iters < num_iters:
    v = v / torch.norm(v)
    mat_v = torch.mv(mat_g, v)
    s_v = torch.dot(v, mat_v)
    error = torch.abs(s_v - singular_val)
    v = mat_v
    singular_val = s_v
    iters += 1
  return singular_val, v / torch.norm(v), iters

@torch.no_grad()
def MatPower(mat_m, p):
  """Computes mat_m^p, for p a positive integer.

  Args:
    mat_m: a square matrix
    p: a positive integer

  Returns:
    mat_m^p
  """
  if p in [1, 2, 4, 8, 16, 32]:
    p_done = 1
    res = mat_m
    while p_done < p:
      res = torch.matmul(res, res)
      p_done *= 2
    return res

  power = None
  while p > 0:
    if p % 2 == 1:
      power = torch.matmul(mat_m, power) if power is not None else mat_m
    p //= 2
    mat_m = torch.matmul(mat_m, mat_m)
  return power

@torch.no_grad()
def ComputePower(mat_g, p,
                 iter_count=100,
                 error_tolerance=1e-6,
                 ridge_epsilon=1e-6):
  """A method to compute G^{-1/p} using a coupled Newton iteration.

  See for example equation 3.2 on page 9 of:
  A Schur-Newton Method for the Matrix p-th Root and its Inverse
  by Chun-Hua Guo and Nicholas J. Higham
  SIAM Journal on Matrix Analysis and Applications,
  2006, Vol. 28, No. 3 : pp. 788-804
  https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf

  Args:
    mat_g: A square positive semidefinite matrix
    p: a positive integer
    iter_count: Stop iterating after this many rounds.
    error_tolerance: Threshold for stopping iteration
    ridge_epsilon: We add this times I to G, to make is positive definite.
                   For scaling, we multiply it by the largest eigenvalue of G.
  Returns:
    (mat_g + rI)^{-1/p} (r = ridge_epsilon * max_eigenvalue of mat_g).
  """
  #print("ComputePower: ", p)
  shape = list(mat_g.shape)
  if len(shape) == 1:
    return torch.pow(mat_g + ridge_epsilon, -1/p)
  identity = torch.eye(shape[0])
  if shape[0] == 1:
    return identity
  alpha = -1.0/p
  max_ev, _, _ = PowerIter(mat_g)
  ridge_epsilon *= max_ev
  mat_g += ridge_epsilon * identity
  z = (1 + p) / (2 * torch.norm(mat_g))
  # The best value for z is
  # (1 + p) * (c_max^{1/p} - c_min^{1/p}) /
  #            (c_max^{1+1/p} - c_min^{1+1/p})
  # where c_max and c_min are the largest and smallest singular values of
  # mat_g.
  # The above estimate assumes that c_max > c_min * 2^p
  # Can replace above line by the one below, but it is less accurate,
  # hence needs more iterations to converge.
  # z = (1 + p) / tf.trace(mat_g)
  # If we want the method to always converge, use z = 1 / norm(mat_g)
  # or z = 1 / tf.trace(mat_g), but these can result in many
  # extra iterations.

  mat_root = identity * torch.pow(z, 1.0/p)
  mat_m = mat_g * z
  error = torch.max(torch.abs(mat_m - identity))
  count = 0
  while error > error_tolerance and count < iter_count:
    tmp_mat_m = (1 - alpha) * identity + alpha * mat_m
    new_mat_root = torch.matmul(mat_root, tmp_mat_m)
    mat_m = torch.matmul(MatPower(tmp_mat_m, p), mat_m)
    new_error = torch.max(torch.abs(mat_m - identity))
    if new_error > error * 1.2:
      break
    mat_root = new_mat_root
    error = new_error
    count += 1
  return mat_root