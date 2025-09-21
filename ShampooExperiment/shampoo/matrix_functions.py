# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import torch
import scipy
from scipy.linalg.lapack import dtrtri, dpotrf, dgetrf, dgetri


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

  # temp=torch.linalg.cholesky_ex(mat_g)
  # mat_c=temp.L
  # value=temp.info
  # #value=0
  # if value==0:
  #   #mat_g=mat_c
  #   #p=2
  #   #alpha=-1.0/2
  #   inv,_=dtrtri(mat_c.numpy(), lower=1)
  #   inv2,_=dtrtri(mat_c.numpy(), lower=0)
  #   print(inv2)
  #   #print(torch.linalg.norm(identity-mat_c@torch.Tensor(inv)),torch.linalg.norm(identity-mat_c@torch.Tensor(inv2)))
  #   # lu, piv, _ = dgetrf(mat_c.numpy())
  #   # inv, _ = dgetri(lu, piv)
  #   #inv=torch.linalg.inv(mat_c)
  #   #return torch.Tensor(inv)
  #   return torch.Tensor(sqrtm_newton_schulz(torch.Tensor(inv)))

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
  z = (1 + p) / (2 * torch.norm(mat_g))
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
  #print(p)
  #print(torch.linalg.cond(mat_g))
  return mat_root

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

def newton_schulz_inverse(A, num_iterations, alpha=None):
    """
    Approximates the inverse of a batch of matrices using the Newton-Schulz iteration.

    Args:
        A (torch.Tensor): The input tensor of shape (..., N, N).
        num_iterations (int): The number of iterations to perform.
        alpha (float, optional): Scaling factor for the initial guess.
                                  If None, a heuristic value is computed.

    Returns:
        torch.Tensor: The approximate inverse of shape (..., N, N).
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("Input matrix must be square, potentially with batch dimensions.")
    
    N = A.shape[-1]
    I = torch.eye(N, dtype=A.dtype, device=A.device)
    
    # Heuristic for alpha if not provided
    if alpha is None:
        # Compute a stable initial guess. A small scalar can also work.
        # Alternatively, using A.mT @ A and an inverse power iteration can be more robust.
        # This simple scaling works for many well-conditioned matrices.
        # For more robustness, use a small scalar.
        alpha = 1.0 / (torch.linalg.norm(A, ord=1, dim=(-2, -1), keepdim=True) * 
                       torch.linalg.norm(A, ord=float('inf'), dim=(-2, -1), keepdim=True))
        
    X_k = alpha * A.transpose(-2, -1)
    
    for _ in range(num_iterations):
        X_k = X_k @ (2 * I - A @ X_k)
    
    return X_k

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