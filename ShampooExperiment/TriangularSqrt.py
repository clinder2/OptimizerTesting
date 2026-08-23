from Base import *

def JacobiIter(P: torch.Tensor, num_iters=150):
    diag = torch.sqrt(torch.diagonal(P))
    denom = diag[:,None] + diag[None,:]
    S=torch.diag_embed(diag)

    for _ in range(num_iters):
        R=P-S@S
        Delta=torch.triu(R/denom, diagonal=1)
        S+=Delta

    return S

import torch


def triangular_sqrt_wavefront(A, radius=None):
    """
    Approximate upper-triangular square root S where S @ S ≈ A.

    Parameters
    ----------
    A : (..., n, n) upper triangular
    radius : int or None
        Number of previous superdiagonals used in the convolution.
        None -> exact Björck-Hammarling recurrence.

    Returns
    -------
    S : (..., n, n)
    """

    *batch, n, _ = A.shape
    S = torch.zeros_like(A)

    # exact diagonal
    diag = torch.sqrt(torch.diagonal(A, dim1=-2, dim2=-1))
    idx = torch.arange(n, device=A.device)
    S[..., idx, idx] = diag

    for k in range(1, n):

        rows = torch.arange(n - k, device=A.device)
        cols = rows + k

        value = A[..., rows, cols]

        if k > 1:

            if radius is None:
                r0 = 1
            else:
                r0 = max(1, k - radius)

            for r in range(r0, k):

                value = value - (
                    S[..., rows, rows + r]
                    * S[..., rows + r, cols]
                )

        value = value / (diag[..., rows] + diag[..., cols])

        S[..., rows, cols] = value

    return S

def Heron_Sqrt(A: torch.Tensor):
    # norm_A=torch.linalg.norm(A,ord='fro')
    # norm_A/=norm_A
    # A/=norm_A

    X=torch.sqrt(A)
    E=torch.zeros_like(A)
    I=torch.eye(A.shape[0])
    X=I
    for i in range(100):
        X=1/2*(X+torch.linalg.solve_triangular(X, I, upper=True)@A)
        # E=(A-X@X-X@E)@torch.linalg.solve_triangular((X+E), I, upper=True)
        # X+=E
        #print('X: ', X)
    return X


n=100
A=torch.triu(torch.rand((n,n)))+10*torch.eye(n)
print(A)
X=Heron_Sqrt(A)
print(X[0])
print(A[0])
print(torch.linalg.norm(A-X@X,ord='fro'))
# S=triangular_sqrt_wavefront(A)
# print(torch.linalg.norm(A-S@S,ord='fro'))