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

n=64
A=torch.triu(10*torch.rand((n,n)))
S=triangular_sqrt_wavefront(A)
print(torch.linalg.norm(A-S@S,ord='fro'))