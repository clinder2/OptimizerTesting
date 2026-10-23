import torch

def skew(A: torch.Tensor):
    return (A-A.mT)/2

def sym(A: torch.Tensor):
    return (A+A.mT)/2

def proj_T(Z: torch.Tensor, Y: torch.Tensor):
    I = torch.eye(Y.shape[1]).unsqueeze(0).repeat(Y.shape[0], 1, 1)
    return Y@skew(Y.mT@Z) + (I-Y@Y.mT)@Z

def stiefel_grad(G: torch.Tensor, Y: torch.Tensor):
    return G - Y@G.mT@Y