import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class MatrixSimple(nn.Module):
    def __init__(self, A):
        super().__init__()
        torch.manual_seed(2)
        self.A=torch.Tensor(A)
        self.W=nn.Parameter(1*torch.randn(self.A.shape)+torch.eye(self.A.shape[0])) #torch.randn(self.A.shape)

    def forward(self):
        G=2*(self.W-self.A)
        with torch.no_grad():
            self.W.grad=G
        return G, torch.linalg.norm(self.W-self.A,ord='fro')
    

class MLP(nn.Module):
    def __init__(self, n, Y):
        super().__init__()
        self.n=n
        self.Y=Y
        self.l1=nn.Linear(self.n, self.n, False)
        self.l2=nn.Linear(self.n, self.n, False)

    def forward(self, X):
        X=self.l1(X)
        X=self.l2(X)
        return X, torch.linalg.norm(X-self.Y,ord='fro')