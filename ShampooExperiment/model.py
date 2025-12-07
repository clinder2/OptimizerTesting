import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class MatrixSimple(nn.Module):
    def __init__(self, A):
        super().__init__()
        torch.manual_seed(1)
        self.A=torch.Tensor(A)
        self.W=nn.Parameter(torch.randn(self.A.shape)+torch.eye(self.A.shape[0])) #torch.randn(self.A.shape)

    def forward(self):
        G=2*(self.W-self.A)
        with torch.no_grad():
            self.W.grad=G
        return G, torch.linalg.norm(self.W-self.A,ord='fro')