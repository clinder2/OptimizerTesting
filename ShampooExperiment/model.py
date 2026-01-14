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
    def __init__(self, n, m, Y):
        super().__init__()
        self.n=n
        self.m=m
        self.Y=Y
        self.l1=nn.Linear(self.n, self.n, False)
        self.relu=nn.ReLU()
        self.lrelu=nn.LeakyReLU()
        self.tanh=nn.Tanh()
        self.l2=nn.Linear(self.n, 2*self.n, False)
        self.l3=nn.Linear(2*self.n, self.m, False)

    def forward(self, X):
        X=self.l1(X)
        #X=self.lrelu(X)
        #X=self.tanh(X)
        X=self.l2(X)
        X=self.l3(X)
        return X, torch.linalg.norm(X-self.Y)
    
class ComplicatedMLP(nn.Module):
    def __init__(self, n, m, Y):
        super().__init__()
        self.n=n
        self.m=m
        self.Y=Y
        self.l1=nn.Linear(self.n, self.n, False)
        self.relu=nn.ReLU()
        self.lrelu=nn.LeakyReLU()
        self.tanh=nn.Tanh()
        self.l2=nn.Linear(self.n, 4*self.n, False)
        self.l3=nn.Linear(4*self.n, 4*self.n, False)
        self.l4=nn.Linear(4*self.n, 4*self.n, False)
        self.l5=nn.Linear(4*self.n, self.m, False)

    def forward(self, X):
        X=self.l1(X)
        #X=self.lrelu(X)
        #X=self.tanh(X)
        X=self.lrelu(X)
        X=self.l2(X)
        X=self.l3(X)
        #X=self.tanh(X)
        X=self.lrelu(X)
        X=self.l4(X)
        #X=self.tanh(X)
        X=self.lrelu(X)
        X=self.l5(X)
        return X, torch.linalg.norm(X-self.Y)