import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import math

def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    #return learning_rate
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

class MatrixSimple(nn.Module):
    def __init__(self, A, i):
        super().__init__()
        torch.manual_seed(i)
        self.A=torch.Tensor(A)
        mult=1 #1

        # random_mat1 = torch.randn(self.A.shape)
        # random_mat2 = torch.randn(self.A.shape)
        # U, _ = torch.linalg.qr(random_mat1)
        # Vt, _ = torch.linalg.qr(random_mat2)
        # s_values = torch.logspace(0, -7, steps=self.A.shape[0])  # ranges from 1.0 down to 1e-7
        # S = torch.diag(s_values)
        # self.W = nn.Parameter(U @ S @ Vt)

        self.W=nn.Parameter(mult*torch.randn(self.A.shape)+torch.eye(self.A.shape[0])) #torch.randn(self.A.shape)

    def forward(self):
        ### old
        #G=2*(self.W-self.A)

        P=self.A.T@self.A
        G=2*(P@self.W-P)
        # with torch.no_grad():
        #     self.W.grad=G
        return G, torch.linalg.norm((self.A@self.W-self.A)**2,ord='fro')
    

class MLP(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super().__init__()
        self.in_dimension=in_dimension
        self.out_dimension=out_dimension
        intermediate = (in_dimension+out_dimension)//2
        intermediate = 1*in_dimension #2
        #self.r1=torch.nn.RMSNorm(self.in_dimension,elementwise_affine=False)
        #self.r2=torch.nn.RMSNorm(2*self.in_dimension,elementwise_affine=False)
        self.l1=nn.Linear(self.in_dimension, self.in_dimension, True)
        self.relu=nn.ReLU()
        # self.lrelu=nn.LeakyReLU()
        self.tanh=nn.Tanh()
        self.l2=nn.Linear(self.in_dimension, intermediate, True)
        #self.additional=nn.Linear(intermediate, intermediate, False)
        self.l3=nn.Linear(intermediate, self.out_dimension, True)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X=self.l1(X)
        # X=self.r1(X)
        X=self.relu(X)
        #X=self.tanh(X)
        X=self.l2(X)
        #X=self.additional(X)
        X=self.relu(X)
        X=self.l3(X)
        # X=self.r1(X)
        #X=self.sigmoid(X)
        return X
    
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
        self.r1=torch.nn.RMSNorm(self.n,elementwise_affine=False)
        self.l2=nn.Linear(self.n, 4*self.n, False)
        self.r2=torch.nn.RMSNorm(4*self.n,elementwise_affine=False)
        self.l3=nn.Linear(4*self.n, 4*self.n, False)
        self.r3=torch.nn.RMSNorm(4*self.n,elementwise_affine=False)
        self.l4=nn.Linear(4*self.n, 4*self.n, False)
        self.r4=torch.nn.RMSNorm(4*self.n,elementwise_affine=False)
        self.l5=nn.Linear(4*self.n, self.m, False)

    def forward(self, X):
        X=self.l1(X)
        X=self.r1(X)
        #X=self.lrelu(X)
        #X=self.tanh(X)
        X=self.lrelu(X)
        X=self.l2(X)
        X=self.r2(X)
        X=self.l3(X)
        X=self.r3(X)
        #X=self.tanh(X)
        X=self.lrelu(X)
        X=self.l4(X)
        X=self.r4(X)
        #X=self.tanh(X)
        X=self.lrelu(X)
        X=self.l5(X)
        return X, torch.linalg.norm(X-self.Y)
    
class MLP2(nn.Module):
    def __init__(self, n, m, h):
        super().__init__()
        self.n=n
        self.m=m
        self.h=h
        self.l1=nn.Linear(self.n, self.h, False)
        self.lrelu=nn.LeakyReLU()
        #self.tanh=nn.Tanh()
        #self.s=nn.Sigmoid()
        self.l2=nn.Linear(self.h, self.h, False)
        self.l3=nn.Linear(self.h, self.m, False)

    def forward(self, X):
        X=self.l1(X)
        # X=self.r1(X)
        X=self.lrelu(X)
        X=self.l2(X)
        X=self.l3(X)
        # X=self.r1(X)
        return X