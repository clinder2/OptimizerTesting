import sys
import os

ShampooExperiment_PATH = os.environ.get(
    "SHAMPOO_PATH", os.path.expanduser("~/Desktop/OptimizerTesting/ShampooExperiment")
)
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from torch.optim import Optimizer
from models.model import MatrixSimple
from common.TrainingScripts import make_target_param
from common.Base import make_optimizer, OPTS
from NewtonSchultz import NS
import numpy as np
import matplotlib.pyplot as plt
import time


def skew(A: torch.Tensor):
    return (A-A.mT)/2

def proj_T(Z: torch.Tensor, Y: torch.Tensor):
    I = torch.eye(Y.shape[1]).unsqueeze(0).repeat(Y.shape[0], 1, 1)
    return Y@skew(Y.mT@Z) + (I-Y@Y.mT)@Z

def stiefel_grad(G: torch.Tensor, Y: torch.Tensor):
    return G - Y@G.mT@Y
                

class StiefelGeod(Optimizer):
    def __init__(self, params, defaults={}):
        super().__init__(params, defaults)
        for g in self.param_groups:
            g['lr'] = defaults['lr']

    def step(self):
        for g in self.param_groups:
            n, p = g['params'][0].shape
            stacked_params = torch.stack([p.data for p in g['params']])
            stacked_grads = torch.stack([p.grad for p in g['params']])
            b = stacked_params.shape[0]
            #H = proj_T(stacked_grads, stacked_params)
            H = -stiefel_grad(stacked_grads, stacked_params)
            #H = -stacked_grads

            A = stacked_params.mT@H
            A=skew(A)
            Q, R = torch.linalg.qr(H - stacked_params@A)
            r1 = torch.concat([A, -R.mT], dim=2)
            r2 = torch.concat([R, torch.zeros([b,p,p])], dim=2)
            blocks = g['lr']*torch.concat([r1, r2], dim=1)
            MN = torch.linalg.matrix_exp(blocks)
            MN = MN[:,:,:p]
            
            Yt = stacked_params@MN[:,:p,:] + Q@MN[:,p:,:]
            Ht = H@MN[:,:p,:] - stacked_params@R.mT@MN[:,p:,:]

            for p, pt in zip(g['params'], Yt.unbind(0)):
                p.data = pt
            # for p, delta in zip(g['params'], Ht.unbind(0)):
            #     p.data -= delta

"""
modification of stiefgeod from https://arxiv.org/pdf/physics/9806030 to incorporate EMA of current gradient
with transported direction
"""
class StiefelGeodMomentum(Optimizer):
    def __init__(self, params, defaults={}):
        super().__init__(params, defaults)
        self.beta=.2
        for g in self.param_groups:
            g['lr'] = defaults['lr']
            g['Ht'] = None

    def step(self):
        for g in self.param_groups:
            n, p = g['params'][0].shape
            stacked_params = torch.stack([p.data for p in g['params']])
            stacked_grads = torch.stack([p.grad for p in g['params']])
            b = stacked_params.shape[0]
            H = -stiefel_grad(stacked_grads, stacked_params) #search direction
            if g['Ht']==None:
                g['Ht'] = H.clone()

            #EMA between current stiefel gradient H and current parallel transported direction Ht, 
            #computed and cached from previous iteration
            H = torch.lerp(H, g['Ht'], self.beta)
            
            A = stacked_params.mT@H
            A=skew(A)
            Q, R = torch.linalg.qr(H - stacked_params@A)
            r1 = torch.concat([A, -R.mT], dim=2)
            r2 = torch.concat([R, torch.zeros([b,p,p])], dim=2)
            blocks = g['lr']*torch.concat([r1, r2], dim=1)
            MN = torch.linalg.matrix_exp(blocks)
            MN = MN[:,:,:p]
            
            Yt = stacked_params@MN[:,:p,:] + Q@MN[:,p:,:]
            Ht = H@MN[:,:p,:] - stacked_params@R.mT@MN[:,p:,:] #parallel transport to get new search direction
            
            #Ht = -H + g['lr']*Ht
            g['Ht'] = Ht

            for p, pt in zip(g['params'], Yt.unbind(0)):
                p.data = pt

class StiefelNormal(Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        for g in self.param_groups:
            g['lr'] = .1
            g['piT_G']=torch.zeros_like(g['params'][0])

    def step(self):
        for g in self.param_groups:
            stacked_params = torch.stack([p.data for p in g['params']])
            stacked_grads = torch.stack([p.grad for p in g['params']])

            I = torch.eye(stacked_params.shape[1]).unsqueeze(0).repeat(stacked_params.shape[0], 1, 1)
            Vt = stacked_params@skew(stacked_params.mT@stacked_grads) + (I-stacked_params@stacked_params.mT)@stacked_grads
            Vt/=torch.linalg.norm(Vt, ord='fro', dim=(-1,-2))

            update = -g['lr']*Vt - g['lr']**2 * stacked_params@Vt.mT@Vt
            update = -g['lr']*(stacked_grads-stacked_params@stacked_grads.mT@stacked_params)
            
            update=g['lr']*Vt
            for p, delta in zip(g['params'], update.unbind(0)):
                p.data -= delta

"""
modification of stiefgeod from https://arxiv.org/pdf/physics/9806030 which will allow Y to travel off mainfold, 
try projection via newton-schultz back to manifold
"""
class StiefelGeodUnconstrained(Optimizer):
    def __init__(self, params, defaults={}):
        super().__init__(params, defaults)
        self.beta=.2
        for g in self.param_groups:
            g['lr'] = defaults['lr']
            g['Ht'] = None

    def step(self):
        for g in self.param_groups:
            n, p = g['params'][0].shape
            stacked_params = torch.stack([p.data for p in g['params']])
            #stacked_params = NS(stacked_params)
            stacked_grads = torch.stack([p.grad for p in g['params']])
            b = stacked_params.shape[0]
            H = -stiefel_grad(stacked_grads, stacked_params) #search direction
 
            A = stacked_params.mT@H
            A=skew(A)

            #NS approximation to Q
            # Q = NS(H - stacked_params@A)
            # R = Q.mT@(H - stacked_params@A)

            I = torch.eye(p).unsqueeze(0).repeat(stacked_params.shape[0], 1, 1)
            
            # Q = torch.eye(stacked_params.shape[1]).unsqueeze(0).repeat(stacked_params.shape[0], 1, 1)-stacked_params@stacked_params.mT
            # R = H

            #default
            Q, R = torch.linalg.qr(H - stacked_params@A)

            # r1 = torch.concat([A, -R.mT], dim=2)
            # r2 = torch.concat([R, torch.zeros([b,p,p])], dim=2)
            # blocks = g['lr']*torch.concat([r1, r2], dim=1)
            # MN = torch.linalg.matrix_exp(blocks)
            # MN = MN[:,:,:p]
            # M, N = MN[:,:p,:], MN[:,p:,:]

            M = I + g['lr']*A + (g['lr']**2 / 2)*(A@A - R.mT@R)
            N = g['lr']*R + (g['lr']**2 / 2)*R@A
            #M, N = NS(M), NS(N)
            
            Yt = stacked_params@M + Q@N
            Yt = NS(Yt)
            Ht = H@M - stacked_params@R.mT@N #parallel transport to get new search direction

            for p, pt in zip(g['params'], Yt.unbind(0)):
                p.data = pt

def train(name):
    rand_seed=0
    n=100
    spectrum = [0,-5]
    target=make_target_param(n, rand_seed, spectrum)

    model=MatrixSimple(target, rand_seed)
    model.W.data = torch.linalg.qr(model.W.data)[0]
    if torch.linalg.det(model.W.data)<0:
        model.W.data[:,0]*=-1
    print(torch.linalg.det(model.W.data))
    params=[p for p in model.parameters()]

    if name=="SG":
        optimizer=StiefelGeod(params, {'lr': .3})
    elif name=="SGM":
        optimizer=StiefelGeodMomentum(params, {'lr': .2})
    elif name=="SGU":
        optimizer=StiefelGeodUnconstrained(params, {'lr': .05})
    elif name=="MUON":
        optimizer=make_optimizer(OPTS.MUON, params, {'lr': .05})

    iters=200
    losses = []

    def get_lr(iter, lr):
        percent_done: float = iter/iters
        return lr * np.cos(percent_done*np.pi/2)

    s=time.time()
    for i in range(iters):
        lr = get_lr(i, optimizer.defaults['lr'])
        for g in optimizer.param_groups:
            g['lr'] = lr
        _, L = model()
        L.backward()
        losses.append(L.item())
        print(L.item(),  torch.allclose(model.W.T@model.W, torch.eye(n)))

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    e=time.time()
    print(e-s)
    plt.plot(np.arange(0,iters), np.log(losses), label=name)

#train("SG")
train("MUON")
train("SGM")
plt.legend()
plt.show()
# python3 -m StiefelOptimizer.StiefelNormal