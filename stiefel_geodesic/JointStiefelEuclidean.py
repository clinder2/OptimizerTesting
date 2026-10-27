import sys
import os

ShampooExperiment_PATH = os.environ.get(
    "SHAMPOO_PATH", os.path.expanduser("~/Desktop/OptimizerTesting/ShampooExperiment")
)
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# `common` lives at the project root, alongside `analysis`, rather than inside
# this script's directory.
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.optim import Optimizer
import numpy as np

from stiefel_geodesic.StiefelUtil import proj_T, sym
from stiefel_geodesic.NewtonSchultz import NS

"""
modification of stiefgeod from https://arxiv.org/pdf/physics/9806030 which will allow Y to travel off mainfold, 
joint Stiefel-Euclidean optimization of position Y=XS decomposition
"""
class JointStiefelEuclidean(Optimizer):
    def __init__(self, params, defaults={}):
        super().__init__(params, defaults)
        for g in self.param_groups:
            g['lr'] = defaults['lr']
            g['St'] = None
            g['Xt'] = None

    def step(self):
        for g in self.param_groups:
            n, p = g['params'][0].shape
            stacked_params = torch.stack([p.data for p in g['params']])
            stacked_grads = torch.stack([p.grad for p in g['params']])
            b = stacked_params.shape[0]

            if g['St'] is None:
                g['St'] = torch.eye(p).unsqueeze(0).repeat(b, 1, 1)
            if g['Xt'] is None:
                g['Xt'] = torch.eye(n, p).unsqueeze(0).repeat(b, 1, 1)

            X_t = g['Xt']
            S_t = g['St']
            #X_t, S_t = torch.linalg.qr(stacked_params)
            g_X = proj_T(stacked_grads@S_t.mT, X_t)
            g_S = sym(X_t.mT@stacked_grads)

            S_next = S_t - g['lr']*g_S
            X_next = X_t - g['lr']*g_X
            X_next = (X_next)/(np.sqrt(1+g['lr']**2))
    
            Y_next = X_next@S_next

            g['Xt'] = X_next
            g['St'] = S_next

            for p, pt in zip(g['params'], Y_next.unbind(0)):
                p.data = pt
            return S_next, X_next