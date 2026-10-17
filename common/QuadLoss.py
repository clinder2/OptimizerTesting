import sys
import os

ShampooExperiment_PATH = os.environ.get(
    "SHAMPOO_PATH", os.path.expanduser("~/Desktop/OptimizerTesting/ShampooExperiment")
)
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from models.model import MatrixSimple
from common.TrainingScripts import make_target_param
from common.Base import make_optimizer, OPTS
from stiefel_geodesic.StiefelNormal import *
import numpy as np
import matplotlib.pyplot as plt
import time

def QuadExperiment(name):
    rand_seed=0
    n=100
    spectrum = [0,0]
    target=make_target_param(n, rand_seed, spectrum)

    model=MatrixSimple(target, rand_seed)
    model.W.data = torch.linalg.qr(model.W.data)[0]
    if torch.linalg.det(model.W.data)<0:
        model.W.data[:,0]*=-1
    print(torch.linalg.det(model.W.data))
    params=[p for p in model.parameters()]

    if name=="SG":
        optimizer=StiefelGeod(params, {'lr': .5})
    elif name=="SGM":
        optimizer=StiefelGeodMomentum(params, {'lr': .5})
    elif name=="SGU":
        optimizer=StiefelGeodUnconstrained(params, {'lr': .5})
    elif name=="MUON":
        optimizer=make_optimizer(OPTS.MUON, params, {'lr': .05})
    elif name=="SA":
        optimizer=make_optimizer(OPTS.STIEFEL_ADAM, params, {'lr': .05})

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