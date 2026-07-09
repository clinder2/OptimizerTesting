from enum import IntEnum
import torch
from torch import optim as torch_optim
from CustomShampoo import CustomShampoo
from WhiteningShampoo import WhiteningShampoo
from SCIShampoo import SCIShampoo

class OPTS(IntEnum):
    S=0  #Shampoo
    CS=1 #Shampoo+chol
    WS=2 #Whitening Shampoo
    S_P2=3 #Shampoo P=2
    SCI=4 #SCIShampoo
    SGD=5 #Stochastic Gradient Descent

#OPTS=[S,CS,WS,S_P2,SCI,SGD]


def make_optimizer(optimizer_type: OPTS, params, hyperparams: dict, **kwargs):
    """Create an optimizer from OPTS and hyperparameters.

    Args:
        optimizer_type: one of the OPTS enum values.
        params: iterable of model parameters.
        hyperparams: dict containing 'lr' or 'learning_rate'.
        **kwargs: optional extra optimizer arguments.

    Returns:
        torch optimizer instance.
    """
    lr = hyperparams.get('lr', hyperparams.get('learning_rate'))
    if lr is None:
        raise ValueError("hyperparams must include 'lr' or 'learning_rate'")

    beta2 = hyperparams.get('beta2', 0.85)

    match optimizer_type:
        case OPTS.S:
            return CustomShampoo(W=params, lr=lr, chol=False, beta2=beta2, **kwargs)
        case OPTS.CS:
            return CustomShampoo(W=params, lr=lr, chol=True, beta2=beta2, **kwargs)
        case OPTS.WS:
            return WhiteningShampoo(groups=params, lr=lr, pure=True, beta2=beta2, **kwargs)
        case OPTS.S_P2:
            return CustomShampoo(W=params, lr=lr, chol=True, p=2, beta2=beta2, **kwargs)
        case OPTS.SCI:
            return SCIShampoo(W=params, lr=lr, beta2=beta2, **kwargs)
        case OPTS.SGD:
            return torch_optim.SGD(params, lr=lr, **kwargs)
        case _:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")