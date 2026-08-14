from enum import IntEnum
import sys
import os
import importlib.util
import torch
import torch.optim as opt

from StackedShampoo import StackedShampoo
from ExperimentalShampoo import ExperimentalShampoo

# Ensure local ShampooExperiment modules can be imported when this package
# is imported from a child process or from the workspace root.
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from CustomShampoo import CustomShampoo
from WhiteningShampoo import WhiteningShampoo
from SCIShampoo import SCIShampoo

# Explicitly load the local ShampooExperiment StiefelOptimizers wrapper by path.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
wrapper_path = os.path.join(BASE_DIR, "StiefelOptimizers.py")
if not os.path.exists(wrapper_path):
    raise FileNotFoundError(f"Cannot find local StiefelOptimizers.py at {wrapper_path}")

spec = importlib.util.spec_from_file_location("ShampooExperiment.StiefelOptimizers", wrapper_path)
StiefelOptimizers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = StiefelOptimizers
sys.modules["StiefelOptimizers"] = StiefelOptimizers
spec.loader.exec_module(StiefelOptimizers)

MuonAdamW = StiefelOptimizers.MuonAdamW
VariationalStiefelSGD = StiefelOptimizers.VariationalStiefelSGD
VariationalStiefelAdam = StiefelOptimizers.VariationalStiefelAdam

class OPTS(IntEnum):
    S=0  #Shampoo
    CS=1 #Shampoo+chol
    WS=2 #Whitening Shampoo
    S_P2=3 #Shampoo P=2
    SCI=4 #SCIShampoo
    SGD=5 #Stochastic Gradient Descent
    MUON=6
    STIEFEL_SGD=7
    STIEFEL_ADAM=8
    AdamW=9
    SCS=10 #stacked shampoo
    SS=11
    EXS=12

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

    # resolve beta2 (exponential average) from various possible keys
    def _resolve_beta2(hp, default=0.85):
        if 'beta2' in hp:
            return hp['beta2']
        if 'betas' in hp:
            try:
                return hp['betas'][1]
            except Exception:
                return default
        if 'beta' in hp:
            return hp['beta']
        return default

    beta2 = _resolve_beta2(hyperparams, 0.85)
    momentum = hyperparams.get('momentum', 0.9)
    weight_decay = hyperparams.get('weight_decay', 0.0)
    betas = hyperparams.get('betas', (0.9, 0.999))

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
            return opt.SGD(params, lr=lr, **kwargs)
        case OPTS.MUON:
            other_params=[]
            mat_params=[]
            for p in params:
                if len(p.shape)>=2:
                    mat_params.append(p)
                else:
                    other_params.append(p)
            param_groups=[]
            for shape in sorted({p.shape for p in other_params}):
                group_params = [p for p in other_params if p.shape == shape]
                param_groups.append(dict(
                    kind='adamw', params=group_params, lr=lr,
                    momentum=momentum, weight_decay=weight_decay,
                ))
            for shape in sorted({p.shape for p in mat_params}):
                group_params = [p for p in mat_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=lr,
                    momentum=momentum, ns_steps=5, beta2=beta2, weight_decay=weight_decay,
                ))
            # muon_groups = [{
            #     'params': params,
            #     'kind': 'muon',
            #     'lr': lr,
            #     'momentum': momentum,
            #     'ns_steps': 5,
            #     'beta2': beta2,
            #     'weight_decay': weight_decay,
            # }]
            return MuonAdamW(param_groups)
        case OPTS.STIEFEL_SGD:
            return VariationalStiefelSGD(params, lr=lr, momentum=momentum, **kwargs)
        case OPTS.STIEFEL_ADAM:
            return VariationalStiefelAdam(params, lr=lr, betas=betas, **kwargs)
        case OPTS.AdamW:
            return opt.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay, **kwargs)
        case OPTS.SCS:
            param_groups=[]
            for shape in sorted({p.shape for p in params}):
                group_params = [p for p in params if p.shape == shape]
                param_groups.append(dict(
                    kind='SCS', params=group_params, lr=lr,
                    beta2=beta2,
                ))
            return StackedShampoo(param_groups, chol=True, grafting=hyperparams['grafting'])
        case OPTS.SS:
            param_groups=[]
            for shape in sorted({p.shape for p in params}):
                group_params = [p for p in params if p.shape == shape]
                param_groups.append(dict(
                    kind='SS', params=group_params, lr=lr,
                    beta2=beta2,
                ))
            return StackedShampoo(param_groups, chol=False, grafting=hyperparams['grafting'])
        case OPTS.EXS:
            param_groups=[]
            for shape in sorted({p.shape for p in params}):
                group_params = [p for p in params if p.shape == shape]
                param_groups.append(dict(
                    kind='EXS', params=group_params, lr=lr,
                    beta2=beta2,
                ))
            return ExperimentalShampoo(param_groups)
        case _:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")