"""Adapter module that imports Muon and Variational Stiefel optimizers
from local repositories and exposes lightweight factory wrappers.

This module does not reimplement algorithms; it looks for
`~/Desktop/nanochat` and `~/Desktop/VariationalStiefelOptimizer` and imports
the implementations found there. If those paths are not present on `sys.path`,
they are temporarily added.
"""
import sys
import os
import importlib
import importlib.util
from typing import Iterable

import torch

# Attempt to import Stiefel optimizer implementations from your local repos.
VAR_STIEFEL_PATH = os.path.expanduser("~/Desktop/VariationalStiefelOptimizer")
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Keep the original import exceptions for clearer diagnostics if import fails
STIEFEL_IMPORT_ERROR = None

StiefelSGD = None
StiefelAdam = None
try:
    external_path = os.path.join(VAR_STIEFEL_PATH, "StiefelOptimizers.py")
    if not os.path.exists(external_path):
        raise FileNotFoundError(f"Cannot find StiefelOptimizers.py in {VAR_STIEFEL_PATH}")

    spec = importlib.util.spec_from_file_location("external_StiefelOptimizers", external_path)
    external_mod = importlib.util.module_from_spec(spec)
    saved_sys_path = list(sys.path)
    if VAR_STIEFEL_PATH not in sys.path:
        sys.path.insert(0, VAR_STIEFEL_PATH)
    try:
        spec.loader.exec_module(external_mod)
    finally:
        sys.path[:] = saved_sys_path
    StiefelSGD = getattr(external_mod, "StiefelSGD", None)
    StiefelAdam = getattr(external_mod, "StiefelAdam", None)
except Exception as e:
    STIEFEL_IMPORT_ERROR = e

# MuonAdamW = importlib.import_module("/Users/christopherlinder/Desktop/stiefel-nanochat/train.py")
# print(MuonAdamW, " 2whi is it null")
# def MuonOptimizer(params: Iterable, lr: float = 1e-3, momentum: float = 0.9, ns_steps: int = 5, beta: float = 0.999, beta2: Optional[float] = None, weight_decay: float = 0.0):
#     """Factory that returns an instance of the Muon optimizer from your nanochat repo.

#     For the simple `MatrixSimple` experiment we put all parameters into a single
#     Muon group, unless the caller has already supplied parameter groups.
#     """
#     if MuonAdamW is None:
#         msg = f"Muon implementation not found. Tried paths: {NANOCHAT_PATH} (module 'nanochat.optim') or top-level 'optim'."
#         raise ImportError(msg) from MUON_IMPORT_ERROR
#     params_list = list(params)
#     beta2 = beta if beta2 is None else beta2
#     if params_list and isinstance(params_list[0], dict):
#         groups = params_list
#     else:
#         groups = [{
#             'params': params_list,
#             'kind': 'muon',
#             'lr': lr,
#             'momentum': momentum,
#             'ns_steps': ns_steps,
#             'beta2': beta2,
#             'weight_decay': weight_decay,
#         }]
#     return MuonAdamW(groups)

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    # Move scalars to correct device and dtype
    step_t = step_t.to(device=p.device, dtype=p.dtype)
    lr_t = lr_t.to(device=p.device, dtype=p.dtype)
    beta1_t = beta1_t.to(device=p.device, dtype=p.dtype)
    beta2_t = beta2_t.to(device=p.device, dtype=p.dtype)
    eps_t = eps_t.to(device=p.device, dtype=p.dtype)
    wd_t = wd_t.to(device=p.device, dtype=p.dtype)
    
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Move scalars to correct device and dtype
    momentum_t = momentum_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    lr_t = lr_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    wd_t = wd_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    beta2_t = beta2_t.to(device=stacked_params.device, dtype=stacked_params.dtype)

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    
    # Needs to match second_momentum_buffer.dtype for lerp_
    beta2_cast = beta2_t.to(second_momentum_buffer.dtype)
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2_cast)
    
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        print("***INIT MUONADAMW***")
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        
        # Compile only for CUDA. On Mac CPU/MPS, use the fallback Python kernels.
        device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        compiler_kwargs = {"dynamic": False, "fullgraph": True}
        if device_type == "cuda":
            self.adamw_step_fused = torch.compile(adamw_step_fused, **compiler_kwargs)
            self.muon_step_fused = torch.compile(muon_step_fused, **compiler_kwargs)
        else:
            self.adamw_step_fused = adamw_step_fused
            self.muon_step_fused = muon_step_fused

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            self.adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        self.muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)


def VariationalStiefelSGD(params: Iterable, lr: float = 1e-3, momentum: float = 0.9, dampening: float = 0, expm_method: str = 'ForwardEuler', inner_prod: str = 'Canonical', inner_iter: int = 10):
    """Factory that returns `StiefelSGD` from your VariationalStiefelOptimizer repo."""
    if StiefelSGD is None:
        msg = f"StiefelSGD implementation not found. Tried path: {VAR_STIEFEL_PATH} (modules 'StiefelOptimizers'/'stiefel_optimizers')."
        raise ImportError(msg) from STIEFEL_IMPORT_ERROR
    return StiefelSGD(list(params), lr=lr, momentum=momentum, dampening=dampening, expm_method=expm_method, inner_prod=inner_prod, inner_iter=inner_iter)


def VariationalStiefelAdam(params: Iterable, lr: float = 1e-3, betas=(0.9, 0.999), epsilon: float = 1e-5, expm_method: str = 'ForwardEuler', inner_prod: str = 'Canonical', inner_iter: int = 10):
    """Factory that returns `StiefelAdam` from your VariationalStiefelOptimizer repo."""
    if StiefelAdam is None:
        msg = f"StiefelAdam implementation not found. Tried path: {VAR_STIEFEL_PATH} (modules 'StiefelOptimizers'/'stiefel_optimizers')."
        raise ImportError(msg) from STIEFEL_IMPORT_ERROR
    return StiefelAdam(list(params), lr=lr, betas=betas, epsilon=epsilon, expm_method=expm_method, inner_prod=inner_prod, inner_iter=inner_iter)
