import torch
from distributed_shampoo.preconditioner.matrix_functions_types import CholeskyConfig, CoupledNewtonConfig, EigenConfig
from distributed_shampoo.preconditioner.shampoo_preconditioner_list import RootInvShampooPreconditionerList
from model import MatrixSimple
from shampoo.shampoo import Shampoo
from distributed_shampoo import AdamPreconditionerConfig, DistributedShampoo, RootInvShampooPreconditionerConfig
from CustomShampoo import CustomShampoo
import math

warmup_iters=20
learning_rate=.9
lr_decay_iters=180
min_lr = 6e-5

#@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

def get_lr(it):
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

n=5
m=5
A=torch.eye(m,n)
model = MatrixSimple(A) #A=torch.eye(m,n)
params = [p for p in model.parameters()]

optim_groups = [
    {'params': params, 'weight_decay': 0.0}
]
#shampoo = Shampoo(optim_groups) ###Google-research shampoo

# preconditioner=RootInvShampooPreconditionerList(
#     preconditioner_config=RootInvShampooPreconditionerConfig(
#         amortized_computation_config=CholeskyConfig(),
#     ),
#     block_info_list=[params, True],
#     block_list=params,
#     state=None
# )

preconditioner_config=RootInvShampooPreconditionerConfig(
    amortized_computation_config=CholeskyConfig(),
    #amortized_computation_config=CoupledNewtonConfig()
    #amortized_computation_config=EigenConfig()
)

shampoo = DistributedShampoo(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    epsilon=1e-1, #1e-2
    precondition_frequency=1,
    preconditioner_config=preconditioner_config,
)

shampoo=CustomShampoo(1e-3, params, chol=False) #basic custom Shampoo implementation, no kronecker factor optimization

max_iters=100
iter_num=0

while True:
    lr = get_lr(iter_num)
    for param_group in shampoo.param_groups:
        param_group['lr'] = lr
    G, L=model()
    #L.backward()
    shampoo.step()
    shampoo.zero_grad(set_to_none=True)
    #model.W.data=zeropower_via_newtonschulz5(model.W.data)
    #temp=zeropower_via_newtonschulz5(model.A)
    #print(model.A-model.W, torch.linalg.norm(model.A-model.W).item())
    #print(L.item())
    iter_num+=1
    if iter_num>max_iters:
        break
print(model.W, model.A-model.W, torch.linalg.norm(model.A-model.W).item())
#print(model.W.T@model.W)
#print(model.A, model.W)

# tensor([[ 0.2669,  0.0035,  0.0032,  0.0043],
#         [ 0.0035,  0.2655,  0.0019,  0.0061],
#         [ 0.0032,  0.0019,  0.2587, -0.0005],
#         [ 0.0043,  0.0061, -0.0005,  0.2657]])
# tensor([[ 0.2665,  0.0000,  0.0000,  0.0000],
#         [ 0.0065,  0.2652,  0.0000,  0.0000],
#         [ 0.0066,  0.0041,  0.2588,  0.0000],
#         [ 0.0088,  0.0123, -0.0007,  0.2660]])
# tensor(0.0001) tensor(38.5655)