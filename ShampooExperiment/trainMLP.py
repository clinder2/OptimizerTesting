import torch
from distributed_shampoo.preconditioner.matrix_functions_types import CholeskyConfig, CoupledNewtonConfig, EigenConfig
from distributed_shampoo.preconditioner.shampoo_preconditioner_list import RootInvShampooPreconditionerList
from model import MatrixSimple, MLP
from shampoo.shampoo import Shampoo
from distributed_shampoo import AdamPreconditionerConfig, DistributedShampoo, RootInvShampooPreconditionerConfig
from CustomShampoo import CustomShampoo
import math

warmup_iters=20
learning_rate=.9
lr_decay_iters=180
min_lr = 6e-5

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

n=2
X=torch.eye(n)
Y=2*torch.eye(n)
torch.manual_seed(1)
model = MLP(n, Y)
params = [p for p in model.parameters()]

shampoo=CustomShampoo(1e-3, params, p=4, chol=True, optimized=False, debug=True) #basic custom Shampoo implementation, no kronecker factor optimization

max_iters=100
iter_num=0
temp=None

while True:
    lr = get_lr(iter_num)
    for param_group in shampoo.param_groups:
        param_group['lr'] = lr
    temp, L=model(X)
    L.backward()
    for p in model.parameters():
        print(p.grad)
    print(torch.linalg.norm(Y-temp, ord='fro').item())
    shampoo.step()
    shampoo.zero_grad(set_to_none=True)
    iter_num+=1
    if iter_num>max_iters:
        break
print(temp, torch.linalg.norm(Y-temp, ord='fro').item())