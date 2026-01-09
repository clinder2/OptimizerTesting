import torch
from distributed_shampoo.preconditioner.matrix_functions_types import CholeskyConfig, CoupledNewtonConfig, EigenConfig
from distributed_shampoo.preconditioner.shampoo_preconditioner_list import RootInvShampooPreconditionerList
from model import MatrixSimple, MLP
from shampoo.shampoo import Shampoo
from distributed_shampoo import AdamPreconditionerConfig, DistributedShampoo, RootInvShampooPreconditionerConfig
from CustomShampoo import CustomShampoo
import numpy as np
import math
import time

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

trials=1
total_time=0.0
total_loss=0.0
for i in range(trials):
    n=10
    m=10
    torch.manual_seed(1)
    X=torch.rand(n)
    torch.manual_seed(6)
    Y=10*torch.rand(m)
    print(X)
    print(Y)
    model = MLP(n, m, Y)
    params = [p for p in model.parameters()]

    shampoo=CustomShampoo(1e-3, params, p=4, chol=True, optimized=False, debug=False) #basic custom Shampoo implementation, no kronecker factor optimization

    max_iters=1500
    iter_num=0
    temp=None

    s=time.time()
    while True:
        lr = get_lr(iter_num)
        for param_group in shampoo.param_groups:
            param_group['lr'] = lr
        temp, L=model(X)
        L.backward()
        #print(torch.linalg.norm(Y-temp).item())
        shampoo.step()
        shampoo.zero_grad(set_to_none=True)
        iter_num+=1
        if iter_num>max_iters:
            break
    e=time.time()
    print(Y, temp, torch.linalg.norm(Y-temp).item())
    total_loss+=torch.linalg.norm(Y-temp).item()
    temp=torch.eye(n)
    for p in model.l1.parameters():
        temp=p@temp
    for p in model.l2.parameters():
        temp=p@temp
    total_time+=e-s
print(total_time/trials, total_loss/trials)

###
#10 trials, n=m=20:
#default: ave time=3.2739959001541137
#Cholesky: ave time=3.2245911598205566, ave loss=4.731678927782923e-05
#10 trials, n=m=50:
#default: ave time=5.090388298034668, ave loss=0.00307787605561316
#Cholesky: ave time=3.005687785148621, ave loss=0.016716450452804565
#1 trial, n=m=100:
#default: ave time=15.172842979431152, ave loss=0.014870761893689632
#Cholesky: ave time=6.765733003616333, ave loss=0.005552446469664574

#10 trials: n=m=10, no activation
#default: ave time=3.032610774040222, ave loss=4.553738472168334e-05
#Cholesky: ave time=2.0442226409912108, ave loss=7.000631740083918e-05
###