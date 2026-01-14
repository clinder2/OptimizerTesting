import torch
from distributed_shampoo.preconditioner.matrix_functions_types import CholeskyConfig, CoupledNewtonConfig, EigenConfig
from distributed_shampoo.preconditioner.shampoo_preconditioner_list import RootInvShampooPreconditionerList
from model import MatrixSimple, MLP, ComplicatedMLP
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

trials=20
total_time=0.0
total_loss=0.0
X=[]
Y=[]
n=10
m=10
for i in range(0,trials,2):
    torch.manual_seed(i)
    x=torch.rand(n)
    torch.manual_seed(i+1)
    #y=torch.rand(m)
    y=10*x
    X.append(x)
    Y.append(y)
for i in range(trials//2):
    # n=10
    # m=10
    # torch.manual_seed(1)
    # X=torch.rand(n)
    # torch.manual_seed(6)
    # Y=torch.rand(m)
    x=X[i]
    y=Y[i]
    print(x)
    print(y)
    #model = MLP(n, m, y)
    model = ComplicatedMLP(n, m, y)
    params = [p for p in model.parameters()]

    shampoo=CustomShampoo(1e-3, params, p=4, chol=False, optimized=True, debug=False) #basic custom Shampoo implementation, no kronecker factor optimization

    max_iters=1500
    iter_num=0
    temp=None

    s=time.time()
    while True:
        lr = get_lr(iter_num)
        for param_group in shampoo.param_groups:
            param_group['lr'] = lr
        temp, L=model(x)
        L.backward()
        #print(torch.linalg.norm(Y-temp).item())
        shampoo.step()
        shampoo.zero_grad(set_to_none=True)
        iter_num+=1
        if iter_num>max_iters:
            break
    e=time.time()
    print(y, temp, torch.linalg.norm(y-temp).item())
    total_loss+=torch.linalg.norm(y-temp).item()
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

#10 trials: n=m=10, simple MLP no activation 10 randomly seeded pairs
#default: ave time=2.06697256565094, ave loss=1.592513890500413e-05
#Cholesky: ave time=1.4701266407966613, ave loss=1.5822704199308646e-05

#10 trials: n=m=10, complex MLP 10 pairs
#default: ave time=3.682239758968353, ave loss=9.314075759903062e-06
#Cholesky: ave time=2.4288045525550843, ave loss=7.323874206122127e-06

#10 trials: n=m=10, simple MLP no activation 10 pairs, y=10*x
#default: ave time=2.5859535098075868, ave loss=0.00010020767658716067
#Cholesky: ave time=1.7528061985969543, ave loss=7.314181420952082e-05
#opt (R**-1/2): ave time=1.4481961965560912, ave loss=9.861242942861281e-05

#10 trials: n=m=10, complex MLP 10 pairs, y=10*x
#default: ave time=4.4688972473144535, ave loss=8.533688487659674e-05
#Cholesky: ave time=2.2778189420700072, ave loss=5.5509957383037546e-05
#opt (R**-1/2): ave time=2.322601652145386, ave loss=5.631251806335058e-05
###