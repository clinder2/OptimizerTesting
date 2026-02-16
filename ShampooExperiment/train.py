import torch
from distributed_shampoo.preconditioner.matrix_functions_types import CholeskyConfig, CoupledNewtonConfig, EigenConfig
from distributed_shampoo.preconditioner.shampoo_preconditioner_list import RootInvShampooPreconditionerList
from model import MatrixSimple
from shampoo.shampoo import Shampoo
from distributed_shampoo import AdamPreconditionerConfig, DistributedShampoo, RootInvShampooPreconditionerConfig
from CustomShampoo import CustomShampoo
import math
import time

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

n=50
m=50
torch.manual_seed(1)
A=1.5*torch.rand((m,n))
A=torch.eye(n)
model = MatrixSimple(A,0) #A=torch.eye(m,n)
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

shampoo=CustomShampoo(1e-3, params, p=4, chol=False, optimized=False, debug=False) #basic custom Shampoo implementation, no kronecker factor optimization

trials=10
total_time=0.0
total_loss=0.0

for i in range(trials):
    max_iters=1500
    iter_num=0
    n=100
    m=100
    torch.manual_seed(i+1)
    A=1.5*torch.rand((m,n))
    #A=torch.eye(n)
    model = MatrixSimple(A,0) #A=torch.eye(m,n)
    #print(A)
    #print(model.W)
    params = [p for p in model.parameters()]
    shampoo=CustomShampoo(1e-3, params, p=4, chol=False, optimized=True, debug=False) #basic custom Shampoo implementation, no kronecker factor optimization
    s=time.time()
    while True:
        lr = get_lr(iter_num)
        for param_group in shampoo.param_groups:
            param_group['lr'] = lr
        G, L=model()
        #L.backward()
        shampoo.step()
        shampoo.zero_grad(set_to_none=True)
        #print(L.item())
        #print('R', torch.linalg.norm(shampoo.R))
        #model.W.data=zeropower_via_newtonschulz5(model.W.data)
        #temp=zeropower_via_newtonschulz5(model.A)
        iter_num+=1
        if iter_num>max_iters:
            break
    e=time.time()
    #print(model.W)
    #print(model.A)
    print("loss", torch.linalg.norm(model.A-model.W, ord='fro').item())
    print(e-s)
    total_loss+=torch.linalg.norm(model.A-model.W, ord='fro').item()
    total_time+=e-s
print(total_time/trials, total_loss/trials)
#print(model.W.T@model.W)
#print(model.A, model.W)

#default, n=m=20, trials=10, random A: L=2.4955728576969705e-06, T=1.5385940313339233
#chol, n=m=20, trials=10, random A: L=1.0191468902576161e-06, T=1.2499472856521607
#opt, n=m=20, trials=10, random A: L=2.504628525912267e-06, T=0.9256163120269776

#default: n=m=50, trials=10, random A: L=0.0031975640449672937, T=2.1838069200515746
#chol: n=m=50, trials=10, random A: L=5.026856615586439e-06, T=1.4225712060928344
#opt: n=m=50, trials=10, random A: L=0.0033664277754724024, T=1.270915699005127

#default: n=m=100, trials=10, random A: L=0.7255538463592529, T=4.644129133224487
#chol: n=m=100, trials=10, random A: L=1.0850950548046968e-05, T=3.5395977020263674
#opt: n=m=100, trials=10, random A: L=0.7511579930782318, T=2.159401607513428