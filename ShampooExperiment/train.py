import torch
from model import MatrixSimple
from shampoo.shampoo import Shampoo
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

model = MatrixSimple(torch.randn(4,4))
params = [p for p in model.parameters()]

optim_groups = [
    {'params': params, 'weight_decay': 0.0}
]
shampoo = Shampoo(optim_groups)

max_iters=70
iter_num=0

while True:
    lr = get_lr(iter_num)
    for param_group in shampoo.param_groups:
        param_group['lr'] = lr
    G, L=model()
    #L.backward()
    shampoo.step()
    shampoo.zero_grad(set_to_none=True)
    print(model.A-model.W, torch.linalg.norm(model.A-model.W).item())
    iter_num+=1
    if iter_num>max_iters:
        break