from model import MatrixSimple, MLP, ComplicatedMLP
import torch
import torch.optim as opt
import torch.multiprocessing as mp
import math, time, copy, json, itertools
from CustomShampoo import CustomShampoo
from WhiteningShampoo import WhiteningShampoo
from Base import *

###for MLP rand and mult
grid = {
    'lr': [.001, .01, .1, .8, .9, .99],
    'warmup_iters': [.1,.2,.3,.4],
    'lr_decay_iters': [.05,.1,.2,.3,.4],
    'min_lr': [6e-5,6e-2,1e-2,1e-1],
}

fine_grid = {
    'lr': [.0001, .001, .01, .1, .2, .8, .9, .99],
    'warmup_iters': [.05,.1,.2,.3,.4],
    'lr_decay_iters': [.05,.1,.2,.3,.4,.7],
    'min_lr': [6e-5,6e-4,6e-2,1e-2,1e-1],
}

def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
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

def train(optimizer, model, hyperparams, max_iters=4000):
    iter_num=0
    init_lr=hyperparams[0]
    warmup=hyperparams[1]
    decay=hyperparams[2]
    min_lr=hyperparams[3]

    n=50
    m=50
    torch.manual_seed(1)
    x=torch.rand(n)
    y=10*x
    model=model(n,m,y)
    params=[p for p in model.parameters()]
    match optimizer:
        case 0:
            optimizer=CustomShampoo(W=params,lr=init_lr,chol=False)
        case 1:
            optimizer=CustomShampoo(W=params,lr=init_lr,chol=True)
        case WS:
            optimizer=WhiteningShampoo(groups=params,lr=init_lr,pure=True)

    s=time.time()

    while True:
        lr = get_lr(iter_num, init_lr, warmup*max_iters, decay*max_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        temp, L=model(x)
        L.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        iter_num+=1
        if iter_num>max_iters:
            break
    e=time.time()
    del optimizer
    del model
    hp={'lr':init_lr, 'warmup_iters': warmup, 'lr_decay_iters': decay, 'min_lr': min_lr}
    hp['loss']=L.item()
    hp['time']=e-s
    print(L.item())
    return hp

def trainMS(optimizer, model, hyperparams, max_iters=4000):
    iter_num=0
    init_lr=hyperparams[0]
    warmup=hyperparams[1]
    decay=hyperparams[2]
    min_lr=hyperparams[3]

    n=50
    m=50
    torch.manual_seed(1)
    x=torch.rand(n)
    model=model(torch.eye(n), 0)
    params=[p for p in model.parameters()]
    match optimizer:
        case 0:
            optimizer=CustomShampoo(W=params,lr=init_lr,chol=False)
        case 1:
            optimizer=CustomShampoo(W=params,lr=init_lr,chol=True)
        case WS:
            optimizer=WhiteningShampoo(groups=params,lr=init_lr,pure=True)

    s=time.time()

    while True:
        lr = get_lr(iter_num, init_lr, warmup*max_iters, decay*max_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        G, L=model()
        L.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        iter_num+=1
        if iter_num>max_iters:
            break
    e=time.time()
    del optimizer
    del model
    hp={'lr':init_lr, 'warmup_iters': warmup, 'lr_decay_iters': decay, 'min_lr': min_lr}
    hp['loss']=L.item()
    hp['time']=e-s
    print(L.item())
    return hp

def grid_search(optimizer, model, grid=grid, num_workers=16):
    hyperparams={'lr': 0, 'warmup_iters': 0, 'lr_decay_iters': 0, 'min_lr': 0}
    best_loss=100

    hp_list=itertools.product(grid['lr'], grid['warmup_iters'], 
                              grid['lr_decay_iters'], grid['min_lr'])
    hp_list=[h for h in hp_list if h[1]!=h[2]]

    ctx=mp.get_context("spawn")

    with ctx.Pool(num_workers) as pool:
        output=pool.starmap(
            trainMS, 
            [
                (
                    optimizer,
                    model,
                    hp,
                )
                for hp in hp_list
            ]
        )

    hyperparams=min(output, key=lambda x: x['loss'])
    return output, hyperparams

if __name__=='__main__':
    #for O in OPTS:
        O=WS
        output, hp=grid_search(O, MatrixSimple, fine_grid)
        
        print(hp)
        ###NAME hyperparameter dictionary json as MODEL_OPTIMIZER_hp.json
        ###MS(ntom)
        ###MLP(ntom-{rand, mult})
        ###WS-WhiteningShampoo, CS-CustomShampoo with chol=True, S-CustomShampoo
        ###with chol=False
        match O:
            case 0:
                with open(f"data/MS(50to50)_S_hp.json", 'w') as f:
                    json.dump(hp, f)
                with open("data/MS(50to50)_S_gridresults.json", 'w') as f:
                    json.dump(output, f)
            case 1:
                with open(f"data/MS(50to50)_CS_hp.json", 'w') as f:
                    json.dump(hp, f)
                with open("data/MS(50to50)_CS_gridresults.json", 'w') as f:
                    json.dump(output, f)
            case WS:
                with open(f"data/MS(50to50)_4000_WS_fine_hp.json", 'w') as f:
                    json.dump(hp, f)
                with open("data/MS(50to50)_4000_WS_finegridresults.json", 'w') as f:
                    json.dump(output, f)
    
    