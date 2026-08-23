from email.policy import default
from os import times
import os
import sys

# Ensure local ShampooExperiment modules are importable in spawn workers.
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib.pyplot as plt
from sympy import beta
from model import *
import torch
import torch.optim as opt
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
import math, time, copy, json, itertools
# from CustomShampoo import CustomShampoo
# from WhiteningShampoo import WhiteningShampoo
# from SCIShampoo import SCIShampoo
from Base import *


def _resolve_beta2(hp: dict, default: float = 0.85):
    """Return the beta2 (exp. average) value from hyperparams.

    Preference order: 'beta2' -> second element of 'betas' -> 'beta' -> default
    """
    if not isinstance(hp, dict):
        return default
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

def testQuad(curr_Optimizer, hyper_params, n, rand_seed, max_iters=10000):
    # max_iters=10000 #4000 for pure
    # warmup_iters=.001*max_iters #.2*max_iters
    # learning_rate=.9 #.99 MatrixSimple
    # lr_decay_iters=.01*max_iters #.8*max_iters
    # min_lr = 6e-3 #6e-5 default, 6e-2 for pure

    # Shp={'learning_rate': .99, 'warmup_iters':.00001, 'lr_decay_iters':.01,'min_lr':6e-3}
    # CShp={'learning_rate': .9, 'warmup_iters':.001, 'lr_decay_iters':.01,'min_lr':6e-3}
    # CS_2hp={'learning_rate': .9, 'warmup_iters':.001, 'lr_decay_iters':.01,'min_lr':6e-3}

    learning_rate=hyper_params.get('lr')
    warmup_iters=hyper_params.get('warmup_iters')
    lr_decay_iters=hyper_params.get('lr_decay_iters')
    min_lr = hyper_params.get('min_lr')
    beta2=_resolve_beta2(hyper_params)
    momentum=hyper_params.get('momentum', 0.9)
    weight_decay=hyper_params.get('weight_decay', 0.0)
    betas=hyper_params.get('betas', (0.9,0.999))

    model=MatrixSimple(torch.eye(n),rand_seed)
    params=[p for p in model.parameters()]

    match curr_Optimizer:
        case 0:
            optimizer=CustomShampoo(W=params,lr=learning_rate,chol=False,beta2=.999)
        case 1:
            optimizer=CustomShampoo(W=params,lr=learning_rate,chol=True,beta2=.999)
        case OPTS.MUON:
            muon_groups = [{
                'params': params,
                'kind': 'muon',
                'lr': learning_rate,
                'momentum': momentum,
                'ns_steps': 5,
                'beta2': beta2,
                'weight_decay': weight_decay,
            }]
            optimizer=MuonAdamW(muon_groups)
        case OPTS.STIEFEL_SGD:
            optimizer=VariationalStiefelSGD(params, lr=learning_rate, momentum=momentum)
        case OPTS.STIEFEL_ADAM:
            optimizer=VariationalStiefelAdam(params, lr=learning_rate, betas=betas)
        case 3:
            optimizer=CustomShampoo(W=params,lr=learning_rate,chol=True,p=2,beta2=.85)
        case OPTS.SCI:
            optimizer=SCIShampoo(W=params,lr=learning_rate,beta2=beta)
        case OPTS.SGD:
            optimizer=opt.SGD(params, lr=learning_rate)

    ###CS-CI
    # warmup_iters=.0001*max_iters
    # learning_rate=.9
    # lr_decay_iters=.0009*max_iters
    # min_lr = 6e-3

    ###SCIShampoo
    # warmup_iters=.00001*max_iters #.2*max_iters
    # learning_rate=.99 #.99 MatrixSimple
    # lr_decay_iters=.01*max_iters #.8*max_iters
    # min_lr = .8 #6e-5 default, 6e-2 for pure

    #S=CustomShampoo(learning_rate,params,p=4,chol=False)
    #optimizer=opt.SGD(params)
    #optimizer=SCIShampoo(learning_rate, params, .85) #142.371915102005, 1179
    #optimizer=CustomShampoo(learning_rate,params,p=4,chol=True,beta2=.999)

    iter_num=0
    print(f"OPTIMIZER {curr_Optimizer.name}")
    s=time.time()
    loss=[]
    i=0
    while True:
        lr=.99
        lr = get_lr(iter_num, learning_rate, warmup_iters*max_iters, lr_decay_iters*max_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        G, L=model()
        L.backward()
        loss.append(L.item())
        i+=1
        #print("Loss: ", L.item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        iter_num+=1
        if iter_num>=max_iters:
            break
    e=time.time()
    diff_time=e-s
    #np.save("data/losses/quad-n=1000-SGD.npy", np.array(losses[O]))
    # a=np.load("data/quad-n=100-S.npy")
    # b=np.load("data/quad-n=100-CS.npy")
    # c=np.load("data/quad-n=100-S_P2.npy")
    # d=np.load("data/quad-n=100-SGD.npy")
    # print(a.shape, b.shape,c.shape)
    
    print(iter_num-1)
    #print(optimizer.fails/(iter_num-1), optimizer.fails)
    # np.save("data/quad-n=100-SGD.npy", np.array(losses[0]))
    # np.save("data/quad-n=100-CS.npy", np.array(losses[CS]))
    # np.save("data/quad-n=100-S_P2.npy", np.array(losses[S_P2]))
    # plt.plot(np.arange(len(losses[0])), losses[0],color='blue',label='Shampoo-p=4')
    # plt.plot(np.arange(len(losses[CS])), losses[CS],color='green',label='CholeskyS-p=4')
    # plt.plot(np.arange(len(losses[S_P2])), losses[S_P2],color='red',label='CholeskyS-p=2')
    # plt.legend()
    # plt.show()
    return loss, diff_time

def grid_Search_Quad(OP, hyperparams, n, rand_seed=2, spectrum=[0,1]):
    iter_num=0

    init_lr=hyperparams['lr']
    warmup=hyperparams['warmup_iters']
    decay=hyperparams['lr_decay_iters']
    min_lr=hyperparams['min_lr']
    max_iters=hyperparams['max_iters']
    beta2=_resolve_beta2(hyperparams)

    torch.manual_seed(rand_seed)

    target= make_target_param(n, rand_seed, spectrum)
    #target=torch.eye(n)
    #print(torch.linalg.cond(target))

    model=MatrixSimple(target,rand_seed)
    params=[p for p in model.parameters()]

    optimizer = make_optimizer(OP, params, hyperparams)

    s=time.time()
    loss=[]
    while True:
        lr = get_lr(iter_num, init_lr, warmup*max_iters, decay*max_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        G, L=model()
        L.backward()
        loss.append(L.item())
        #print("Loss: ", L.item(), "lr: ", lr)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        iter_num+=1
        if iter_num>=max_iters:
            break
    e=time.time()
    del optimizer
    del model
    hp={'lr':init_lr, 'warmup_iters': warmup, 'lr_decay_iters': decay, 'min_lr': min_lr}
    hp['beta']=beta2
    if 'betas' in hyperparams:
        hp['betas']=hyperparams['betas']
    hp['max_iters']=iter_num-1
    hp['loss']=loss[-1]
    hp['time']=e-s
    print(f"{OP.name}-time: {e-s}-loss: {hp['loss']}-lr: {init_lr}")
    return hp

def get_Stats(W):
    fro_norm = torch.linalg.norm(W, ord='fro')
    inf_norm = torch.linalg.vector_norm(W, ord=torch.inf)
    spec_norm = torch.linalg.vector_norm(W, ord=2)
    return fro_norm, inf_norm, spec_norm

def make_target_param(n, rand_seed, spectrum):
    g=torch.Generator().manual_seed(rand_seed)
    
    random_mat1 = torch.randn((n,n),generator=g)
    random_mat2 = torch.randn((n,n),generator=g)
    U, _ = torch.linalg.qr(random_mat1)
    Vt, _ = torch.linalg.qr(random_mat2)
    s_values = torch.logspace(spectrum[0], spectrum[1], steps=n)  # ranges from 10^spectrum[0] to 10^spectrum[1]

    S = torch.diag(s_values)
    return nn.Parameter(U @ S @ Vt)

def analysis_Quad(OP, hyperparams, n, rand_seed=2, spectrum=[0,-5]):
    iter_num=0

    init_lr=hyperparams['lr']
    warmup=hyperparams['warmup_iters']
    decay=hyperparams['lr_decay_iters']
    min_lr=hyperparams['min_lr']
    max_iters=hyperparams['max_iters']

    #torch.manual_seed(rand_seed)

    target= make_target_param(n, rand_seed, spectrum)
    #target=torch.eye(n)
    kappa=torch.linalg.cond(target)

    model=MatrixSimple(target,rand_seed)
    params=[p for p in model.parameters()]

    optimizer=make_optimizer(OP, params, hyperparams)


    s=time.time()
    loss=[]
    i=0
    print("initlr", init_lr)
    while True:
        lr = get_lr(iter_num, init_lr, warmup*max_iters, decay*max_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        G, L=model()
        L.backward()
        loss.append(L.item())
        i+=1
        #print("Loss: ", L.item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        iter_num+=1
        if iter_num>=max_iters:
            break
    e=time.time()
    print('time', e-s, loss[-1])
    return loss, e-s, kappa

import functools
def save_optimizer_step(func):
  @functools.wraps(func)
  def wrapper(self, *args, **kwargs):
    # Save parameters before the update step
    before_params = [
        p.clone() for group in self.param_groups for p in group['params']
    ]

    # Run the original step function
    result = func()

    # Calculate and store the update difference
    self.last_updates = []
    idx = 0
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is not None:
          diff = p.data - before_params[idx]
          self.last_updates.append(diff)
        else:
          self.last_updates.append(torch.zeros_like(p.data))
        idx += 1

    return result

  return wrapper

def analysis_Quad_Stats(OP, hyperparams, n, spectrum=[0,0], rand_seed=2):
    iter_num=0

    init_lr=hyperparams['lr']
    warmup=hyperparams['warmup_iters']
    decay=hyperparams['lr_decay_iters']
    min_lr=hyperparams['min_lr']
    max_iters=hyperparams['max_iters']

    target= make_target_param(n, rand_seed, spectrum)
    model=MatrixSimple(target,rand_seed)
    params=[p for p in model.parameters()]

    optimizer=make_optimizer(OP, params, hyperparams)
    optimizer.step = save_optimizer_step(optimizer.step).__get__(
        optimizer, optimizer.__class__
    )

    s=time.time()
    loss=[]
    stats = {"L_fro_norm": [], "L_inf_norm": [], "L_spec_norm": [], "R_fro_norm": [], "R_inf_norm": [],
             "R_spec_norm": [], "G_fro_norm": [], "G_inf_norm": [], "G_spec_norm": [], "L":[], "R":[], "G":[]}
    stats['P'] = []
    i=0
    while True:
        lr = get_lr(iter_num, init_lr, warmup*max_iters, decay*max_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        G, L=model()
        L.backward()
        loss.append(L.item())

        i+=1
        #print(model.W)
        #print("Loss: ", L.item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        stats['G']+=optimizer.last_updates
        stats['P']+=[model.W.clone()]

        if OP==OPTS.S:
            p = params[0]
            state = optimizer.get_state()
            L = state[p]['Lp']
            R = state[p]['Rp']
            stats['L'].append(L)
            stats['R'].append(R)
            stats['G'].append(G)
            print(L.sum()-L.trace(), R.sum()-R.trace())
            a, b, c = get_Stats(L)
            stats['L_fro_norm'].append(a)
            stats['L_inf_norm'].append(b)
            stats['L_spec_norm'].append(c)
            a, b, c = get_Stats(R)
            stats['R_fro_norm'].append(a)
            stats['R_inf_norm'].append(b)
            stats['R_spec_norm'].append(c)
            a, b, c = get_Stats(G)
            stats['G_fro_norm'].append(a)
            stats['G_inf_norm'].append(b)
            stats['G_spec_norm'].append(c)

        iter_num+=1
        if iter_num>=max_iters:
            break
    e=time.time()
    print('time', e-s, loss[-1])
    return loss, e-s, stats

def trainMLP2(optimizer, hyperparams, n, h, mult, samples=10, batch_size=10, i=2):

    iter_num=0
    O=optimizer
    # init_lr=hyperparams[0]
    # warmup=hyperparams[1]
    # decay=hyperparams[2]
    # min_lr=hyperparams[3]
    # max_iters=hyperparams[4]
    # beta2=hyperparams[5]

    init_lr=hyperparams['lr']
    warmup=hyperparams['warmup_iters']
    decay=hyperparams['lr_decay_iters']
    min_lr=hyperparams['min_lr']
    max_iters=hyperparams['max_iters']
    beta2=_resolve_beta2(hyperparams)

    max_iters=1000 #4000

    ###TRAIN RAND VECTORS-80%
    torch.manual_seed(i)
    x=torch.rand(samples,n)
    y=mult*x
    dl=TensorDataset(x,y)
    ds=DataLoader(dl,batch_size,True)

    ###TEST RAND VECTORS-20%
    torch.manual_seed(10*i+2)
    x=torch.rand(samples//4,n)
    y=mult*x
    dl_test=TensorDataset(x,y)
    ds_test=DataLoader(dl_test,batch_size,True)


    model = MLP2(n,n,h)
    lower=False
    #if O==4 and lower: ###init weights to be lower triangular
    for p in model.parameters():
        if len(p.shape)==2:
            print("lower")
            with torch.no_grad():
                sdv=1./np.sqrt(p.shape[1])
                print("lower")
                p.data.uniform_(-sdv,sdv)
                p.data.copy_(torch.tril(p.data)) #use default uniform, don't want random
    params=[p for p in model.parameters()]
    match optimizer:
        case 0:
            optimizer=CustomShampoo(W=params,lr=init_lr,chol=False,beta2=beta2)
        case 1:
            optimizer=CustomShampoo(W=params,lr=init_lr,chol=True,beta2=beta2)
        case 2:
            optimizer=WhiteningShampoo(groups=params,lr=init_lr,pure=True,beta2=beta2)
        case 3:
            optimizer=CustomShampoo(W=params,lr=init_lr,chol=True,p=2,beta2=beta2)
        case SCI:
            optimizer=SCIShampoo(W=params,lr=init_lr,beta2=beta2)

    s=time.time()

    count=0
    loss=0.0
    loss_arr=[]
    while True:
        lr = get_lr(iter_num, init_lr, warmup*max_iters, decay*max_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        loss=0.0
        for f, l in ds:
            temp=model(f)
            L=torch.sum(torch.norm(temp-l,dim=1))/batch_size
            L.backward()
            loss+=L.item()
        loss/=(samples//batch_size)
        loss_arr.append(loss)
        print(f"LOSS at iter {iter_num}: {loss}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        iter_num+=1
        if loss<=.1:
            count+=1
        else:
            count=0
        if iter_num>max_iters or count>=5:
            break
    print("TRAINLOSS: ", loss)

    e=time.time()
    fa=optimizer.fails
    torch.save(model.state_dict(), f"data/models/MLP2(n={n},h={h},mult={mult})_{O}_hp")
    
    loss2=0.0
    for f, l in ds_test:
        temp=model(f)
        L=torch.sum(torch.norm(temp-l,dim=1))/batch_size
        loss2+=L.item()
        # print("temp", temp[0])
        # print("act", l[0])
    loss2/=((samples//4)//batch_size)
    print("TESTLOSS: ", loss2)
    
    del optimizer
    del model
    hp={'lr':init_lr, 'warmup_iters': warmup, 'lr_decay_iters': decay, 'min_lr': min_lr}
    hp['beta']=beta2
    if 'betas' in hyperparams:
        hp['betas']=hyperparams['betas']
    hp['max_iters']=iter_num-1
    hp['loss']=loss
    hp['time']=e-s
    print('time', e-s, "fails", fa)
    return hp, loss_arr

def evalMLP2(n, h, mult, i=2):
    """TODO"""

if __name__=='__main__':
    grid_Search_Quad(OPTS.MUON, {'params':[torch.rand(10,10)], 'kind':'muon', 'lr':.1, 'momentum':.9, 'ns_steps':5, 'beta2':.999, 
        'weight_decay':0.0, 'warmup_iters':.05, 'lr_decay_iters':.1, 'min_lr':.0001, 'max_iters':4000}, n=2)
    #x=MuonAdamW([{'params':[torch.rand(10,10)], 'kind':'muon', 'lr':.1, 'momentum':.9, 'ns_steps':5, 'beta2':.999, 'weight_decay':0.0}])
    print("TrainingScripts.py")