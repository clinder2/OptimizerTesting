from model import *
import torch
import torch.optim as opt
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
import math, time, copy, json, itertools
from CustomShampoo import CustomShampoo
from WhiteningShampoo import WhiteningShampoo
from SCIShampoo import SCIShampoo
from Base import *

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

import time
import matplotlib.pyplot as plt
def testQuad(opti, model):
    max_iters=10000 #4000 for pure
    # warmup_iters=.001*max_iters #.2*max_iters
    # learning_rate=.9 #.99 MatrixSimple
    # lr_decay_iters=.01*max_iters #.8*max_iters
    # min_lr = 6e-3 #6e-5 default, 6e-2 for pure
    Shp={'learning_rate': .9, 'warmup_iters':.001, 'lr_decay_iters':.01,'min_lr':6e-3}
    CShp={'learning_rate': .9, 'warmup_iters':.001, 'lr_decay_iters':.01,'min_lr':6e-3}
    CS_2hp={'learning_rate': .9, 'warmup_iters':.001, 'lr_decay_iters':.01,'min_lr':6e-3}
    losses={}
    times={}

    for O in [1]:
        n=100
        model=MatrixSimple(torch.eye(n),2)
        params=[p for p in model.parameters()]
        match O:
            case 0:
                hp=Shp
                init_lr=hp['learning_rate']
                optimizer=CustomShampoo(W=params,lr=init_lr,chol=False,beta2=.85)
            case 1:
                hp=CShp
                init_lr=hp['learning_rate']
                optimizer=CustomShampoo(W=params,lr=init_lr,chol=True,beta2=.85)
            case S_P2:
                hp=CS_2hp
                init_lr=hp['learning_rate']
                optimizer=CustomShampoo(W=params,lr=init_lr,chol=True,p=2,beta2=.85)

        warmup_iters=.0001*max_iters #.2*max_iters
        learning_rate=.99 #.99 MatrixSimple
        lr_decay_iters=.0005*max_iters #.8*max_iters
        min_lr = .7 #6e-5 default, 6e-2 for pure
        #S=CustomShampoo(learning_rate,params,p=4,chol=False)
        #S=opt.AdamW(params)
        optimizer=SCIShampoo(learning_rate, params) #142.371915102005, 1179
        optimizer=CustomShampoo(learning_rate,params,p=4,chol=False)

        iter_num=0
        print(f"OPTIMIZER {O}")
        s=time.time()
        loss=[]
        i=0
        while True:
            lr = get_lr(iter_num, learning_rate, warmup_iters*max_iters, lr_decay_iters*max_iters, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            G, L=model()
            L.backward()
            loss.append(L.item())
            i+=1
            print("Loss: ", L.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            iter_num+=1
            if iter_num>=max_iters or L.item()<=1e-6:
                break
        e=time.time()
        losses[O]=loss
        times[O]=e-s
    # a=np.load("data/quad-n=100-S.npy")
    # b=np.load("data/quad-n=100-CS.npy")
    # c=np.load("data/quad-n=100-S_P2.npy")
    # d=np.load("data/quad-n=100-SGD.npy")
    # print(a.shape, b.shape,c.shape)
    
    print(times)
    print(optimizer.fails, iter_num-1, optimizer.fails/(iter_num-1))
    # np.save("data/quad-n=100-SGD.npy", np.array(losses[0]))
    # np.save("data/quad-n=100-CS.npy", np.array(losses[CS]))
    # np.save("data/quad-n=100-S_P2.npy", np.array(losses[S_P2]))
    # plt.plot(np.arange(len(losses[0])), losses[0],color='blue',label='Shampoo-p=4')
    # plt.plot(np.arange(len(losses[CS])), losses[CS],color='green',label='CholeskyS-p=4')
    # plt.plot(np.arange(len(losses[S_P2])), losses[S_P2],color='red',label='CholeskyS-p=2')
    # plt.legend()
    # plt.show()

def trainQuad(optimizer, hyperparams, n, h, mult, samples=10, batch_size=10, i=2):
    iter_num=0

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
    beta2=hyperparams['beta2']

    max_iters=10000 #4000

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
            optimizer=SCIShampoo(W=params,lr=init_lr)

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
    hp['beta2']=beta2
    hp['max_iters']=iter_num-1
    hp['loss']=loss
    hp['time']=e-s
    print('time', e-s, "fails", fa)
    return hp, loss_arr

def evalMLP2(n, h, mult, i=2):
    """TODO"""

if __name__=='__main__':
    testQuad(0,0)