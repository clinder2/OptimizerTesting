from model import MatrixSimple, MLP, ComplicatedMLP
import torch
import torch.optim as opt
import torch.multiprocessing as mp
import math, time, copy, json, itertools
from CustomShampoo import CustomShampoo
from WhiteningShampoo import WhiteningShampoo
from GridSearch import get_lr
from Base import *

def MLP_rand(optimizer,x,i,hp):
    torch.manual_seed(i+1)
    x=torch.rand(100)
    #torch.manual_seed(i+1)
    y=10*x
    model=MLP(100,100,y)
    params=[p for p in model.parameters()]

    match optimizer:
        case 0:
            optimizer=CustomShampoo(W=params,lr=hp['lr'],chol=False)
        case 1:
            optimizer=CustomShampoo(W=params,lr=hp['lr'],chol=True)
        case 2:
            optimizer=WhiteningShampoo(groups=params,lr=hp['lr'],pure=True)
        case S_P2:
            optimizer=CustomShampoo(W=params,lr=hp['lr'],chol=True,p=2)

    #optimizer=WhiteningShampoo(groups=params,lr=hp['lr'],pure=True)

    max_iters=hp['max_iters']
    warmup=max_iters*hp['warmup_iters']
    decay=max_iters*hp['lr_decay_iters']

    iter_num=0

    s=time.time()
    while True:
        lr = get_lr(iter_num,hp['lr'],warmup,decay,hp['min_lr'])
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
    print(L.item())
    return L.item(), e-s

if __name__=='__main__':
    for O in [WS]:
        if O==S:
            with open("data/MLP(100to100-mult)_S_hp.json", 'r') as f:
                hp=json.load(f)
        elif O==CS:
            with open("data/MLP(100to100-mult)_CS_hp.json", 'r') as f:
                hp=json.load(f)
        elif O==WS:
            with open("data/MLP(100to100-mult)_WS_hp.json", 'r') as f:
                hp=json.load(f)
        else:
            with open("data/MS(50to50)_CS-P2_hp.json", 'r') as f:
                hp=json.load(f)
        trials=50
        tot_loss=0.0
        tot_time=0.0
        torch.manual_seed(0)
        x=torch.rand(50)

        ctx=mp.get_context("spawn")

        with ctx.Pool(16) as pool:
            output=pool.starmap(
                MLP_rand, 
                [
                    (
                        O,
                        x,
                        t,
                        hp,
                    )
                    for t in range(trials)
                ]
            )

        for a in output:
            tot_loss+=a[0]
            tot_time+=a[1]
        print(f"FINAL: OPT={O}", tot_loss/trials,tot_time/trials)

###IF UNSPECIFIED max_iters=2000

###rand MLP###
###Shampoo results: tot_loss=0.00011033530237909872, tot_time=17.53478723526001
###Shampoo-chol=True results: tot_loss=0.0001562694059975911, tot_time=17.704886736869813
###WS-pure=True results: tot_loss=0.0005598210134121473, tot_time=10.261562476158142

###mult MLP###
###Shampoo results: tot_loss=0.0009620271751191467, tot_time=32.41621156692505
###Shampoo-chol=True results: tot_loss=0.0009825874387752265, tot_time=23.271225261688233
###2000iters_WS-pure=True results: tot_loss=0.12119188532233238, tot_time=17.216809701919555
###4000iters_WS-pure=True results: tot_loss=0.10710389330983162, tot_time=36.720536155700685

###8000iters_S results: tot_loss=0.0008305382245453075, tot_time=151.40726024627685
###8000iters_CS results: tot_loss=0.0012912023917306214, tot_time=99.86386492729187
###8000iters_WS-pure=True results: tot_loss=0.04123151498381048, tot_time=88.64280379295349

###2000iters_S-P2 results: tot_loss=0.0011928852161508985, tot_time=35.248321561813356