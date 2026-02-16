from model import MatrixSimple, MLP, ComplicatedMLP
import torch
import torch.optim as opt
import torch.multiprocessing as mp
import math, time, copy, json, itertools
from CustomShampoo import CustomShampoo
from WhiteningShampoo import WhiteningShampoo
from GridSearch import get_lr

def MLP_rand(x,i,hp):
    torch.manual_seed(i+1)
    x=torch.rand(50)
    #torch.manual_seed(i+1)
    y=10*x
    model=MLP(50,50,y)
    params=[p for p in model.parameters()]
    optimizer=WhiteningShampoo(groups=params,lr=hp['lr'],pure=True)

    max_iters=2000
    warmup=2000*hp['warmup_iters']
    decay=2000*hp['lr_decay_iters']

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
    with open("data/MLP(50to50-mult)_4000_WS_hp.json", 'r') as f:
        hp=json.load(f)
    max_iters=2000
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
    print("FINAL: ", tot_loss/trials,tot_time/trials)

###rand MLP###
###Shampoo results: tot_loss=0.00011033530237909872, tot_time=17.53478723526001
###Shampoo-chol=True results: tot_loss=0.0001562694059975911, tot_time=17.704886736869813
###WS-pure=True results: tot_loss=0.0005598210134121473, tot_time=10.261562476158142

###mult MLP###
###Shampoo results: tot_loss=0.0009620271751191467, tot_time=32.41621156692505
###Shampoo-chol=True results: tot_loss=0.0009825874387752265, tot_time=23.271225261688233
###2000iters_WS-pure=True results: tot_loss=0.12119188532233238, tot_time=17.216809701919555
###4000iters_WS-pure=True results: tot_loss=0.10710389330983162, tot_time=36.720536155700685