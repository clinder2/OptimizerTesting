import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
from typing import List
#from ShampooExperiment.Grafting import AdagradGraft
from Grafting import AdagradGraft
from CustomShampoo import CustomShampoo
#from ShampooExperiment.CustomShampoo import CustomShampoo

"""Whitening Shampoo
ref: https://en.wikipedia.org/wiki/Whitening_transformation, https://arxiv.org/pdf/2509.22938, https://arxiv.org/pdf/1512.00809
"""
class WhiteningShampoo(Optimizer):
    def __init__(self, groups, lr, debug=False, pure=False, beta2=.85):
        data=dict(lr=lr)
        super().__init__(groups, data)
        self.device=groups[0].device
        self.state={} #paramteter state dictionary
        self.cache={}
        for g in self.param_groups:
            for p in g['params']:
                self.state[p]={} #init each parameter's state
                self.state[p]['graft']=AdagradGraft(None, p) #init Adagrad grafting
                self.state[p]['L']=torch.eye(p.shape[0],device=self.device) #p's left preconditioner
                self.state[p]['R']=torch.eye(p.shape[1],device=self.device) #p's right preconditioner
                self.cache[p]={}
                self.cache[p]['Lp']=torch.eye(p.shape[0],device=self.device)
                self.cache[p]['Rp']=torch.eye(p.shape[1],device=self.device)
        self.debug=debug
        self.pure=pure
        self.iter=0
        self.beta2=beta2 #for L, R exp. decay (.85)

    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                graft=self.state[p]['graft']
                grad=p.grad
                #update preconditioners
                # L=self.state[p]['L']
                # R=self.state[p]['R']
                # L=self.beta2*L+(1-self.beta2)*grad@grad.T #update left/right preconditioners
                # R=self.beta2*R+(1-self.beta2)*grad.T@grad
                # self.state[p]['L']=L #update state of preconditioners
                # self.state[p]['R']=R

                self.state[p]['L'].lerp_(grad@grad.T,1-self.beta2)
                self.state[p]['R'].lerp_(grad.T@grad,1-self.beta2)
                L=self.state[p]['L']
                R=self.state[p]['R']
                if self.pure:
                    #Cholesky of original Sigma
                    Lp=torch.linalg.cholesky_ex(.001*torch.eye(L.shape[0],device=self.device)+L)
                    Rp=torch.linalg.cholesky_ex(.001*torch.eye(R.shape[0],device=self.device)+R)
                    if Lp.info!=0 or Rp.info!=0:
                        #print("failed 1st Cholesky")
                        update=grad
                        break
                    Lp=Lp.L
                    Rp=Rp.L

                    Lp=torch.linalg.inv_ex(.001*torch.eye(L.shape[0],device=self.device)+Lp)
                    assert Lp.info==0, "failed Lp inverse"
                    #get inverse kronecker factor for inverse Sigma
                    #Lp=Lp.inverse@Lp.inverse.T
                    Lp=Lp.inverse.T@Lp.inverse

                    Rp=torch.linalg.inv_ex(.001*torch.eye(R.shape[0],device=self.device)+Rp)
                    assert Rp.info==0, "failed Rp inverse"
                    #get inverse kronecker factor for inverse Sigma
                    #Rp=Rp.inverse@Rp.inverse.T
                    Rp=Rp.inverse.T@Rp.inverse

                    #Lp=torch.linalg.inv_ex(.001*torch.eye(L.shape[0])+L).inverse
                    #Rp=torch.linalg.inv_ex(.001*torch.eye(R.shape[0])+R).inverse
                    #cholesky_factor=L of inverse Sigma and set W=L.T
                    Lp=torch.linalg.cholesky_ex(.001*torch.eye(L.shape[0],device=self.device)+Lp)
                    Rp=torch.linalg.cholesky_ex(.001*torch.eye(R.shape[0],device=self.device)+Rp)
                    if Lp.info!=0 or Rp.info!=0:
                        #print("failed 2nd Cholesky")
                        update=grad
                    else:
                        #print(torch.trace(self.state[p]['L']).item())
                        update=Lp.L.T@grad@Rp.L/torch.trace(self.state[p]['L'])
                        #update=Lp.L@grad@Rp.L.T
                else:
                    #compute "Cholesky" whitening matrix

                    success=True
                    Lp=torch.linalg.cholesky_ex(.001*torch.eye(self.state[p]['L'].shape[0],device=self.device)+self.state[p]['L'])
                    #assert Lp.info==0, "failed Lp Cholesky"
                    if Lp.info:
                        success=False
                        Lp=self.cache[p]['Lp']
                        print(f"failed Lp Cholesky on iter {self.iter}")
                    else:
                        LF=Lp.L
                        Lp=Lp.L
                        self.cache[p]['Lp']=Lp

                    Rp=torch.linalg.cholesky_ex(.001*torch.eye(self.state[p]['R'].shape[0],device=self.device)+self.state[p]['R'])
                    #assert Rp.info==0, "failed Rp Cholesky"
                    if Rp.info:
                        success=False
                        Rp=self.cache[p]['Rp']
                        print(f"failed Rp Cholesky on iter {self.iter}")
                    else:
                        RF=Rp.L
                        Rp=Rp.L
                        self.cache[p]['Rp']=Rp

                    Lp=torch.linalg.inv_ex(Lp)
                    assert Lp.info==0, "failed Lp inverse"
                    Lp=Lp.inverse

                    Rp=torch.linalg.inv_ex(Rp)
                    assert Rp.info==0, "failed Rp inverse"
                    Rp=Rp.inverse

                    if success:
                        update=Lp@grad@Rp.T
                        #update=Lp@Lp@grad@Rp.T@Rp.T
                        #update=Lp.T@update@Rp
                        
                        #update=Lp.T@LF.T@Lp@grad@Rp.T@RF@Rp
                    else:
                        print(f"failure of Cholesky on iter {self.iter}")
                        update=grad
                #grafting, update
                graft.add_statistics(grad) #update grafting state
                graft_grad=graft.precondition_gradient(grad) #do grafting
                graft_n=torch.linalg.norm(graft_grad, ord='fro')
                shampoo_n=torch.linalg.norm(update, ord='fro')
                #print('lr: ', g['lr'], "graft_n: ", graft_n.item(), "s_n", shampoo_n.item())
                p.data-=g['lr']*(graft_n/(shampoo_n+1e-8))*update #param update with grafting
                #p.data-=g['lr']*update
                if self.debug and self.iter%10==0:
                    print(f"PRECONDITIONERS at {self.iter}:")
                    print("SHAPE: ", "Lp: ", Lp.shape, " Rp: ", Rp.shape)
                    print("NORM (fro)", torch.linalg.norm(Lp,ord="fro"))
                    if self.opt and Rp.inverse!=None:
                        print("R: ", Rp.inverse)
                    else:
                        print("L: ", Lp.data)
                        print("R: ", Lp.data)
                    print("GRAD:")
                    print(grad)
                    print(f"UPDATE at {self.iter}:")
                    print(update)
        self.iter+=1

    def zero_grad(self, set_to_none = True):
        super().zero_grad(set_to_none)

###TESTS
from model import *
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as opt
import math, time
if __name__=='__main__':
    max_iters=4000 #4000 for pure
    warmup_iters=.4*max_iters #.2*max_iters
    learning_rate=.1 #.99 MatrixSimple
    lr_decay_iters=.3*max_iters #.8*max_iters
    min_lr = 6e-2 #6e-5 default, 6e-2 for pure

    ###for MatrixSimple, WS does better with constant lr=.99
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
    m=10
    torch.manual_seed(1)
    x=torch.rand(n)
    torch.manual_seed(3)
    #y=torch.ones(m)
    #y[0]=1
    x=torch.rand(10,n)
    y=2*x
    dl=TensorDataset(x,y)
    ds=DataLoader(dl,5,True)
    #model = ComplicatedMLP(n, m, y)

    #model = MLP2(n,m,y)

    model=MatrixSimple(torch.eye(n),2)
    params=[p for p in model.parameters()]
    #WS=WhiteningShampoo(params,learning_rate,pure=True)
    WS=CustomShampoo(learning_rate,params,p=4,chol=False)
    #WS=opt.AdamW(params, learning_rate)
    iter_num=0

    s=time.time()
    while True:
        lr = get_lr(iter_num)
        #lr=.99
        for param_group in WS.param_groups:
            param_group['lr'] = lr
        for f, l in ds:
            temp=model(f)
            #G, L=model()
            L=torch.sum(torch.norm(temp-l,dim=1))/5
            print(L.item())
            L.backward()
            #print("Loss: ", L.item())
        WS.step()
        WS.zero_grad(set_to_none=True)
        iter_num+=1
        if iter_num>max_iters:
            break
    e=time.time()
    print(e-s)
    print(WS.fails)
    # print(y)
    # print(temp)
    #print(torch.linalg.norm(y-temp).item())
    #print(model.A, model.W)

#1000 iters
#100, rand
#S-7.5, .0001
#WS-2.2, .0003

#100, ones
#S-7.0, .03
#WS-2.3, .87

#100, x2
#S-7.5, .02
#WS-2.2, 1.08

#100, x10
#S-7.7, .07
#CS-4.2, .06
#WS-2.5, 22.28