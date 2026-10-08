from cProfile import label

from StackedShampoo import *

import matplotlib.pyplot as plt
import numpy as np

batch=10
S=[]
CS=[]
ns=[100,200,300,400,500,550,600,650,700,800,850,900,1000,1100,1200,1300]
for n in ns:

    St=0
    CSt=0
    spectrum=[0,-5]
    for b in range(batch):
        torch.manual_seed(b)
        A = torch.randn(n,n)

        random_mat1 = torch.randn(n,n)
        random_mat2 = torch.randn(n,n)
        U, _ = torch.linalg.qr(random_mat1)
        Vt, _ = torch.linalg.qr(random_mat2)
        s_values = torch.logspace(spectrum[0], spectrum[1], steps=n)  # ranges from 10^spectrum[0] to 10^spectrum[1]
    
        Si = torch.diag(s_values)
        A= U @ Si @ Vt

        A=A@A.T+3*torch.eye(n)
        s=time.time()
        ComputePower(A, 4,iter_count=5)
        e=time.time()
        St+=e-s
        print(e-s)

        s=time.time()
        L, i = torch.linalg.cholesky_ex(A)
        if i==0:
            #inverse_sqrtm_newton_schulz(L,num_iters=5)
            ComputePower(L, 2, iter_count=5)
            e=time.time()
            CSt+=e-s
        else:
            print("fail")
    S.append(St/batch)
    CS.append(CSt/batch)
    print(St/batch, CSt/batch)
x=np.arange(len(ns))
plt.plot(x, np.array(S), color='green', label='shampoo:-1/4')
plt.plot(x, np.array(CS), color='red', label='choleskyShampoo:chol + -1/2')
plt.xticks(x, ns)
plt.title("NxN mat size v. wall clock time (sec)")
plt.legend()
plt.show()