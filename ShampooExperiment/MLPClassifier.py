import matplotlib.pyplot as plt
from model import *
import torch.optim as opt
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
import math, time, copy, json, itertools
from CustomShampoo import CustomShampoo
from WhiteningShampoo import WhiteningShampoo
from SCIShampoo import SCIShampoo
from Base import *
from GridSearch import *

def get_uniform_normal_vectors(total_samples, in_dimension):
    x_uniform=torch.rand(total_samples//2,in_dimension)
    x_uniform=-1+2*x_uniform
    x_normal=torch.randn(total_samples//2,in_dimension)
    return x_uniform, x_normal

def trainMLPClassifier(optimizer, hyperparams, in_dimension, out_dimension, max_iters=1000, 
                       total_train_samples=100, total_test_samples=20, batch_size=10, 
                       rand_seed=2, debug=False, validate=False):
    name = optimizer.name
    model = MLP(in_dimension, out_dimension)
    criterion = nn.BCEWithLogitsLoss()

    init_lr=hyperparams['lr']
    warmup=hyperparams['warmup_iters']
    decay=hyperparams['lr_decay_iters']
    min_lr=hyperparams['min_lr']
    beta2=hyperparams['beta2']

    vector_params = [p for p in model.parameters() if len(p.shape)<=1]
    mat_params = [p for p in model.parameters() if len(p.shape)>1]
    optimizer = make_optimizer(optimizer, mat_params, hyperparams)
    sgd = opt.SGD(vector_params, lr=init_lr)
    #optimizer = optimizer(W=[p for p in model.parameters()], **hyperparams)
    #optimizer = opt.SGD([p for p in model.parameters()])
    iter_num=0

    torch.manual_seed(rand_seed)
    x_uniform, x_normal = get_uniform_normal_vectors(total_train_samples, in_dimension)
    x=torch.cat((x_uniform, x_normal), dim=0)
    y=torch.zeros((total_train_samples,1), dtype=torch.float32)
    y[total_train_samples//2:]=1.0
    train_ds=TensorDataset(x,y)
    train_dl=DataLoader(train_ds,batch_size,shuffle=True)
    
    x_test_uniform, x_test_normal = get_uniform_normal_vectors(total_test_samples, in_dimension)
    x=torch.cat((x_test_uniform, x_test_normal), dim=0)
    y=torch.zeros((total_test_samples,1), dtype=torch.float32)
    y[total_test_samples//2:]=1.0
    test_ds=TensorDataset(x,y)
    test_dl=DataLoader(test_ds,batch_size,shuffle=True)
    num_batches=np.ceil(total_train_samples/batch_size)

    s=time.time()

    count=0
    loss=0.0
    err=0
    loss_arr=[]
    while True:
        lr = get_lr(iter_num, init_lr, warmup*max_iters, decay*max_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        loss=0.0
        err=0
        for batch, labels in train_dl:
            temp=model(batch)
            L=criterion(temp, labels)
            #L=torch.sum(torch.norm(temp-labels,dim=1))/batch_size
            L.backward()
            loss+=L.item()
            probabilities = torch.sigmoid(temp)
            #print((probabilities > 0.5).float()-labels)
            diff=(probabilities > 0.5).float()-labels
            #print(torch.nonzero(diff))
            err+=torch.count_nonzero(diff).item()
        loss/=num_batches
        if debug:
            print(f"LOSS at iter {iter_num}: {loss}, ERROR: {err}/{.8*total_train_samples}")
        loss_arr.append(loss)
        optimizer.step()
        sgd.step()
        optimizer.zero_grad(set_to_none=True)
        sgd.zero_grad()
        iter_num+=1
        if loss<=.1 or (len(loss_arr) >= 2 and abs(loss-loss_arr[-2])<1e-4):
            count+=1
        else:
            count=0
        if iter_num>max_iters: #5, 10
            break
    e=time.time()
    print(loss, e-s, err, init_lr, name, "fails: ", optimizer.fails)
    if debug:
        print("TRAINLOSS: ", loss)
        print("TRAINERROR: ", err)

    val_err=0
    if validate:
        loss=0
        err=0
        for batch, labels in test_dl:
            temp=model(batch)
            L=criterion(temp, labels)
            L.backward()
            loss+=L.item()
            probabilities = torch.sigmoid(temp)
            diff=(probabilities > 0.5).float()-labels
            err+=torch.count_nonzero(diff).item()
        val_err=err/total_test_samples
        if debug:
            print("TESTLOSS: ", loss)
            print("VALIDATIONERROR: ", err, .2*total_train_samples)
    print("TESTLOSS: ", loss, err, total_test_samples)
    
    if name=="CS":
        print(optimizer.fails)
    return loss_arr, e-s, val_err, iter_num

#if __name__=='__main__':
    # hp={'lr': .6, 'warmup_iters':.01, 'lr_decay_iters':.05,'min_lr':6e-1,'beta2':.85}
    # hp['chol']=True
    # for optimizer in [CustomShampoo, CustomShampoo]:
    #     if hp['chol']==True:
    #         hp['chol']=False
    #     else:
    #         hp['chol']=True
    #     arr, t = trainMLPClassifier(optimizer, hp, 10, 1, total_samples=100, batch_size=10,  max_iters=1000)

    # model = "MLPClassifier"
    # total_samples=1000
    # test_samples=100
    # model={"model": model, "n": total_samples, "in_dimension": 10, "out_dimension": 1, "total_samples": total_samples, "test_samples": test_samples, "batch_size": 100}
    # for op in [OPTS.S, OPTS.CS, OPTS.SGD]:
    #     output, hp=grid_search(op, model, fine_grid)
    #     print(output)
    #     print(hp)
    #     with open(f"data/optimalHyperParams/{model['model']}(total_samples={total_samples})_{op.name}_hp.json", 'w') as f:
    #         json.dump(hp, f)