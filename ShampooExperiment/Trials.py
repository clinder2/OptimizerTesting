# from model import MatrixSimple, MLP, ComplicatedMLP
# import torch
# import torch.optim as opt
# import torch.multiprocessing as mp
# import math, time, copy, json, itertools
# from CustomShampoo import CustomShampoo
# from WhiteningShampoo import WhiteningShampoo
# from GridSearch import get_lr, trainMLP2
# from Base import *
from TrainingScripts import *

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
    max_iters=4000

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

def eval_MLP2(path, n, h, mult, batch_size, samples, i):
    model=MLP2(n,n,h)
    model.load_state_dict(torch.load(path))
    model.eval()

    torch.manual_seed(i)
    x=torch.rand(samples,n)
    y=mult*x
    dl=TensorDataset(x,y)
    ds=DataLoader(dl,batch_size,True)

    loss=0.0
    for f, l in ds:
        temp=model(f)
        L=torch.sum(torch.norm(temp-l,dim=1))/batch_size
        loss+=L.item()
    loss/=(samples//batch_size)
    print(loss)

def plot_TrainTestLoss():
    labels=['Shampoo', 'CholeskyS']

    x = np.arange(2)  # label locations
    width = 0.35  # bar width

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, [.016,.18], width, label='Train Loss', color='skyblue')
    rects2 = ax.bar(x + width/2, [1.42,1.33], width, label='Test Loss', color='salmon')

    # rects3 = ax.bar(x - width/2, .18, width, color='skyblue')
    # rects4 = ax.bar(x + width/2, 1.33, width, color='salmon')

    # Add labels and title
    ax.set_ylabel('Loss Value')
    ax.set_title('Training vs Testing Loss Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add values on top of bars
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    # ax.bar_label(rects3, padding=3)
    # ax.bar_label(rects4, padding=3)

    plt.show()

def plot_losses():
    a=np.load("data/losses/quad-n=100-S.npy")
    b=np.load("data/losses/quad-n=100-CS.npy")
    c=np.load("data/losses/quad-n=100-S_P2.npy")
    d=np.load("data/losses/quad-n=100-SGD.npy")

    a=np.load(f"data/losses/MLP2(n={n},h={h},mult={mult})-S.npy")
    b=np.load(f"data/losses/MLP2(n={n},h={h},mult={mult})-CS.npy")
    c=np.load(f"data/losses/MLP2(n={n},h={h},mult={mult})-S_P2.npy")
    plt.plot(np.arange(d.shape[0]), d,color='blue',label='SGD(10000 steps, 0.86 sec)')
    plt.plot(np.arange(a.shape[0]), a,color='orange',label='Shampoo(7228 steps, 19.90 sec)')
    plt.plot(np.arange(b.shape[0]), b,color='green',label='CholeskyS(3928 steps, 12.44 sec)')
    plt.plot(np.arange(c.shape[0]), c,color='red',label='CholeskyS(p=1/2)(5951 steps, 13.73 sec)')
    plt.legend()
    plt.show()

if __name__=='__main__':
    ev=False
    if ev:
        O=SCI
        n=50
        h=2*n
        mult=10
        path=f"/Users/christopherlinder/Desktop/OptimizerTesting/data/models/MLP2(n={n},h={h},mult={mult})_{O}_hp"
        eval_MLP2(path,n,h,mult,batch_size=10,samples=100,i=3)        
    else:
        m="MLP2"
        n=50
        mult=10
        h=2*n
        losses=[]
        hps=[]
        for O in [SCI]:
            if O==S:
                with open(f"data/hyperparams/{m}(n={n},h={h},mult={mult})_S_hp.json", 'r') as f:
                    hp=json.load(f)
            elif O==CS:
                with open(f"data/hyperparams/{m}(n={n},h={h},mult={mult})_CS_hp.json", 'r') as f:
                    hp=json.load(f)
            elif O==WS:
                with open(f"data/hyperparams/{m}(n={n},h={h},mult={mult})_WS_hp.json", 'r') as f:
                    hp=json.load(f)
            else:
                with open(f"data/hyperparams/{m}(n={n},h={h},mult={mult})_CS-P2-torchinv_hp.json", 'r') as f:
                    hp=json.load(f)
            
            hp={'lr': 0.6, 'warmup_iters': 0.2, 'lr_decay_iters': 0.3, 'min_lr': 6e-5, 'beta2': 0.999, 'max_iters': 1596, 'loss': 0.0017872139578685164, 'time': 5.994225025177002}
            
            trials=1 #50
            tot_loss=0.0
            tot_time=0.0
            
            currhp, loss_arr=trainMLP2(
                            O,
                            hp,
                            n,
                            h,
                            mult,
                            1000,
                            100,
                            3,
                        )
            losses.append(loss_arr)
            hps.append(currhp)
    # np.save(f"data/losses/MLP2(n={n},h={h},mult={mult})-S.npy", np.array(losses[0]))
    # np.save(f"data/losses/MLP2(n={n},h={h},mult={mult})-CS.npy", np.array(losses[1]))
    # np.save(f"data/losses/MLP2(n={n},h={h},mult={mult})-S_P2.npy", np.array(losses[2]))

###IF UNSPECIFIED max_iters=2000

###rand MLP###
###Shampoo results: tot_loss=0.00011033530237909872, tot_time=17.53478723526001
###Shampoo-chol=True results: tot_loss=0.0001562694059975911, tot_time=17.704886736869813
###WS-pure=True results: tot_loss=0.0005598210134121473, tot_time=10.261562476158142

###mult MLP###
###50x50###
###Shampoo results: tot_loss=0.0009620271751191467, tot_time=32.41621156692505
###Shampoo-chol=True results: tot_loss=0.0009825874387752265, tot_time=23.271225261688233
###2000iters_WS-pure=True results: tot_loss=0.12119188532233238, tot_time=17.216809701919555
###4000iters_WS-pure=True results: tot_loss=0.10710389330983162, tot_time=36.720536155700685

###2000iters_S-P2 results: tot_loss=0.0011928852161508985, tot_time=35.248321561813356

###100x100###
###8000iters_S results: tot_loss=0.0008305382245453075, tot_time=151.40726024627685
###8000iters_CS results: tot_loss=0.0012912023917306214, tot_time=99.86386492729187
###8000iters_WS-pure=True results: tot_loss=0.04123151498381048, tot_time=88.64280379295349

###4000iters_CS results: tot_loss=0.001590018289280124, tot_time=62.48247771263122

###BATCH,n=10,mult=2
#S: loss=0.18353414988145234, time=4.235884518623352
#CS: loss=0.35371837466955186, time=7.00960045337677
#WS: loss=0.5060925925988704, time=4.35161069393158
#S-P2: loss=0.24939304580911995, time=7.337163028717041

###BATCH,n=20,mult=2
#S: loss=0.04624248451553285, time=1.7281010723114014
#CS: loss=0.008099197305273264, time=5.631789455413818
#WS: loss=4.587289538383484, time=14.80098394870758
#S-P2: loss=0.45296312229707836, time=12.071191849708557

###BATCH,n=50,mult=2
#S: loss=0.0014067116437945515, time=22.640675687789916
#CS: loss=0.0019941365183331074, time=5.646757555007935
#WS: loss=7.636781015396118, time=75.99598598480225
#S-P2: loss=0.5123092622123658, time=42.90474042415619