from token import OP

import scipy as sp

from TrainingScripts import *

if __name__=='__main__':
    n=100
    rand_seed=20
    colors = ['green', 'red', 'blue', 'orange', 'black']
    i=0
    a=""
    for curr_optimizer in [OPTS.EXS, OPTS.MUON, OPTS.STIEFEL_ADAM, OPTS.STIEFEL_SGD]:
    #for curr_optimizer in [OPTS.EXS, OPTS.SS]:
        optimizer=curr_optimizer
        if curr_optimizer!=OPTS.S and curr_optimizer!=OPTS.CS:
            a="New"
        else:
            a=""
        if curr_optimizer!=OPTS.EXS:
            with open(f"/Users/christopherlinder/Desktop/OptimizerTesting/data/optimalHyperParams/{a}Quad(n=100)_{curr_optimizer.name}_hp.json", 'r') as f:
                hyper_params=json.load(f)
        else:
            hyper_params={"lr": 0.5, "warmup_iters": 0.05, "lr_decay_iters": 0.7, "min_lr": 0.1, "beta2": 0.999}
        hyper_params['max_iters']=2000
        # if curr_optimizer==OPTS.MUON:
        #     #hyper_params['lr']=.9
        #     hyper_params['min_lr']=.05

        spec=[0,-.000005]
        losses=[]
        for j in range(1):
            hyper_params['grafting']=True
            loss, t, kappa = analysis_Quad(optimizer, hyper_params, n, rand_seed=j, spectrum=spec)
            losses.append(np.log(loss))
            #plt.plot(np.arange(len(loss)), np.log(loss), color=colors[i], label=f"{optimizer.name}_{t:.2f}_sec_graft={hyper_params['grafting']}")
            i+=1

        mean=np.mean(losses,axis=0)
        std=np.std(losses,axis=0)
        temp=0
        for v in std:
            temp+=np.mean(v)
        print("std: ", temp/10)
        plt.plot(np.arange(len(mean)), mean, label=f"mean_{curr_optimizer.name}_std={temp/10}")
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.5, label=f"{curr_optimizer.name}_std")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(rf'Mean of 10 runs-{hyper_params["max_iters"]} iters-$\kappa={kappa}$')
    plt.legend()
    plt.show()