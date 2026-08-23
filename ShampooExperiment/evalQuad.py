from token import OP

import scipy as sp

from TrainingScripts import *

if __name__=='__main__':
    n=100
    rand_seed=2
    colors = ['green', 'red', 'blue', 'orange', 'black']
    i=0
    a=""
    times={}
    for curr_optimizer in [OPTS.MUON, OPTS.STIEFEL_ADAM, OPTS.STIEFEL_SGD]:
    #for curr_optimizer in [OPTS.SS, OPTS.SCS]:
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
        hyper_params['max_iters']=3000
        if curr_optimizer==OPTS.MUON:
            # hyper_params['lr']=9
            # hyper_params['lr_decay_iters']=.1
            # hyper_params['min_lr']=.1
            print("a")
        if curr_optimizer==OPTS.STIEFEL_ADAM:
            #hyper_params['lr']=1
            hyper_params['lr_decay_iters']=.15#.9
        #     hyper_params['min_lr']=.05
        stats=False

        spec=[0,-5]
        for numIters in [20]:
            times[curr_optimizer]=0
            hyper_params['numIters']=numIters
            losses=[]
            runs=1
            for j in range(runs):
                hyper_params['grafting']=True
                print("hp: ", hyper_params)
                loss, t, kappa = analysis_Quad(optimizer, hyper_params, n, rand_seed=j, spectrum=spec)
                losses.append(np.log(loss))
                times[curr_optimizer]+=t
                if not stats:
                    plt.plot(np.arange(len(loss)), np.log(loss), color=colors[i], label=f"{optimizer.name}_{t:.2f}_sec_graft={hyper_params['grafting']}")
                i+=1

            if stats:
                mean=np.mean(losses,axis=0)
                std=np.std(losses,axis=0)
                times[curr_optimizer]/=runs
                temp=0
                for v in std:
                    temp+=np.mean(v)
                print("std: ", temp/runs)
                plt.plot(np.arange(len(mean)), mean, label=f"mean_{curr_optimizer.name}_std={temp/runs:.2f}_time={times[curr_optimizer]:.2f}_iters={numIters}")
                plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.5, label=f"{curr_optimizer.name}_std")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(rf'{hyper_params["max_iters"]} iters-$\kappa={kappa}$')
    plt.legend()
    #ax=plt.subplot(111)
    #ax.legend(bbox_to_anchor=(.5, -.15), loc='lower center', ncol=3)
    #plt.tight_layout()
    plt.show()