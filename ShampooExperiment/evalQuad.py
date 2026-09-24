from token import OP

import scipy as sp

from TrainingScripts import *

if __name__=='__main__':
    n=100
    i=0
    a=""
    times={}
    for curr_optimizer in [OPTS.MUON, OPTS.STIEFEL_ADAM]:
    #for curr_optimizer in [OPTS.SS, OPTS.SCS]:
        optimizer=curr_optimizer
        if curr_optimizer!=OPTS.S and curr_optimizer!=OPTS.CS:
            a="New"
        else:
            a=""
        if curr_optimizer!=OPTS.EXS:
            with open(f"/Users/christopherlinder/Desktop/OptimizerTesting/data/optimalHyperParams/{a}Quad(n=100)_{curr_optimizer.name}_hp.json", 'r') as f:
                hyper_params=json.load(f)
                print(hyper_params)
        else:
            hyper_params={"lr": 0.5, "warmup_iters": 0.05, "lr_decay_iters": 0.7, "min_lr": 0.1, "beta2": 0.999}
        hyper_params['max_iters']=2000
        if curr_optimizer==OPTS.MUON:
            hyper_params['lr']=.5
            hyper_params['lr_decay_iters']=1
            #hyper_params['min_lr']=6e-5
            hyper_params['warmup_iters']=.00
            print("a")
        if curr_optimizer==OPTS.STIEFEL_ADAM:
            hyper_params['lr_decay_iters']=.08#.9 #kappa=1
            #hyper_params['lr_decay_iters']=.04#.9 #kappa=100k
            #hyper_params['min_lr']=.3
        # if curr_optimizer==OPTS.STIEFEL_ADAM and i==0:
        #     hyper_params['lr']=.4
        #     hyper_params['warmup_iters']=.5
        #     hyper_params['lr_decay_iters']=.45#.9 #kappa=1
        stats=False

        spec=[0,-5]
        for numIters in [20]:
            times[curr_optimizer]=0
            hyper_params['numIters']=numIters
            losses=[]
            runs=10
            cmap = plt.colormaps['tab20'] 
            colors = cmap(np.linspace(0, 1, runs))
            for j in range(2,3):
                hyper_params['grafting']=True
                #print("hp: ", hyper_params)
                loss, t, kappa, target = analysis_Quad(optimizer, hyper_params, n, rand_seed=j, spectrum=spec, eye=False)
                losses.append(np.log(loss))
                times[curr_optimizer]+=t
                if not stats:
                    plt.plot(np.arange(len(loss)), np.log(loss), color=colors[i], label=f"{optimizer.name}, {t:.2f}_sec, det={torch.det(target)}")
            i+=1

            if stats:
                mean=np.mean(losses,axis=0)
                std=np.std(losses,axis=0)
                times[curr_optimizer]/=runs
                temp=0
                for v in std:
                    temp+=np.mean(v)
                print("std: ", temp/runs)
                plt.plot(np.arange(len(mean)), mean, label=f"{curr_optimizer.name}")
                plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.5)
                print(f"mean time {optimizer.name}: {times[curr_optimizer]}")
    plt.xlabel('iter')
    plt.ylabel('Log Loss (base 10)')
    plt.title(rf'Muon vs StiefelAdam-Quadratic Problem with $\kappa={kappa:.2f}$')
    plt.legend()
    #ax=plt.subplot(111)
    #ax.legend(bbox_to_anchor=(.5, -.15), loc='lower center', ncol=3)
    #plt.tight_layout()
    plt.show()