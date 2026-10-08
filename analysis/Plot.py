import os
import mplcursors

from TrainingScripts import *

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir, "data"))
HP_DIR = os.path.join(DATA_DIR, "optimalHyperParams")

muon_hp={
  "lr": 0.9,
  "warmup_iters": 0.1,
  "lr_decay_iters": 0.4,
  "min_lr": 0.00006,
  "max_iters": 999,
  "beta2": 0.8,
  "momentum": 0.8,
  "weight_decay": 0.00001,
  "beta": 0.8,
  "optimizer": "MUON",
  "rand_seed": 2,
  "loss": 0.000002680559418877,
  "time": 3.33208513259888
}

stiefel_hp={
  "lr": 0.99,
  "warmup_iters": 0.05,
  "lr_decay_iters": 0.3,
  "min_lr": 0.06,
  "max_iters": 999,
  "betas": [0.7, 0.999],
  "beta": 0.999,
  "optimizer": "STIEFEL_ADAM",
  "rand_seed": 2,
  "loss": 1.88961108045504e-12,
  "time": 10.6982145309448
}

if __name__=='__main__':
    n=100
    max_iters=2000

    seeds=10
    kappa_range=5
    optimizers=[OPTS.MUON, OPTS.STIEFEL_ADAM]
    cmap=plt.colormaps['tab20']
    colors=cmap(np.linspace(0, 1, kappa_range+1))
    kappa=[]

    results={o.name:seeds*[kappa_range*[0]] for o in optimizers}
    for color, curr_optimizer in zip(colors, optimizers):
        if curr_optimizer==OPTS.MUON:
            hyper_params=muon_hp
        elif curr_optimizer==OPTS.STIEFEL_ADAM:
            hyper_params=stiefel_hp
            hyper_params['lr_decay_iters']=.9
        hyper_params['max_iters']=max_iters
        for s in range(seeds):
            for j in range(kappa_range):
                spectrum=[0,-j]
                stats=analysis_Quad_LossDecrease(curr_optimizer, hyper_params, n, rand_seed=s, spectrum=spectrum, eye=False)
                if len(kappa)<kappa_range:
                    kappa.append(stats['kappa'].detach().numpy())
                results[curr_optimizer.name][s][j]=stats['loss'][-1]
                print(results[curr_optimizer.name][s][j], stats['loss'][-1])
                print(f"{curr_optimizer.name}: time={stats['time']:.2f}s, final_loss={stats['loss'][-1]:.6g}")

    # --- Loss curve  ---
    plt.figure()
    for idx, (name, stats) in enumerate(results.items()):
        loss = np.mean(np.array([np.log(s) for s in stats]),axis=0)
        std = np.std(np.array([np.log(s) for s in stats]),axis=0)
        # mean_time = np.sum([s['time'] for s in stats])/len(stats)
        color = colors[idx]
        print(loss, std, stats)
        #plt.plot(kappa, np.log(stats[0]), color=color, label=f"{name}")
        
        plt.plot(kappa, loss, color=color, label=f"{name}")
        plt.fill_between(kappa, loss - std, loss + std, alpha=0.5, color=color)
    mplcursors.cursor(hover=True)
    plt.xlabel(r'$\kappa$')
    plt.ylabel('Average Log Loss (base 10)')
    plt.title(rf'Muon vs StiefelAdam-Quadratic Problem over {kappa_range} $\kappa$ values')
    plt.legend()
    plt.show()