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
  "max_iters": 2000,
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
  "max_iters": 2000,
  "betas": [0.7, 0.999],
  "beta": 0.999,
  "optimizer": "STIEFEL_ADAM",
  "rand_seed": 2,
  "loss": 1.88961108045504e-12,
  "time": 10.6982145309448
}

if __name__=='__main__':
    n=100
    rand_seed=2
    spectrum=[0,0]  # kappa ~= 1 (well-conditioned target)
    max_iters=2000

    runs=1
    optimizers=[OPTS.MUON, OPTS.STIEFEL_ADAM]
    cmap=plt.colormaps['tab20']
    colors=cmap(np.linspace(0, 1, runs))

    results={o.name:[] for o in optimizers}
    stats = analysis_split(muon_hp, stiefel_hp, n, rand_seed=2, spectrum=spectrum)
    
    kappa=stats['kappa']

    # --- Loss curve  ---
    plt.figure()
    loss=stats['loss']
    plt.plot(np.arange(max_iters), loss, color=colors[0], label=f"layers")
    mplcursors.cursor(hover=True)
    plt.xlabel('iter')
    plt.ylabel('Log Loss (base 10)')
    plt.title(rf'Muon vs StiefelAdam-Quadratic Problem with $\kappa={float(kappa):.2f}$')
    plt.legend()
    plt.show()