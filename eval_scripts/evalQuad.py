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
    rand_seed=2
    spectrum=[0,0]  # kappa ~= 1 (well-conditioned target)
    max_iters=2000

    runs=2
    optimizers=[OPTS.MUON, OPTS.STIEFEL_ADAM]
    cmap=plt.colormaps['tab20']
    colors=cmap(np.linspace(0, 1, runs))

    results={o.name:[] for o in optimizers}
    #results={OPTS.MUON.name:[], 'StiefelAdam-plateau':[], 'StiefelAdam':[]}
    for color, curr_optimizer in zip(colors, optimizers):
        if curr_optimizer==OPTS.MUON:
            hyper_params=muon_hp
        elif curr_optimizer==OPTS.STIEFEL_ADAM:
            hyper_params=stiefel_hp
            hyper_params['lr_decay_iters']=.9
        hyper_params['max_iters']=max_iters
        for j in range(runs):
            #spectrum=[0,-j]
            stats=analysis_Quad_LossDecrease(curr_optimizer, hyper_params, n, rand_seed=j, spectrum=spectrum, eye=False)
            # if curr_optimizer==OPTS.STIEFEL_ADAM and np.log(stats['loss'][-1])>0:
            #     results['StiefelAdam-plateau'].append(stats)
            # elif curr_optimizer==OPTS.STIEFEL_ADAM:
            #     results['StiefelAdam'].append(stats)
            # else:
            #     results[curr_optimizer.name].append(stats)
            results[curr_optimizer.name].append(stats)
            print(f"{curr_optimizer.name}: time={stats['time']:.2f}s, final_loss={stats['loss'][-1]:.6g}")

    kappa=results[optimizers[0].name][0]['kappa']

    # --- Loss curve  ---
    plt.figure()
    for idx, (name, stats) in enumerate(results.items()):
        loss = np.mean(np.array([np.log(s['loss']) for s in stats]),axis=0)
        std = np.std(np.array([np.log(s['loss']) for s in stats]),axis=0)
        mean_time = np.sum([s['time'] for s in stats])/len(stats)
        color = colors[idx]
        # for i,s in enumerate(stats):
        #     plt.plot(np.arange(max_iters), np.log(s['loss']), color=colors[i], label=f"{name},{i}")
        plt.plot(np.arange(max_iters), loss, color=color,
                  label=f"{name}, {mean_time:.2f}_sec")
        plt.fill_between(np.arange(max_iters), loss - std, loss + std, alpha=0.5, color=color)
    mplcursors.cursor(hover=True)
    plt.xlabel('iter')
    plt.ylabel('Log Loss (base 10)')
    plt.title(rf'Muon vs StiefelAdam-Quadratic Problem with $\kappa={float(kappa):.2f}$')
    plt.legend()

    # --- Empirical loss-decrease decomposition (arXiv 2606.04662v1 eq. 4.1) ---
    # ΔD(W, Z) = <G, Z> - 1/2 <Z, H[Z]>, where Z is the (negated) actual
    # parameter update taken by the optimizer at that step. For this
    # quadratic model, H is the constant operator H[Z] = 2*P@Z, so this
    # decomposition is EXACT (not a local approximation): it splits each
    # step's true loss decrease into a first-order gradient-alignment term
    # and a curvature penalty term, letting us compare how much curvature
    # "costs" each optimizer on the same target matrix / hyperparameters.
    fig, axes=plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    for idx, (name, trial_stats) in enumerate(results.items()):
        color=colors[idx]
        first_order=np.array([stats['first_order'] for stats in trial_stats])
        curvature=np.array([stats['curvature'] for stats in trial_stats])
        mean_first=np.mean(first_order, axis=0)
        std_first=np.std(first_order, axis=0)
        mean_curv=np.mean(curvature, axis=0)
        std_curv=np.std(curvature, axis=0)

        axes[0].plot(mean_first, color=color, label=f"{name} first-order <G,Z>")
        axes[0].fill_between(np.arange(mean_first.size), mean_first-std_first,
                             mean_first+std_first, color=color, alpha=0.2)
        axes[0].plot(mean_curv, color=color, linestyle='--',
                     label=f"{name} curvature 1/2<Z,H[Z]>")
        axes[0].fill_between(np.arange(mean_curv.size), mean_curv-std_curv,
                             mean_curv+std_curv, color=color, alpha=0.2)

        cumulative_first=np.cumsum(first_order, axis=1)
        cumulative_curv=np.cumsum(curvature, axis=1)
        mean_cumulative_first=np.mean(cumulative_first, axis=0)
        std_cumulative_first=np.std(cumulative_first, axis=0)
        mean_cumulative_curv=np.mean(cumulative_curv, axis=0)
        std_cumulative_curv=np.std(cumulative_curv, axis=0)
        axes[1].plot(mean_cumulative_first, color=color,
                     label=f"{name} cumulative first-order")
        axes[1].fill_between(np.arange(mean_cumulative_first.size),
                             mean_cumulative_first-std_cumulative_first,
                             mean_cumulative_first+std_cumulative_first,
                             color=color, alpha=0.2)
        axes[1].plot(mean_cumulative_curv, color=color, linestyle='--',
                     label=f"{name} cumulative curvature")
        axes[1].fill_between(np.arange(mean_cumulative_curv.size),
                             mean_cumulative_curv-std_cumulative_curv,
                             mean_cumulative_curv+std_cumulative_curv,
                             color=color, alpha=0.2)

        total_first=float(np.mean(np.sum(first_order, axis=1)))
        total_curv=float(np.mean(np.sum(curvature, axis=1)))
        total_decrease=float(np.mean([
            stats['loss'][0]-stats['loss'][-1] for stats in trial_stats
        ]))
        ratio=total_curv/total_first if total_first != 0 else float('nan')
        print(f"{name} (mean over {len(trial_stats)} trials): "
              f"total loss decrease={total_decrease:.6g}, "
              f"sum(first_order)={total_first:.6g}, sum(curvature)={total_curv:.6g}, "
              f"curvature/first_order ratio={ratio:.4f}")

    # symlog since per-step first_order/curvature can span several orders of
    # magnitude (large early in training, small once converged) while
    # curvature stays >= 0 (H = 2*P is PSD) and first_order can occasionally
    # dip slightly negative for a non-descent step.
    axes[0].set_yscale('symlog')
    axes[0].set_ylabel('per-step term (symlog)')
    axes[0].set_title(r'Per-step decomposition: $\langle G,Z\rangle$ vs $\frac{1}{2}\langle Z,H[Z]\rangle$')
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel('cumulative term')
    axes[1].set_xlabel('iter')
    axes[1].set_title('Cumulative first-order gain vs curvature penalty')
    axes[1].legend(fontsize=8)
    plt.tight_layout()

    plt.show()
