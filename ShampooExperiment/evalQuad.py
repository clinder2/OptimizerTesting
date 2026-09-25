import copy
import os

from TrainingScripts import *

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir, "data"))
HP_DIR = os.path.join(DATA_DIR, "optimalHyperParams")

if __name__=='__main__':
    n=100
    rand_seed=2
    spectrum=[0,0]  # kappa ~= 1 (well-conditioned target)
    max_iters=2000

    # StiefelAdam's own fine-grid hyperparameter sweep is still running on
    # PACE. Until that completes, use Muon's swept hyperparameters for BOTH
    # optimizers so the loss-decrease comparison below is apples-to-apples
    # (same lr schedule / iteration budget). Once StiefelAdam's sweep lands,
    # go back to loading each optimizer's own hp json separately.
    with open(os.path.join(HP_DIR, "NewQuad(n=100)_MUON_hp.json"), 'r') as f:
        shared_hp=json.load(f)
    shared_hp['max_iters']=max_iters

    optimizers=[OPTS.MUON, OPTS.STIEFEL_ADAM]
    cmap=plt.colormaps['tab20']
    colors=cmap(np.linspace(0, 1, len(optimizers)))

    results={}
    for color, curr_optimizer in zip(colors, optimizers):
        hyper_params=copy.deepcopy(shared_hp)
        stats=analysis_Quad_LossDecrease(curr_optimizer, hyper_params, n, rand_seed=rand_seed, spectrum=spectrum, eye=False)
        results[curr_optimizer.name]=(stats, color)
        print(f"{curr_optimizer.name}: time={stats['time']:.2f}s, final_loss={stats['loss'][-1]:.6g}")

    kappa=results[optimizers[0].name][0]['kappa']

    # --- Loss curve (as before) ---
    plt.figure()
    for name, (stats, color) in results.items():
        plt.plot(np.arange(len(stats['loss'])), np.log(stats['loss']), color=color,
                  label=f"{name}, {stats['time']:.2f}_sec")
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
    for name, (stats, color) in results.items():
        axes[0].plot(stats['first_order'], color=color, label=f"{name} first-order <G,Z>")
        axes[0].plot(stats['curvature'], color=color, linestyle='--', label=f"{name} curvature 1/2<Z,H[Z]>")

        cum_first=np.cumsum(stats['first_order'])
        cum_curv=np.cumsum(stats['curvature'])
        axes[1].plot(cum_first, color=color, label=f"{name} cumulative first-order")
        axes[1].plot(cum_curv, color=color, linestyle='--', label=f"{name} cumulative curvature")

        total_first=float(np.sum(stats['first_order']))
        total_curv=float(np.sum(stats['curvature']))
        total_decrease=stats['loss'][0]-stats['loss'][-1]
        ratio=total_curv/total_first if total_first != 0 else float('nan')
        print(f"{name}: total loss decrease={total_decrease:.6g}, "
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
