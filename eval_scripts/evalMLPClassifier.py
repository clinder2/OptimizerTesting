import sys
import os

ShampooExperiment_PATH = os.environ.get(
    "SHAMPOO_PATH", os.path.expanduser("~/Desktop/OptimizerTesting/ShampooExperiment")
)
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# `common` lives at the project root, alongside `analysis`, rather than inside
# this script's directory.
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.TrainingScripts import *
from models.MLPClassifier import *
from eval_scripts.evalQuad import stiefel_hp

with open("/Users/christopherlinder/Desktop/OptimizerTesting/common/results/data/optimalHyperParams/MuonvSteifelAdamV1_Quad_(n=100)_MUON_hp.json") as f:
    muon_hp = json.load(f)

if __name__=='__main__':
    i=0
    in_dim = 10
    out_dim = 1
    total_samples=1000
    validation_samples = 100
    batch_size=100
    val_errors=[]
    experiments=2
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    optimizers = [OPTS.MUON, OPTS.JSE]
    cmap = plt.get_cmap('tab20', len(optimizers))

    for curr_optimizer in optimizers:
        losses=[]
        total_val_err=0
        for rand_seed in range(experiments):
            if curr_optimizer==OPTS.STIEFEL_ADAM:
                with open(f"common/results/data/optimalHyperParams/MuonvSteifelAdamV1_MLP_(n=100)_STIEFEL_ADAM_hp.json", 'r') as f:
                    hyper_params=json.load(f)
            else:
                with open(f"common/results/data/optimalHyperParams/MuonvSteifelAdamV1_MLP_(n=100)_MUON_hp.json", 'r') as f:
                    hyper_params=json.load(f)
            hyper_params['max_iters']=4000
                
            loss, t, err, iters = trainMLPClassifier(curr_optimizer, hyper_params, in_dim, out_dim, max_iters=hyper_params['max_iters'], 
                rand_seed=rand_seed, total_train_samples=total_samples, total_test_samples=validation_samples, 
                batch_size=batch_size, debug=False, validate=True)
            losses.append(loss)
            total_val_err += err
             
        total_val_err/=experiments
        val_errors.append(total_val_err)
        print(f"VAL ERROR {curr_optimizer.name}: {total_val_err}")
        mean=np.log(np.mean(losses, axis=0))
        axes[0].plot(np.arange(len(mean)), mean, color=cmap(i%6), linewidth=2, label=f"{curr_optimizer.name} mean")
        std = np.log(np.std(losses, axis=0))
        axes[0].fill_between(np.arange(len(mean)), mean - std, mean + std, color=cmap((i)%6), alpha=0.5, label=f"{curr_optimizer.name} std")
        i+=1
    axes[0].set_title(f"Binary Classification-dimension={in_dim}-{experiments} runs-muon and stiefelAdam")
    axes[1].set_title(f"Binary Classification-{experiments} runs-validation loss")
    axes[1].bar([op.name for op in optimizers], val_errors)
    print(val_errors)
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Log Loss')
    axes[1].set_xlabel('Optimizer')
    axes[1].set_ylabel('Misclassification Rate on Held-out Set')
    axes[0].legend()
    plt.show()