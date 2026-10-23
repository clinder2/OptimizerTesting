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
    total_samples=100
    validation_samples = 100
    batch_size=10
    val_errors=[]
    experiments=1
    cmap = plt.get_cmap('tab20', 2*experiments)
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))

    for curr_optimizer in [OPTS.SGD, OPTS.JSE]:
        losses=[]
        total_val_err=0
        for rand_seed in range(experiments):
            optimizer=curr_optimizer
            if curr_optimizer==OPTS.MUON or curr_optimizer==OPTS.JSE:
                hyper_params=muon_hp
                hyper_params['lr']=.05
                # hyper_params['decay_iters']=.8
            else:
                hyper_params=stiefel_hp
                # with open(f"/Users/christopherlinder/Desktop/OptimizerTesting/data/optimalHyperParams/Quad(n=100)_{optimizer.name}_hp.json", 'r') as f:
                #     hyper_params=json.load(f)
            # else:
            #     with open(f"data/optimalHyperParams/{model['model']}(total_samples=100)_{optimizer.name}_hp.json", 'r') as f:
            #         hyper_params=json.load(f)
            if curr_optimizer==OPTS.STIEFEL_ADAM:
                hyper_params['lr']=.5
            #hyper_params['lr']=.5 if curr_optimizer==OPTS.MUON else hyper_params['lr']
            hyper_params['max_iters']=2000
                
            loss, t, err, iters = trainMLPClassifier(optimizer, hyper_params, in_dim, out_dim, max_iters=hyper_params['max_iters'], 
                rand_seed=rand_seed, total_train_samples=total_samples, total_test_samples=validation_samples, 
                batch_size=batch_size, debug=True, validate=True)
            losses.append(loss)
            total_val_err += err
            if experiments==1:
                plt.plot(loss, label=f"{curr_optimizer.name}")
            #plt.plot(np.arange(len(loss)), loss, color=cmap(rand_seed%20), label=f"{optimizer.name}_{t:.2f}_seconds_val-err={err:.2f}")

        if experiments>1:
            total_val_err/=experiments
            val_errors.append(total_val_err)
            print(f"VAL ERROR {curr_optimizer.name}: {total_val_err}")
            #losses=torch.nn.utils.rnn.pad_sequence(losses, True)
            mean=np.log(np.mean(losses, axis=0))
            axes[0].plot(np.arange(len(mean)), mean, color=cmap(i%6), linewidth=2, label=f"{optimizer.name} mean")
            std = np.log(np.std(losses, axis=0))
            axes[0].fill_between(np.arange(len(mean)), mean - std, mean + std, color=cmap((i)%6), alpha=0.5, label=f"{optimizer.name} std")
            #plt.title(f'{optimizer.name}-{mean_iters:.2f} iters, Validation Error: {total_val_err:.2f}')
            i+=1
            #np.save(f"data/optimalHyperParams/{model['model']}(total_samples=100)_{optimizer.name}_mean.npy", mean)
    axes[0].set_title(f"Binary Classification-MLP {experiments} runs-muon and stiefelAdam")
    axes[1].bar(np.arange(len(val_errors)), val_errors)
    print(val_errors)
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    axes[0].legend()
    plt.show()