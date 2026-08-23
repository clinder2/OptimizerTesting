import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from TrainingScripts import *
from MLPClassifier import *

###for MLP rand and mult
grid = {
    'lr': [.001, .01, .1, .8, .9, .99],
    'warmup_iters': [.1,.2,.3,.4],
    'lr_decay_iters': [.05,.1,.2,.3,.4],
    'min_lr': [6e-5,6e-2,1e-2,1e-1],
    'max_iters': [2000]
}

fine_grid = {
    'lr': [.0001, .001, .01, .1, .2, .8, .9, .99],
    'warmup_iters': [.05,.1,.2,.3,.4],
    'lr_decay_iters': [.05,.1,.2,.3,.4,.7],
    'min_lr': [6e-5,6e-4,6e-2,1e-2,1e-1],
    'max_iters': [4000],
    'betas': [.85, .999],
}

# Extended fine grid including Muon / Stiefel hyperparameters
updated_fine_grid = {
    'lr': [.0001, .001, .01, .1, .9, .99],
    'warmup_iters': [.05, .1, .4],
    'lr_decay_iters': [.05, .2, .7],
    'min_lr': [6e-5, 1e-1],
    'max_iters': [2000],
    # Use betas pairs for optimizers that expose (beta, beta2)-style args
    'betas': [(0.9, 0.999), (0.8, 0.95), (0.7, 0.999)],
    'momentum': [0.8, 0.9, 0.95],
    'weight_decay': [1e-6, 1e-5],
}

SCS_updated_fine_grid = {
    'lr': [.0001, .001, .01, .1, .9, .99],
    'warmup_iters': [.05, .1, .4],
    'lr_decay_iters': [.05, .2, .7],
    'min_lr': [6e-5, 1e-1],
    'max_iters': [2000],
    'beta2': [0.999, 0.95, .8],
}

Muon_updated_fine_grid = {
    'lr': [.0001, .001, .01, .99]+list(np.arange(.1,1,.05)),
    'warmup_iters': list(np.arange(.05,1,.05)),
    'lr_decay_iters': list(np.arange(.05,1,.05)),
    'min_lr': [6e-5, 1e-1],
    'max_iters': [2000],
    'beta2': [0.999, 0.95, .8],
    'momentum': [0.8, 0.9, 0.95],
    'weight_decay': [1e-6, 1e-5],
}

stiefelSGD_updated_fine_grid = {
    'lr': [.0001, .001, .01, .1, .9, .99],
    'warmup_iters': [.05, .1, .4],
    'lr_decay_iters': [.05, .2, .7],
    'min_lr': [6e-5, 1e-1],
    'max_iters': [2000],
    # Use betas pairs for optimizers that expose (beta, beta2)-style args
    'momentum': [0.8, 0.9, 0.95],
    'weight_decay': [1e-6, 1e-5],
}

stiefelAdam_updated_fine_grid = {
    'lr': [.0001, .001, .01, .1, .9, .99],
    'warmup_iters': [.05, .1, .4],
    'lr_decay_iters': [.05, .2, .7],
    'min_lr': [6e-5, 1e-1],
    'max_iters': [2000],
    # Use betas pairs for optimizers that expose (beta, beta2)-style args
    'betas': [(0.9, 0.999), (0.8, 0.95), (0.7, 0.999)],
}

AdamW_updated_fine_grid = {
    'lr': [.0001, .001, .01, .1, .9, .99],
    'warmup_iters': [.05, .1, .4],
    'lr_decay_iters': [.05, .2, .7],
    'min_lr': [6e-5, 1e-1],
    'max_iters': [2000],
    # Use betas pairs for optimizers that expose (beta, beta2)-style args
    'betas': [(0.9, 0.999), (0.8, 0.95), (0.7, 0.999)],
    'weight_decay': [1e-6, 1e-5],
}

SGD_updated_fine_grid = {
    'lr': [.0001, .001, .01, .1, .9, .99],
    'warmup_iters': [.05, .1, .4],
    'lr_decay_iters': [.05, .2, .7],
    'min_lr': [6e-5, 1e-1],
    'max_iters': [2000],
    'momentum': [0.8, 0.9, 0.95],
    'weight_decay': [1e-6, 1e-5],
}

updated_coarse_grid = {
    'lr': [.01],
    'warmup_iters': [.05],
    'lr_decay_iters': [.2],
    'min_lr': [6e-5],
    'max_iters': [4000],
    # Use betas pairs for optimizers that expose (beta, beta2)-style args
    'betas': [(0.9, 0.999)],
    'momentum': [0.8],
    'weight_decay': [0.0],
}

coarse_grid = {
    'lr': [.01, .1, .2,.9],
    'warmup_iters': [.1,.2,.3],
    'lr_decay_iters': [.2,.3,.4],
    'min_lr': [6e-5,6e-2,1e-2],
    'max_iters': [4000],
    'beta': [.85,.999]
}

def grid_search(optimizer, model, grid=grid, num_workers=16):
    # support optional hyperparameters like 'momentum', 'weight_decay', 'betas'
    # Prefer 'betas' (pairs) when available; fall back to single 'beta' or legacy 'beta2'
    ordered_keys=['lr','warmup_iters','lr_decay_iters','min_lr','max_iters']
    # prefer betas if provided
    if 'betas' in grid:
        ordered_keys.append('betas')
    elif 'beta' in grid:
        ordered_keys.append('beta')
    elif 'beta2' in grid:
        ordered_keys.append('beta2')
    # then other optional keys
    for k in ['momentum','weight_decay']:
        if k in grid:
            ordered_keys.append(k)

    hp_product=itertools.product(*(grid[k] for k in ordered_keys))
    # filter where warmup == decay if both present
    hp_list=[]
    for h in hp_product:
        d=dict(zip(ordered_keys,h))
        if 'warmup_iters' in d and 'lr_decay_iters' in d and d['warmup_iters']==d['lr_decay_iters']:
            continue
        hp_list.append(d)

    ctx=mp.get_context("spawn")

    with ctx.Pool(num_workers) as pool:
        if model['model']=="MLP2":
            output=pool.starmap(
                trainMLP2, 
                [
                    (
                        optimizer,
                        hp,
                        model['n'],
                        model['h'],
                        model['mult'],
                    )
                    for hp in hp_list
                ]
            )
        elif model['model'] == 'Quad':
            output=pool.starmap(
                _safe_grid_Search_Quad,
                [
                    (
                        optimizer,
                        hp,
                        model['n'],
                    )
                    for hp in hp_list
                ]
            )
        elif model['model'] == 'MLPClassifier':
            output=pool.starmap(
                grid_Search_MLPClassifier,
                [
                    (
                        optimizer,
                        hp,
                        model['in_dimension'],
                        model['out_dimension'],
                        model.get('total_samples', 100),
                        model.get('test_samples', 100),
                        model.get('batch_size', 100),
                    )
                    for hp in hp_list
                ]
            )

    # Filter out any worker errors and raise if all workers failed.
    successful = [x for x in output if isinstance(x, dict)]
    if not successful:
        raise RuntimeError(f"All worker tasks failed. Sample errors: {output[:5]}")
    hyperparams = min(successful, key=lambda x: x['loss'])
    return output, hyperparams


def _safe_grid_Search_Quad(OP, hyperparams, n, rand_seed=2):
    try:
        return grid_Search_Quad(OP, hyperparams, n, rand_seed=rand_seed)
    except Exception as exc:
        import traceback
        return {
            'error': repr(exc),
            'traceback': traceback.format_exc(),
            'hyperparams': hyperparams,
        }


def grid_Search_MLPClassifier(OP, hyperparams, in_dimension, out_dimension, total_samples=100, test_samples=100, batch_size=100, rand_seed=2):
    init_lr=hyperparams['lr']
    warmup=hyperparams['warmup_iters']
    decay=hyperparams['lr_decay_iters']
    min_lr=hyperparams['min_lr']
    beta=hyperparams.get('beta', hyperparams.get('beta2'))
    max_iters = hyperparams['max_iters']

    hp_dict={
        'lr': init_lr,
        'warmup_iters': warmup,
        'lr_decay_iters': decay,
        'min_lr': min_lr,
        'beta': beta,
        'max_iters': max_iters,
    }

    loss_arr, elapsed, err = trainMLPClassifier(OP, hp_dict, in_dimension, out_dimension,
                                           max_iters=max_iters,
                                           total_train_samples=total_samples,
                                           total_test_samples=test_samples,
                                           batch_size=batch_size,
                                           rand_seed=rand_seed)

    hp_dict['loss']=loss_arr[-1] if len(loss_arr) else float('inf')
    hp_dict['time']=elapsed
    hp_dict['error']=err
    return hp_dict

# if __name__=='__main__':
#     model = "MLPClassifier"
#     total_samples=100
#     test_samples=10
#     model={"model": model, "n": total_samples, "in_dimension": 10, "out_dimension": 1, "total_samples": total_samples, "test_samples": test_samples, "batch_size": 100}
#     for op in [OPTS.S, OPTS.CS, OPTS.SGD]:
#         output, hp=grid_search(op, model, fine_grid)
#         print(output)
#         print(hp)
#         with open(f"data/optimalHyperParams/{model['model']}(total_samples={total_samples})_{op.name}_hp.json", 'w') as f:
#             json.dump(hp, f)

    # for O in [S_P2]:
        # m="MLP2"
        # n=50
        # mult=10
        # h=2*n
        # model={"model": m, "n": n, "h": h, "mult": mult}
        # output, hp=grid_search(O, model, coarse_grid)
        
        # print(hp)

        ###NAME hyperparameter dictionary json as MODEL_OPTIMIZER_hp.json
        ###MS(ntom)
        ###MLP(ntom-{rand, mult})
        ###WS-WhiteningShampoo, CS-CustomShampoo with chol=True, S-CustomShampoo
        ###with chol=False

        # match O:
        #     case 0:
        #         with open(f"data/{m}(n={n},h={h},mult={mult})_S_hp.json", 'w') as f:
        #             json.dump(hp, f)
        #         with open(f"data/{m}(n={n},h={h},mult={mult})_S_gridresults.json", 'w') as f:
        #             json.dump(output, f)
        #     case 1:
        #         with open(f"data/{m}(n={n},h={h},mult={mult})_CS_hp.json", 'w') as f:
        #             json.dump(hp, f)
        #         with open(f"data/{m}(n={n},h={h},mult={mult})_CS_gridresults.json", 'w') as f:
        #             json.dump(output, f)
        #     case 2:
        #         with open(f"data/{m}(n={n},h={h},mult={mult})_WS_hp.json", 'w') as f:
        #             json.dump(hp, f)
        #         with open(f"data/{m}(n={n},h={h},mult={mult})_WS_gridresults.json", 'w') as f:
        #             json.dump(output, f)
        #     case S_P2:
        #         with open(f"data/{m}(n={n},h={h},mult={mult})_CS-P2-torchinv_hp.json", 'w') as f:
        #             json.dump(hp, f)
        #         with open(f"data/{m}(n={n},h={h},mult={mult})_CS-P2-torchinv_gridresults.json", 'w') as f:
        #             json.dump(output, f)
    
    