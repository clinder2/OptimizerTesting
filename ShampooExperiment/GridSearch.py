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
    'beta2': [.85,.999]
}

coarse_grid = {
    'lr': [.01, .1, .2,.9],
    'warmup_iters': [.1,.2,.3],
    'lr_decay_iters': [.2,.3,.4],
    'min_lr': [6e-5,6e-2,1e-2],
    'max_iters': [4000],
    'beta2': [.85,.999]
}

def grid_search(optimizer, model, grid=grid, num_workers=16):
    hyperparams={'lr': 0, 'warmup_iters': 0, 'lr_decay_iters': 0, 'min_lr': 0}

    hp_list=itertools.product(grid['lr'], grid['warmup_iters'], 
                              grid['lr_decay_iters'], grid['min_lr'],
                              grid['max_iters'], grid['beta2'])
    hp_list=[h for h in hp_list if h[1]!=h[2]]

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
                grid_Search_Quad, 
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

    hyperparams=min(output, key=lambda x: x['loss'])
    return output, hyperparams


def grid_Search_MLPClassifier(OP, hyperparams, in_dimension, out_dimension, total_samples=100, test_samples=100, batch_size=100, rand_seed=2):
    init_lr=hyperparams[0]
    warmup=hyperparams[1]
    decay=hyperparams[2]
    min_lr=hyperparams[3]
    max_iters=hyperparams[4]
    beta2=hyperparams[5]

    hp_dict={
        'lr': init_lr,
        'warmup_iters': warmup,
        'lr_decay_iters': decay,
        'min_lr': min_lr,
        'beta2': beta2,
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

if __name__=='__main__':
    model = "MLPClassifier"
    total_samples=100
    test_samples=10
    model={"model": model, "n": total_samples, "in_dimension": 10, "out_dimension": 1, "total_samples": total_samples, "test_samples": test_samples, "batch_size": 100}
    for op in [OPTS.S, OPTS.CS, OPTS.SGD]:
        output, hp=grid_search(op, model, fine_grid)
        print(output)
        print(hp)
        with open(f"data/optimalHyperParams/{model['model']}(total_samples={total_samples})_{op.name}_hp.json", 'w') as f:
            json.dump(hp, f)

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
    
    