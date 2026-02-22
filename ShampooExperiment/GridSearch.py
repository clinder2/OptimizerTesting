from TrainingScripts import *

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
    'max_iters': [4000]
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

    hyperparams=min(output, key=lambda x: x['loss'])
    return output, hyperparams

if __name__=='__main__':
    for O in [S_P2]:
        m="MLP2"
        n=50
        mult=10
        h=2*n
        model={"model": m, "n": n, "h": h, "mult": mult}
        output, hp=grid_search(O, model, coarse_grid)
        
        print(hp)
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
    
    