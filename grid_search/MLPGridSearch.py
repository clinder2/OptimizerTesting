import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Repo root plus dirs imported by bare name in GridSearch/TrainingScripts; spawn workers inherit sys.path.
for p in (ROOT_DIR, os.path.join(ROOT_DIR, "common"), os.path.join(ROOT_DIR, "models"), os.path.join(ROOT_DIR, "shampoo_variants")):
    if p not in sys.path:
        sys.path.insert(0, p)

from grid_search.GridSearch import *
# Avoid importing StiefelOptimizers by bare module name here.
#from StiefelOptimizers import MuonAdamW
#from GridSearch import updated_fine_grid, updated_coarse_grid, grid_search

# Compute output paths relative to the repo root instead of a hardcoded,
# machine-specific absolute path, so this script also works unmodified when
# deployed on another machine/cluster (e.g. PACE).
DATA_DIR = os.path.join(ROOT_DIR, "results", "data")
HP_DIR = os.path.join(DATA_DIR, "optimalHyperParams")
SWEEP_DIR = os.path.join(DATA_DIR, "hyperparams")
os.makedirs(HP_DIR, exist_ok=True)
os.makedirs(SWEEP_DIR, exist_ok=True)

if __name__=='__main__':
    model = "MLPClassifier"
    n=100 #100
    model={"model": model, 'in_dimension': 10,
        'out_dimension': 1, 'total_samples': n, 'batch_size': 10}

    # op=OPTS.MUON
    # output, hp=grid_search(
    #     op, model, Muon_coarse_grid,
    #     tsv_path=os.path.join(SWEEP_DIR, f"MuonvSteifelAdamV1_MLP_{n}_{op.name}_gridsweep.tsv"),
    # )
    # print(output)
    # print(hp)
    # with open(os.path.join(HP_DIR, f"MuonvSteifelAdamV1_MLP_(n={n})_{op.name}_hp.json"), 'w') as f:
    #     json.dump(hp, f)

    op=OPTS.STIEFEL_ADAM
    output, hp=grid_search(
        op, model, stiefelAdam_coarse_grid,
        tsv_path=os.path.join(SWEEP_DIR, f"MuonvSteifelAdamV1_MLP_{n}_{op.name}_gridsweep.tsv"),
    )
    print(output)
    print(hp)
    with open(os.path.join(HP_DIR, f"MuonvSteifelAdamV1_MLP_(n={n})_{op.name}_hp.json"), 'w') as f:
        json.dump(hp, f)
