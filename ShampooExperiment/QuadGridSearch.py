import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

#from Base import *
from GridSearch import *
# Avoid importing StiefelOptimizers by bare module name here.
#from StiefelOptimizers import MuonAdamW
#from GridSearch import updated_fine_grid, updated_coarse_grid, grid_search

# Compute output paths relative to the repo root instead of a hardcoded,
# machine-specific absolute path, so this script also works unmodified when
# deployed on another machine/cluster (e.g. PACE).
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir, "data"))
HP_DIR = os.path.join(DATA_DIR, "optimalHyperParams")
SWEEP_DIR = os.path.join(DATA_DIR, "hyperparams")
os.makedirs(HP_DIR, exist_ok=True)
os.makedirs(SWEEP_DIR, exist_ok=True)

if __name__=='__main__':
    model = "Quad"
    n=100 #100
    model={"model": model, "n": n}
    # op=OPTS.SS
    # output, hp=grid_search(op, model, SCS_updated_fine_grid)
    # print(output)
    # print(hp)
    # with open(f"/Users/christopherlinder/Desktop/OptimizerTesting/data/optimalHyperParams/NewQuad(n={n})_{op.name}_hp.json", 'w') as f:
    #     json.dump(hp, f)

    op=OPTS.MUON
    print(Muon_updated_fine_grid)
    output, hp=grid_search(
        op, model, Muon_updated_fine_grid,
        tsv_path=os.path.join(SWEEP_DIR, f"Quad_n={n}_{op.name}_gridsweep.tsv"),
    )
    print(output)
    print(hp)
    with open(os.path.join(HP_DIR, f"NewQuad_kappa=1_(n={n})_{op.name}_hp.json"), 'w') as f:
        json.dump(hp, f)

    # op=OPTS.SCS
    # output, hp=grid_search(op, model, SCS_updated_fine_grid)
    # print(output)
    # print(hp)
    # with open(f"/Users/christopherlinder/Desktop/OptimizerTesting/data/optimalHyperParams/NewQuad_kappa=1_(n={n})_{op.name}_hp.json", 'w') as f:
    #     json.dump(hp, f)

    # op=OPTS.AdamW
    # output, hp=grid_search(op, model, AdamW_updated_fine_grid)
    # print(output)
    # print(hp)
    # with open(f"/Users/christopherlinder/Desktop/OptimizerTesting/data/optimalHyperParams/NewQuad(n={n})_{op.name}_hp.json", 'w') as f:
    #     json.dump(hp, f)

    # op=OPTS.SGD
    # output, hp=grid_search(op, model,SGD_updated_fine_grid)
    # print(output)
    # print(hp)
    # with open(f"/Users/christopherlinder/Desktop/OptimizerTesting/data/optimalHyperParams/NewQuad(n={n})_{op.name}_hp.json", 'w') as f:
    #     json.dump(hp, f)

    op=OPTS.STIEFEL_ADAM
    output, hp=grid_search(
        op, model, stiefelAdam_updated_fine_grid,
        tsv_path=os.path.join(SWEEP_DIR, f"Quad_n={n}_{op.name}_gridsweep.tsv"),
    )
    print(output)
    print(hp)
    with open(os.path.join(HP_DIR, f"NewQuad(n={n})_{op.name}_hp.json"), 'w') as f:
        json.dump(hp, f)

    # op=OPTS.STIEFEL_SGD
    # output, hp=grid_search(op, model, stiefelSGD_updated_fine_grid)
    # print(output)
    # print(hp)
    # with open(f"/Users/christopherlinder/Desktop/OptimizerTesting/data/optimalHyperParams/NewQuad(n={n})_{op.name}_hp.json", 'w') as f:
    #     json.dump(hp, f)