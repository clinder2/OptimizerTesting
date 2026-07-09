import json
from token import OP

from Base import *
from GridSearch import fine_grid, coarse_grid, grid_search

if __name__=='__main__':
    model = "Quad"
    n=100
    model={"model": model, "n": n}
    for op in [OPTS.S, OPTS.SCI, OPTS.SGD]:
        output, hp=grid_search(op, model, fine_grid)
        print(output)
        print(hp)
        with open(f"data/optimalHyperParams/Quad(n={n})_{op.name}_hp.json", 'w') as f:
            json.dump(hp, f)