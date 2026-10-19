import sys
import os

ShampooExperiment_PATH = os.environ.get(
    "SHAMPOO_PATH", os.path.expanduser("~/Desktop/OptimizerTesting/ShampooExperiment")
)
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from models.model import MatrixSimple
from common.TrainingScripts import make_target_param
from common.Base import make_optimizer, OPTS
from stiefel_geodesic.StiefelNormal import *
import numpy as np
import matplotlib.pyplot as plt
import time

def QuadExperiment(optimizer_names, spectrum, num_trials, hyperparam_dicts, n=100):
    if not isinstance(num_trials, int) or isinstance(num_trials, bool) or num_trials < 1:
        raise ValueError("num_trials must be a positive integer")

    optimizer_aliases = {
        "SG": StiefelGeod,
        "SGM": StiefelGeodMomentum,
        "SGU": StiefelGeodUnconstrained,
    }
    for optimizer_name, hyperparam_dict in zip(optimizer_names, hyperparam_dicts):
        if isinstance(optimizer_name, OPTS):
            optimizer_type = optimizer_name
            optimizer_class = None
            name = optimizer_name.name
        elif isinstance(optimizer_name, str):
            name = optimizer_name.upper()
            optimizer_class = optimizer_aliases.get(name)
            if optimizer_class is None:
                optimizer_key = "STIEFEL_ADAM" if name == "SA" else name
                try:
                    optimizer_type = OPTS[optimizer_key]
                except KeyError as error:
                    raise ValueError(f"Unknown optimizer name: {optimizer_name}") from error
        else:
            raise TypeError("optimizer_name must be an OPTS value or optimizer name string")

        base_lr = hyperparam_dict.get("lr", hyperparam_dict.get("learning_rate"))
        if base_lr is None:
            raise ValueError("hyperparam_dict must include 'lr' or 'learning_rate'")
        iters = hyperparam_dict.get("max_iters", 200)
        if not isinstance(iters, int) or isinstance(iters, bool) or iters < 1:
            raise ValueError("hyperparam_dict['max_iters'] must be a positive integer")

        def get_lr(iteration):
            return base_lr * np.cos(iteration / iters * np.pi / 2)

        trial_losses = []
        for rand_seed in range(num_trials):
            target = make_target_param(n, rand_seed, spectrum)
            model = MatrixSimple(target, rand_seed)
            model.W.data = torch.linalg.qr(model.W.data)[0]
            if torch.linalg.det(model.W.data) < 0:
                model.W.data[:, 0] *= -1
            params = [p for p in model.parameters()]

            trial_hyperparams = dict(hyperparam_dict)
            if optimizer_class is not None:
                optimizer = optimizer_class(params, trial_hyperparams)
            else:
                optimizer = make_optimizer(optimizer_type, params, trial_hyperparams)

            losses = []
            start_time = time.time()
            for iteration in range(iters):
                lr = get_lr(iteration)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                _, loss = model()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            trial_losses.append(losses)
            print(
                f"{name} seed {rand_seed}: time={time.time() - start_time:.2f}s, "
                f"final_loss={losses[-1]:.6g}"
            )

        log_losses = np.log(np.asarray(trial_losses))
        mean_loss = np.mean(log_losses, axis=0)
        std_loss = np.std(log_losses, axis=0)
        iterations = np.arange(iters)

        plt.plot(iterations, mean_loss, label=name)
        plt.fill_between(
            iterations,
            mean_loss - std_loss,
            mean_loss + std_loss,
            alpha=0.3,
        )
    plt.xlabel("Iteration")
    plt.ylabel("Log loss")
    #plt.title(f"{name} on quadratic loss ({num_trials} trials)")
    plt.legend()
    plt.show()

    return trial_losses

QuadExperiment(['SG', 'STIEFEL_ADAM'], [0,0], 2, [{'lr': .2}, {'lr': .5}])