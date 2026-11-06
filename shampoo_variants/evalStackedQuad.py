import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model import MatrixSimple
from shampoo_variants.StackedShampoo import StackedShampoo


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def make_model(seed, device):
    generator = torch.Generator().manual_seed(seed)
    left, _ = torch.linalg.qr(torch.randn(100, 100, generator=generator))
    right, _ = torch.linalg.qr(torch.randn(100, 100, generator=generator))
    target = left @ torch.diag(torch.logspace(0, -5, 100)) @ right
    model = MatrixSimple(target, seed)
    with torch.no_grad():
        if torch.linalg.det(model.W) < 0:
            model.W[:, 0] *= -1
    model.to(device)
    model.A = model.A.to(device)
    model.P = model.P.to(device)
    return model, float(torch.linalg.cond(target))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NVIDIA profiling but is not available")

    model, condition_number = make_model(args.seed, device)
    optimizer = StackedShampoo(
        [{"params": [model.W], "lr": args.lr, "beta2": args.beta2}],
        p=4,
        chol=args.variant == "cholesky",
        grafting=True,
        numIters=args.power_iters,
    )
    stats = {key: [] for key in (
        "loss", "first_order", "curvature", "predicted_decrease", "actual_decrease"
    )}
    total_step_time = 0.0
    total_fails = 0
    synchronize(device)
    start = time.perf_counter()
    for iteration in range(args.iters):
        optimizer.param_groups[0]["lr"] = args.lr * 0.5 * (
            1.0 + math.cos(math.pi * iteration / args.iters)
        )
        gradient, loss = model()
        loss.backward()
        before = model.W.detach().clone()
        synchronize(device)
        step_start = time.perf_counter()
        if args.nvtx:
            torch.cuda.nvtx.range_push("optimizer_step")
        try:
            optimizer.step()
        finally:
            if args.nvtx:
                torch.cuda.nvtx.range_pop()
        synchronize(device)
        total_step_time += time.perf_counter() - step_start
        total_fails += optimizer.fails
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            update = before - model.W
            first_order = torch.sum(gradient.detach() * update)
            curvature = torch.sum(update * (model.P @ update))
            after_loss = ((model.A @ model.W - model.A) ** 2).sum()
            stats["loss"].append(loss.item())
            stats["first_order"].append(first_order.item())
            stats["curvature"].append(curvature.item())
            stats["predicted_decrease"].append((first_order - curvature).item())
            stats["actual_decrease"].append((loss.detach() - after_loss).item())

    synchronize(device)
    result = {
        "variant": args.variant,
        "device": str(device),
        "seed": args.seed,
        "matrix_size": 100,
        "spectrum": [0, -5],
        "condition_number": condition_number,
        "iterations": args.iters,
        "learning_rate": args.lr,
        "beta2": args.beta2,
        "power_iterations": args.power_iters,
        "grafting": False,
        "elapsed_seconds": time.perf_counter() - start,
        "optimizer_seconds": total_step_time,
        "cholesky_failures": total_fails,
        "final_loss": after_loss.item(),
        **stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.variant}: time={result['elapsed_seconds']:.2f}s, "
          f"optimizer={total_step_time:.2f}s, final_loss={result['final_loss']:.6g}, "
          f"cholesky_failures={total_fails}")


def main():
    parser = argparse.ArgumentParser(description="Compare StackedShampoo on a quadratic loss")
    parser.add_argument("--variant", choices=("default", "cholesky"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=positive_int, default=2000)
    parser.add_argument("--power-iters", type=positive_int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--beta2", type=float, default=0.85)
    parser.add_argument("--nvtx", action="store_true")
    args = parser.parse_args()
    if args.nvtx and torch.device(args.device).type != "cuda":
        parser.error("--nvtx requires a CUDA device")
    if not 0 < args.beta2 <= 1:
        parser.error("--beta2 must be in (0, 1]")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()