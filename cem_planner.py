#!/usr/bin/env python3
"""Stage B: CEM fine-tuning of the MLP planner against the real evaluator.

We perturb the (pre-trained) MLP weight vector, run the OFFICIAL evaluator
(run_track_bonus.py --no-render) for each candidate, and use a ranking-oriented
fitness:

    completed lap   -> 1000 - finish_time         (so lower time = higher fitness)
    not completed   -> valid_distance_m - penalty (always below any completer)

This mirrors the leaderboard's lexicographic order: any lap-completing planner
beats any non-completing one; among completers, faster wins.

CEM keeps the top-k elites each generation and resamples around their mean. The
evaluator is deterministic (noise/pert disabled at reset) so each candidate is
evaluated once.

Usage:
    python cem_planner.py \
        --checkpoint-dir lowlevel_v7 \
        --base-config configs/planner_mlp.json \
        --init-weights configs/planner_weights.npz \
        --config configs/colab_runtime_config.json \
        --output-dir artifacts/cem \
        --generations 6 --population 10 --elite 3 \
        --eval-seconds 210
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
MLP_IN, MLP_HIDDEN, MLP_OUT = 5, 16, 2


def param_count(in_=MLP_IN, h=MLP_HIDDEN, out=MLP_OUT):
    return in_ * h + h + h * h + h + h * out + out


def fitness_from_metrics(m: dict) -> float:
    completed = (
        float(m.get("lap_completion", 0.0)) >= 0.999
        and not bool(m.get("boundary_violation", False))
        and not bool(m.get("fall", False))
    )
    if completed:
        ft = m.get("finish_time")
        if ft is None:
            ft = m.get("alive_time", 9999.0)
        return 1000.0 - float(ft)
    penalty = 50.0 if (m.get("boundary_violation") or m.get("fall")) else 0.0
    return float(m.get("valid_distance_m", 0.0)) - penalty


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def eval_candidate(theta, *, base_cfg, checkpoint_dir, config, cand_dir,
                   eval_seconds) -> float:
    """Write theta weights + a planner config, run the evaluator, return fitness."""
    weights_path = cand_dir / "weights.npz"
    cand_dir.mkdir(parents=True, exist_ok=True)
    np.savez(weights_path, theta=np.asarray(theta, dtype=np.float64))

    cfg = dict(base_cfg)
    cfg["planner_type"] = "mlp_residual"
    cfg["weights_path"] = str(weights_path.resolve())
    planner_path = cand_dir / "planner_config.json"
    write_json(planner_path, cfg)

    eval_dir = cand_dir / "eval"
    cmd = [
        sys.executable, "run_track_bonus.py",
        "--checkpoint-dir", str(checkpoint_dir),
        "--planner-config", str(planner_path),
        "--config", str(config),
        "--output-dir", str(eval_dir),
        "--duration-seconds", str(eval_seconds),
        "--no-render",
    ]
    try:
        subprocess.run(cmd, cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        res = json.loads((eval_dir / "results.json").read_text(encoding="utf-8"))
        return fitness_from_metrics(res["metrics"]), res["metrics"]
    except Exception as e:  # noqa: BLE001
        return -1e6, {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--base-config", type=Path, required=True,
                    help="planner JSON with vx_lo/vx_hi/yaw_res_scale/etc.")
    ap.add_argument("--init-weights", type=Path, required=True,
                    help="pretrained weights .npz (theta)")
    ap.add_argument("--config", type=Path, required=True, help="colab_runtime_config.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--sigma0", type=float, default=0.15)
    ap.add_argument("--sigma-decay", type=float, default=0.8)
    ap.add_argument("--eval-seconds", type=float, default=210.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_cfg = json.loads(args.base_config.read_text(encoding="utf-8"))
    mean = np.asarray(np.load(args.init_weights)["theta"], dtype=np.float64).ravel()
    n = param_count()
    if mean.size != n:
        raise ValueError(f"init weights have {mean.size} params, expected {n}")

    rng = np.random.default_rng(args.seed)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # evaluate the pretrained init first (generation -1)
    best_theta = mean.copy()
    best_fit, best_metrics = eval_candidate(
        mean, base_cfg=base_cfg, checkpoint_dir=args.checkpoint_dir,
        config=args.config, cand_dir=out / "init", eval_seconds=args.eval_seconds)
    print(f"[init] fitness={best_fit:.2f}  metrics={_short(best_metrics)}", flush=True)
    np.savez(out / "best_weights.npz", theta=best_theta)
    write_json(out / "best_planner_config.json",
               {**base_cfg, "planner_type": "mlp_residual",
                "weights_path": str((out / "best_weights.npz").resolve())})

    sigma = args.sigma0
    history = [{"gen": -1, "fitness": best_fit, "metrics": best_metrics}]

    for gen in range(args.generations):
        pop = [mean + sigma * rng.normal(0, 1, n) for _ in range(args.population)]
        results = []
        for ci, theta in enumerate(pop):
            fit, met = eval_candidate(
                theta, base_cfg=base_cfg, checkpoint_dir=args.checkpoint_dir,
                config=args.config, cand_dir=out / f"gen{gen:02d}" / f"cand{ci:02d}",
                eval_seconds=args.eval_seconds)
            results.append((fit, theta, met))
            print(f"[gen {gen} cand {ci}] fitness={fit:.2f}  {_short(met)}", flush=True)

        results.sort(key=lambda r: r[0], reverse=True)
        elites = results[:args.elite]
        mean = np.mean([e[1] for e in elites], axis=0)
        sigma *= args.sigma_decay

        if elites[0][0] > best_fit:
            best_fit, best_theta, best_metrics = elites[0][0], elites[0][1], elites[0][2]
            np.savez(out / "best_weights.npz", theta=best_theta)
            write_json(out / "best_planner_config.json",
                       {**base_cfg, "planner_type": "mlp_residual",
                        "weights_path": str((out / "best_weights.npz").resolve())})

        history.append({"gen": gen, "best_fitness_this_gen": elites[0][0],
                        "overall_best": best_fit, "best_metrics": best_metrics})
        write_json(out / "cem_history.json", history)
        print(f"== gen {gen} done. overall best fitness={best_fit:.2f} "
              f"(finish_time={1000 - best_fit:.1f}s if completed) ==", flush=True)

    print(json.dumps({"best_fitness": best_fit,
                      "best_weights": str(out / 'best_weights.npz'),
                      "best_planner_config": str(out / 'best_planner_config.json'),
                      "best_metrics": best_metrics}, indent=2))


def _short(m: dict) -> str:
    if "error" in m:
        return f"ERROR {m['error'][:60]}"
    return (f"lap={m.get('lap_completion', 0):.3f} "
            f"finish={m.get('finish_time')} "
            f"dist={m.get('valid_distance_m', 0):.1f} "
            f"bnd={m.get('boundary_violation')} fall={m.get('fall')}")


if __name__ == "__main__":
    main()
