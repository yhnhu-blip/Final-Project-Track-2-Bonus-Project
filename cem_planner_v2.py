#!/usr/bin/env python3
"""Stage B: CEM fine-tuning of the V2 MLP planner (hidden=32).

V2 key improvement: multi-objective fitness that balances finish time AND
lateral error, so the optimizer can't sacrifice line-keeping for raw speed.

Fitness (completed lap):
    1000 - finish_time
    - 30  * rms_lateral_error        (30 pts per meter of RMS cross-track error)
    - 100 * max(0, 0.5 - boundary_margin)  (heavy penalty for hugging the wall)

Fitness (incomplete):
    valid_distance_m - 50*fall_or_violation - 5*rms_lateral_error

This mirrors the leaderboard priority (completers beat non-completers) while
steering the optimizer toward both fast AND centered laps.

Usage (run from repo root):
    python cem_planner_v2.py \
        --checkpoint-dir lowlevel_v7 \
        --base-config    configs/planner_v2_base.json \
        --init-weights   artifacts/v2/pretrain_weights.npz \
        --config         configs/colab_runtime_config.json \
        --output-dir     artifacts/v2/cem \
        --generations 10 --population 15 --elite 4 \
        --sigma0 0.12 --sigma-decay 0.85 \
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
MLP_IN, MLP_HIDDEN, MLP_OUT = 5, 32, 2  # must match track_bonus/planner.py


def param_count(in_=MLP_IN, h=MLP_HIDDEN, out=MLP_OUT):
    return in_ * h + h + h * h + h + h * out + out


def fitness_from_metrics(m: dict) -> float:
    """Multi-objective fitness: balance speed, line-keeping, and boundary safety."""
    lat_err = float(m.get("rms_lateral_error", 9.9))
    margin  = float(m.get("min_boundary_margin_m", 0.0))

    completed = (
        float(m.get("lap_completion", 0.0)) >= 0.999
        and not bool(m.get("boundary_violation", False))
        and not bool(m.get("fall", False))
    )

    if completed:
        ft = float(m.get("finish_time") or m.get("alive_time", 9999.0))
        speed_score    = 1000.0 - ft
        lat_penalty    = 30.0 * lat_err
        margin_penalty = 100.0 * max(0.0, 0.5 - margin)
        return speed_score - lat_penalty - margin_penalty

    penalty = 50.0 if (m.get("boundary_violation") or m.get("fall")) else 0.0
    return float(m.get("valid_distance_m", 0.0)) - penalty - 5.0 * lat_err


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def eval_candidate(theta, *, base_cfg, checkpoint_dir, config, cand_dir,
                   eval_seconds) -> tuple[float, dict]:
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
        "--config",         str(config),
        "--output-dir",     str(eval_dir),
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
    ap.add_argument("--base-config",    type=Path, required=True)
    ap.add_argument("--init-weights",   type=Path, required=True)
    ap.add_argument("--config",         type=Path,
                    default=Path(__file__).resolve().parent / "configs" / "course_config.json",
                    help="course config JSON (defaults to configs/course_config.json)")
    ap.add_argument("--output-dir",     type=Path, required=True)
    ap.add_argument("--generations",    type=int,   default=10)
    ap.add_argument("--population",     type=int,   default=15)
    ap.add_argument("--elite",          type=int,   default=4)
    ap.add_argument("--sigma0",         type=float, default=0.12)
    ap.add_argument("--sigma-decay",    type=float, default=0.85)
    ap.add_argument("--eval-seconds",   type=float, default=210.0)
    ap.add_argument("--seed",           type=int,   default=0)
    args = ap.parse_args()

    base_cfg = json.loads(args.base_config.read_text(encoding="utf-8"))
    mean = np.asarray(np.load(args.init_weights)["theta"], dtype=np.float64).ravel()
    n = param_count()
    if mean.size != n:
        raise ValueError(f"init weights have {mean.size} params, expected {n}")

    rng = np.random.default_rng(args.seed)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    best_theta = mean.copy()
    best_fit, best_metrics = eval_candidate(
        mean, base_cfg=base_cfg, checkpoint_dir=args.checkpoint_dir,
        config=args.config, cand_dir=out / "init", eval_seconds=args.eval_seconds)
    print(f"[init] fitness={best_fit:.2f}  {_short(best_metrics)}", flush=True)
    _save_best(out, best_theta, best_fit, base_cfg)

    sigma   = args.sigma0
    history = [{"gen": -1, "fitness": best_fit, "metrics": best_metrics}]

    for gen in range(args.generations):
        pop     = [mean + sigma * rng.standard_normal(n) for _ in range(args.population)]
        results = []
        for ci, theta in enumerate(pop):
            fit, met = eval_candidate(
                theta, base_cfg=base_cfg, checkpoint_dir=args.checkpoint_dir,
                config=args.config,
                cand_dir=out / f"gen{gen:02d}" / f"cand{ci:02d}",
                eval_seconds=args.eval_seconds)
            results.append((fit, theta, met))
            print(f"[gen {gen} cand {ci:2d}] fit={fit:8.2f}  {_short(met)}", flush=True)

        results.sort(key=lambda r: r[0], reverse=True)
        elites = results[:args.elite]
        mean   = np.mean([e[1] for e in elites], axis=0)
        sigma *= args.sigma_decay

        if elites[0][0] > best_fit:
            best_fit, best_theta, best_metrics = elites[0][0], elites[0][1], elites[0][2]
            _save_best(out, best_theta, best_fit, base_cfg)
            print(f"  *** new best! fitness={best_fit:.2f} ***", flush=True)

        history.append({"gen": gen, "best_fitness_this_gen": elites[0][0],
                        "overall_best": best_fit, "best_metrics": best_metrics})
        write_json(out / "cem_history.json", history)
        _print_gen_summary(gen, best_fit, best_metrics)

    print(json.dumps({
        "best_fitness":        best_fit,
        "best_weights":        str(out / "best_weights.npz"),
        "best_planner_config": str(out / "best_planner_config.json"),
        "best_metrics":        best_metrics,
    }, indent=2))


def _save_best(out: Path, theta: np.ndarray, fit: float, base_cfg: dict) -> None:
    np.savez(out / "best_weights.npz", theta=theta)
    write_json(out / "best_planner_config.json",
               {**base_cfg, "planner_type": "mlp_residual",
                "weights_path": str((out / "best_weights.npz").resolve())})


def _print_gen_summary(gen: int, best_fit: float, m: dict) -> None:
    ft     = m.get("finish_time")
    lat    = m.get("rms_lateral_error")
    margin = m.get("min_boundary_margin_m")
    print(f"== gen {gen} done. overall best fitness={best_fit:.2f} | "
          f"time={ft}s | lat_rms={lat:.3f}m | margin={margin:.3f}m ==", flush=True)


def _short(m: dict) -> str:
    if "error" in m:
        return f"ERROR {m['error'][:60]}"
    lat    = m.get("rms_lateral_error", float("nan"))
    margin = m.get("min_boundary_margin_m", float("nan"))
    return (
        f"lap={m.get('lap_completion', 0):.3f} "
        f"time={m.get('finish_time')} "
        f"dist={m.get('valid_distance_m', 0):.1f}m "
        f"lat={lat:.3f}m "
        f"margin={margin:.3f}m "
        f"bnd={m.get('boundary_violation')} "
        f"fall={m.get('fall')}"
    )


if __name__ == "__main__":
    main()
