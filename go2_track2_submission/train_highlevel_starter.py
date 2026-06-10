#!/usr/bin/env python3
"""Tiny black-box search scaffold for the track bonus high-level planner.

This is intentionally simple and intentionally not a solved planner. It searches
over a small JSON controller by repeatedly running `run_track_bonus.py --no-render`.
Use it to debug the evaluation loop. For a leaderboard submission, replace the
planner internals with a learned policy that maps the official 5D observation
to `[vx, vy, yaw_rate]`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from track_bonus.planner import StarterPlannerConfig


ROOT = Path(__file__).resolve().parent


SEARCH_KEYS = [
    "speed_mps",
    "max_lateral_speed_mps",
    "max_yaw_rate_radps",
    "k_heading",
    "k_lateral",
    "heading_slowdown",
]


BOUNDS = {
    "speed_mps":              (0.20, 2.20),   # raised from 0.90 — covers 1.9 m/s sweet spot
    "max_lateral_speed_mps":  (0.03, 0.22),
    "max_yaw_rate_radps":     (0.12, 0.75),
    "k_heading":              (0.20, 1.40),
    "k_lateral":              (0.02, 0.24),
    "heading_slowdown":       (0.0,  0.80),
}

# Fitness is measured in units of -seconds (higher = faster complete lap).
# Penalties keep every complete-lap solution strictly above every incomplete one.
BIG_PENALTY  = 600.0   # per unit of missing lap fraction (e.g. 50 % missing → -300 extra)
FALL_PENALTY = 300.0   # stacked on top for fall or boundary violation

# Three seeds used for every candidate evaluation — guards against lucky-seed solutions.
EVAL_SEEDS = [20260527, 42, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-planner-config", type=Path, default=ROOT / "configs" / "starter_planner.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "course_config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--eval-seconds", type=float, default=300.0)   # raised from 60 s — need full lap
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clip_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    clipped = dict(candidate)
    for key, (low, high) in BOUNDS.items():
        clipped[key] = float(np.clip(float(clipped[key]), low, high))
    clipped["min_speed_mps"] = min(float(clipped.get("min_speed_mps", 0.12)), float(clipped["speed_mps"]))
    return clipped


def _sample_candidate(center: dict[str, Any], scale: float, rng: np.random.Generator) -> dict[str, Any]:
    candidate = dict(center)
    for key in SEARCH_KEYS:
        low, high = BOUNDS[key]
        sigma = scale * (high - low)
        candidate[key] = float(center[key] + rng.normal(0.0, sigma))
    return _clip_candidate(candidate)


def _compute_fitness(metrics: dict | None, eval_seconds: float) -> float:
    """Convert rollout metrics to a scalar fitness (higher is better).

    Fitness scale (all in -seconds so evolution maximises speed):
      complete lap  →  -finish_time                              (e.g. -129 s)
      incomplete    →  -(eval_seconds + (1-completion)*BIG_PENALTY)   (< -300, always worse)
      fall/boundary →  incomplete penalty  -  FALL_PENALTY            (even worse)
      error/None    →  worst possible value
    """
    if metrics is None:
        return -(eval_seconds + BIG_PENALTY + FALL_PENALTY)

    fall       = bool(metrics.get("fall", False))
    boundary   = bool(metrics.get("boundary_violation", False))
    finish_time = metrics.get("finish_time")
    completion  = float(metrics.get("lap_completion", 0.0))

    if finish_time is not None:
        # Full lap completed — reward speed directly.
        return -float(finish_time)

    incomplete_cost = eval_seconds + (1.0 - completion) * BIG_PENALTY
    if fall or boundary:
        return -(incomplete_cost + FALL_PENALTY)
    return -incomplete_cost


def _run_eval_once(
    *,
    checkpoint_dir: Path,
    planner_path: Path,
    config: Path,
    output_dir: Path,
    eval_seconds: float,
    seed: int,
    force_cpu: bool,
) -> dict | None:
    """Run one evaluation subprocess; returns metrics dict or None on failure."""
    cmd = [
        sys.executable, "run_track_bonus.py",
        "--checkpoint-dir", str(checkpoint_dir),
        "--planner-config",  str(planner_path),
        "--config",          str(config),
        "--output-dir",      str(output_dir),
        "--duration-seconds", str(eval_seconds),
        "--seed",            str(seed),
        "--no-render",
    ]
    if force_cpu:
        cmd.append("--force-cpu")
    try:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        return payload["metrics"]
    except Exception:
        return None


def _run_eval_robust(
    *,
    checkpoint_dir: Path,
    planner_path: Path,
    config: Path,
    output_dir: Path,
    eval_seconds: float,
    force_cpu: bool,
) -> tuple[float, list[dict]]:
    """Evaluate a candidate across EVAL_SEEDS; return (worst-case fitness, per-seed log).

    Strategy:
    - If any seed produces a fall or boundary violation, the candidate is immediately
      disqualified and remaining seeds are skipped (fail-fast).
    - Otherwise fitness = worst (slowest) finish_time across all seeds — ensures the
      returned solution is robust, not a lucky single-seed fluke.
    """
    per_seed: list[dict] = []
    worst_fitness: float | None = None

    for seed in EVAL_SEEDS:
        seed_dir = output_dir / f"seed_{seed}"
        metrics  = _run_eval_once(
            checkpoint_dir=checkpoint_dir,
            planner_path=planner_path,
            config=config,
            output_dir=seed_dir,
            eval_seconds=eval_seconds,
            seed=seed,
            force_cpu=force_cpu,
        )
        fitness = _compute_fitness(metrics, eval_seconds)
        per_seed.append({"seed": seed, "fitness": round(fitness, 3), "metrics": metrics})

        if metrics is not None and (metrics.get("fall") or metrics.get("boundary_violation")):
            # Immediately disqualify — don't waste time on remaining seeds.
            return fitness, per_seed

        worst_fitness = fitness if worst_fitness is None else min(worst_fitness, fitness)

    if worst_fitness is None:
        worst_fitness = _compute_fitness(None, eval_seconds)
    return worst_fitness, per_seed


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base   = StarterPlannerConfig.load(args.base_planner_config).to_dict()
    center = _clip_candidate(base)
    best        = dict(center)
    best_score  = -1e9
    history: list[dict] = []

    for iteration in range(int(args.iterations)):
        scale = max(0.04, 0.18 * (0.72 ** iteration))
        candidates = [dict(best if best_score > -1e9 else center)]
        while len(candidates) < int(args.population):
            candidates.append(_sample_candidate(best if best_score > -1e9 else center, scale, rng))

        for candidate_idx, candidate in enumerate(candidates):
            candidate_dir = output_dir / "candidates" / f"iter_{iteration:02d}_cand_{candidate_idx:02d}"
            planner_path  = candidate_dir / "planner_config.json"
            _write_json(planner_path, candidate)

            score, per_seed = _run_eval_robust(
                checkpoint_dir=args.checkpoint_dir,
                planner_path=planner_path,
                config=args.config,
                output_dir=candidate_dir / "eval",
                eval_seconds=float(args.eval_seconds),
                force_cpu=bool(args.force_cpu),
            )

            record = {
                "iteration": iteration,
                "candidate": candidate_idx,
                "score": score,
                "planner": candidate,
                "per_seed": per_seed,
            }
            history.append(record)

            if score > best_score:
                best_score = score
                best = dict(candidate)
                _write_json(output_dir / "best_planner_config.json", best)
                _write_json(output_dir / "best_score.json", {
                    "score": best_score,
                    "finish_time_equiv": -best_score if best_score > -300 else None,
                    "iteration": iteration,
                    "candidate": candidate_idx,
                })

            seeds_ran = len(per_seed)
            print(
                f"iter={iteration} cand={candidate_idx:02d} "
                f"score={score:8.1f} best={best_score:8.1f} seeds_ran={seeds_ran}",
                flush=True,
            )

        _write_json(output_dir / "search_summary.json", {
            "best_score": best_score,
            "best_planner": best,
            "history": history,
        })

    print(json.dumps({
        "best_score": best_score,
        "best_planner_config": str(output_dir / "best_planner_config.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
