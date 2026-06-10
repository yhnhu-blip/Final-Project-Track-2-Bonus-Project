#!/usr/bin/env python3
"""Collect BC training data from adaptive PD teacher for MLP v3.

Uses planner_adaptive_pd.json (straight=2.0, corner=1.9) as teacher.
Collects 10 diverse seeds and merges into bc_data_adaptive.npz.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "lowlevel_v7"
TEACHER_CONFIG = ROOT / "configs" / "planner_adaptive_pd.json"
OUT_DIR = ROOT / "bc_collect" / "adaptive_v3"
OUTFILE = ROOT / "bc_data_adaptive.npz"

COLLECT_SEEDS = [20260527, 42, 1000, 54321, 2025, 12345, 9999, 88888, 31415, 555]


def collect_seed(seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    seed_dir = OUT_DIR / f"seed_{seed}"
    rollout_file = seed_dir / "race_rollouts.npz"

    if rollout_file.exists():
        print(f"  [seed={seed}] already collected, loading ...", flush=True)
    else:
        print(f"  [seed={seed}] running ...", flush=True)
        cmd = [
            sys.executable, "run_track_bonus.py",
            "--checkpoint-dir", str(CHECKPOINT_DIR),
            "--planner-config", str(TEACHER_CONFIG),
            "--output-dir", str(seed_dir),
            "--duration-seconds", "300",
            "--seed", str(seed),
            "--no-render",
        ]
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=420)
        if not rollout_file.exists():
            print(f"  [seed={seed}] FAILED: {result.stderr[-300:]}", flush=True)
            return None

    data = np.load(rollout_file)
    obs = data["track_observation"].astype(np.float32)   # (T, 5)
    cmd = data["command"].astype(np.float32)             # (T, 3)
    # Filter out standstill frames (vx ≈ 0) to avoid diluting adaptive signal
    moving = obs[:, 0] > 0.05   # lap_fraction > 0 means robot is moving; use vx proxy
    # Actually filter by command vx > 0 to skip the stand phase
    moving_mask = cmd[:, 0] > 0.05
    obs = obs[moving_mask]
    cmd = cmd[moving_mask]
    print(f"  [seed={seed}] samples={len(obs)} (after filtering standstill)", flush=True)
    return obs, cmd


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_obs, all_cmd = [], []

    for seed in COLLECT_SEEDS:
        result = collect_seed(seed)
        if result is not None:
            obs, cmd = result
            all_obs.append(obs)
            all_cmd.append(cmd)

    if not all_obs:
        print("No data collected!")
        return

    obs_all = np.concatenate(all_obs, axis=0)
    cmd_all = np.concatenate(all_cmd, axis=0)
    print(f"\nTotal samples: {len(obs_all):,}  obs={obs_all.shape}  cmd={cmd_all.shape}")

    # Show command statistics to verify adaptive behavior is captured
    print("\nCommand statistics (adaptive teacher):")
    names = ["vx", "vy", "yaw_rate"]
    for i, name in enumerate(names):
        print(f"  {name}: min={cmd_all[:, i].min():.4f}  max={cmd_all[:, i].max():.4f}"
              f"  mean={cmd_all[:, i].mean():.4f}  std={cmd_all[:, i].std():.4f}")

    # Verify vx distribution shows two modes (straight~2.0, corner~1.9)
    vx = cmd_all[:, 0]
    print(f"\nvx distribution:")
    print(f"  vx > 1.95: {(vx > 1.95).mean()*100:.1f}%  (straight samples)")
    print(f"  1.8 < vx <= 1.95: {((vx > 1.8) & (vx <= 1.95)).mean()*100:.1f}%  (corner samples)")
    print(f"  vx <= 1.8: {(vx <= 1.8).mean()*100:.1f}%  (slow/transition samples)")

    np.savez(OUTFILE, obs=obs_all, cmd=cmd_all)
    print(f"\nSaved → {OUTFILE}  ({OUTFILE.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
