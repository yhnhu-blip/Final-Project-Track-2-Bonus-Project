#!/usr/bin/env python3
"""Stage A: supervised pre-training of the V2 MLP planner (hidden=32).

Reads race_rollouts.npz produced by running the starter_pd rule planner, then
trains the MLP to reproduce the same commands in residual space.  This gives
CEM a warm start that already completes the lap before any search begins.

Usage (run from repo root):
    python pretrain_planner_v2.py \
        --rollout artifacts/rule_eval/race_rollouts.npz \
        --out     artifacts/v2/pretrain_weights.npz \
        --vx-lo 0.45 --vx-hi 1.00 --yaw-res-scale 0.25 --k-heading 0.65
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Must match track_bonus/planner.py
MLP_IN, MLP_HIDDEN, MLP_OUT = 5, 32, 2
TURN_RADIUS_M = 18.25
HALF_WIDTH_M  = 2.0


def param_count(in_=MLP_IN, h=MLP_HIDDEN, out=MLP_OUT):
    return in_ * h + h + h * h + h + h * out + out


def unpack(theta, in_=MLP_IN, h=MLP_HIDDEN, out=MLP_OUT):
    i = 0
    W1 = theta[i:i + in_ * h].reshape(in_, h); i += in_ * h
    b1 = theta[i:i + h];                        i += h
    W2 = theta[i:i + h * h].reshape(h, h);      i += h * h
    b2 = theta[i:i + h];                        i += h
    W3 = theta[i:i + h * out].reshape(h, out);  i += h * out
    b3 = theta[i:i + out];                      i += out
    return W1, b1, W2, b2, W3, b3


def forward_cache(theta, X):
    W1, b1, W2, b2, W3, b3 = unpack(theta)
    z1 = X @ W1 + b1; a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2; a2 = np.tanh(z2)
    z3 = a2 @ W3 + b3; a3 = np.tanh(z3)
    return (X, a1, a2, a3), (W1, b1, W2, b2, W3, b3)


def loss_and_grad(theta, X, Y):
    (X_, a1, a2, a3), (W1, b1, W2, b2, W3, b3) = forward_cache(theta, X)
    m = X.shape[0]
    diff = a3 - Y
    loss = float(np.mean(diff ** 2))
    d3 = (2.0 / m) * diff * (1 - a3 ** 2)
    gW3 = a2.T @ d3; gb3 = d3.sum(0)
    d2 = (d3 @ W3.T) * (1 - a2 ** 2)
    gW2 = a1.T @ d2; gb2 = d2.sum(0)
    d1 = (d2 @ W2.T) * (1 - a1 ** 2)
    gW1 = X_.T @ d1; gb1 = d1.sum(0)
    g = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2, gW3.ravel(), gb3])
    return loss, g


def command_to_residual_targets(track_obs, commands, vx_lo, vx_hi, yaw_res_scale, k_heading):
    """Invert the MLP planner mapping to compute supervision targets."""
    curv_norm = track_obs[:, 4]
    head_err  = track_obs[:, 3]
    curvature = curv_norm / TURN_RADIUS_M

    vx_cmd  = commands[:, 0]
    yaw_cmd = commands[:, 2]

    mid  = 0.5 * (vx_lo + vx_hi)
    span = max(0.5 * (vx_hi - vx_lo), 1e-6)
    vx_res = np.clip((vx_cmd - mid) / span, -0.999, 0.999)

    yaw_ff  = curvature * vx_cmd + k_heading * head_err
    yaw_res = np.clip((yaw_cmd - yaw_ff) / max(yaw_res_scale, 1e-6), -0.999, 0.999)

    return np.stack([vx_res, yaw_res], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", type=Path, required=True,
                    help="race_rollouts.npz from a rule-planner (starter_pd) eval")
    ap.add_argument("--out",    type=Path, required=True, help="output weights .npz")
    ap.add_argument("--vx-lo",         type=float, default=0.45)
    ap.add_argument("--vx-hi",         type=float, default=1.00)
    ap.add_argument("--yaw-res-scale", type=float, default=0.25)
    ap.add_argument("--k-heading",     type=float, default=0.65)
    ap.add_argument("--iters",         type=int,   default=5000)
    ap.add_argument("--lr",            type=float, default=0.04)
    ap.add_argument("--seed",          type=int,   default=0)
    args = ap.parse_args()

    data      = np.load(args.rollout)
    track_obs = np.asarray(data["track_observation"], dtype=np.float64)
    commands  = np.asarray(data["command"],           dtype=np.float64)
    T = min(len(track_obs), len(commands))
    track_obs, commands = track_obs[:T], commands[:T]

    moving = np.linalg.norm(commands, axis=1) > 1e-3
    track_obs, commands = track_obs[moving], commands[moving]
    print(f"training samples: {len(track_obs)}")
    print(f"MLP arch: {MLP_IN} -> {MLP_HIDDEN} -> {MLP_HIDDEN} -> {MLP_OUT}  "
          f"({param_count()} params)")

    X = track_obs
    Y = command_to_residual_targets(
        track_obs, commands,
        args.vx_lo, args.vx_hi, args.yaw_res_scale, args.k_heading,
    )

    rng   = np.random.default_rng(args.seed)
    theta = rng.normal(0, 0.1, param_count())  # smaller init for hidden=32

    for it in range(args.iters):
        loss, g = loss_and_grad(theta, X, Y)
        theta -= args.lr * g
        if it % 1000 == 0:
            print(f"iter {it:5d}  mse={loss:.6f}")
    print(f"final mse={loss:.6f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, theta=theta.astype(np.float64))
    print(f"wrote {args.out}  ({theta.size} params)")


if __name__ == "__main__":
    main()
