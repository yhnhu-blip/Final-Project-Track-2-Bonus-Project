#!/usr/bin/env python3
"""Behavior-cloning training for the MLP high-level planner.

Architecture : 5 -> 64 -> 64 -> 3  (ReLU hidden, linear output)
Loss         : MSE vs PD-planner demonstrations
Optimizer    : Adam
Output       : planner_weights.npz  (w0/b0 .. w2/b2)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
    _BACKEND = "jax"
except ImportError:
    _BACKEND = "numpy"


# ---------------------------------------------------------------------------
# JAX path
# ---------------------------------------------------------------------------

def _train_jax(obs: np.ndarray, cmd: np.ndarray, steps: int, lr: float,
               batch_size: int, seed: int) -> dict:
    key = jax.random.PRNGKey(seed)

    def _init(key, sizes):
        params = []
        for i in range(len(sizes) - 1):
            key, k1 = jax.random.split(key)
            fan_in = sizes[i]
            w = jax.random.normal(k1, (fan_in, sizes[i + 1])) * np.sqrt(2.0 / fan_in)
            b = jnp.zeros(sizes[i + 1])
            params.append((w, b))
        return params

    def forward(params, x):
        for i, (w, b) in enumerate(params):
            x = x @ w + b
            if i < len(params) - 1:
                x = jax.nn.relu(x)
        return x

    @jax.jit
    def loss_fn(params, x, y):
        return jnp.mean((forward(params, x) - y) ** 2)

    @jax.jit
    def step_fn(params, opt_state, x, y):
        val, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state_new = opt.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, val

    params = _init(key, [5, 64, 64, 3])
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    obs_j  = jnp.asarray(obs)
    cmd_j  = jnp.asarray(cmd)
    n      = len(obs)
    rng    = np.random.default_rng(seed)
    losses = []

    for step in range(1, steps + 1):
        idx    = rng.choice(n, size=min(batch_size, n), replace=False)
        params, opt_state, val = step_fn(params, opt_state, obs_j[idx], cmd_j[idx])
        if step % max(1, steps // 20) == 0 or step == steps:
            losses.append((step, float(val)))
            print(f"  step {step:5d}/{steps}  loss={float(val):.6f}", flush=True)

    weights = {}
    for i, (w, b) in enumerate(params):
        weights[f"w{i}"] = np.asarray(w)
        weights[f"b{i}"] = np.asarray(b)
    return weights, losses, lambda x: np.asarray(forward(params, jnp.asarray(x)))


# ---------------------------------------------------------------------------
# Pure-numpy fallback (manual Adam)
# ---------------------------------------------------------------------------

def _train_numpy(obs: np.ndarray, cmd: np.ndarray, steps: int, lr: float,
                 batch_size: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    def _init(sizes):
        params = []
        for i in range(len(sizes) - 1):
            w = rng.standard_normal((sizes[i], sizes[i + 1])) * np.sqrt(2.0 / sizes[i])
            b = np.zeros(sizes[i + 1])
            params.append([w.astype(np.float32), b.astype(np.float32)])
        return params

    def relu(x):   return np.maximum(0, x)
    def drelu(x):  return (x > 0).astype(np.float32)

    def forward_full(params, x):
        acts = [x]
        for i, (w, b) in enumerate(params):
            z = acts[-1] @ w + b
            acts.append(relu(z) if i < len(params) - 1 else z)
        return acts

    def backward(params, acts, y):
        delta = 2.0 * (acts[-1] - y) / len(y)
        grads = []
        for i in range(len(params) - 1, -1, -1):
            gw = acts[i].T @ delta
            gb = delta.sum(axis=0)
            grads.insert(0, (gw, gb))
            if i > 0:
                delta = (delta @ params[i][0].T) * drelu(acts[i])
        return grads

    params = _init([5, 64, 64, 3])
    # Adam state
    m  = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
    v  = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = len(obs)
    losses = []

    for step in range(1, steps + 1):
        idx  = rng.choice(n, size=min(batch_size, n), replace=False)
        xb, yb = obs[idx], cmd[idx]
        acts = forward_full(params, xb)
        loss = float(np.mean((acts[-1] - yb) ** 2))
        grads = backward(params, acts, yb)
        t = step
        for i, ((gw, gb), (mw, mb), (vw, vb)) in enumerate(zip(grads, m, v)):
            mw = b1 * mw + (1 - b1) * gw;  mb = b1 * mb + (1 - b1) * gb
            vw = b2 * vw + (1 - b2) * gw**2; vb = b2 * vb + (1 - b2) * gb**2
            mw_h = mw / (1 - b1**t);  mb_h = mb / (1 - b1**t)
            vw_h = vw / (1 - b2**t);  vb_h = vb / (1 - b2**t)
            params[i][0] -= lr * mw_h / (np.sqrt(vw_h) + eps)
            params[i][1] -= lr * mb_h / (np.sqrt(vb_h) + eps)
            m[i] = (mw, mb);  v[i] = (vw, vb)
        if step % max(1, steps // 20) == 0 or step == steps:
            losses.append((step, loss))
            print(f"  step {step:5d}/{steps}  loss={loss:.6f}", flush=True)

    weights = {}
    for i, (w, b) in enumerate(params):
        weights[f"w{i}"] = w.astype(np.float32)
        weights[f"b{i}"] = b.astype(np.float32)

    def predict(x):
        out = x.astype(np.float32)
        for i, (w, b) in enumerate(params):
            out = out @ w + b
            if i < len(params) - 1:
                out = np.maximum(0, out)
        return out

    return weights, losses, predict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",       type=Path, default=Path("bc_data.npz"))
    parser.add_argument("--output",     type=Path, default=Path("planner_weights.npz"))
    parser.add_argument("--steps",      type=int,  default=5000)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,  default=256)
    parser.add_argument("--seed",       type=int,  default=0)
    args = parser.parse_args()

    data = np.load(args.data)
    obs  = data["obs"].astype(np.float32)
    cmd  = data["cmd"].astype(np.float32)
    print(f"Loaded: obs={obs.shape}, cmd={cmd.shape}  backend={_BACKEND}")

    print(f"\nTraining MLP [5→64→64→3] for {args.steps} steps (lr={args.lr}, batch={args.batch_size}) ...")
    if _BACKEND == "jax":
        weights, losses, predict = _train_jax(obs, cmd, args.steps, args.lr, args.batch_size, args.seed)
    else:
        weights, losses, predict = _train_numpy(obs, cmd, args.steps, args.lr, args.batch_size, args.seed)

    # --- per-output error on full training set ---
    pred = predict(obs)
    names = ["vx", "vy", "yaw_rate"]
    print("\n=== Per-output error on training set ===")
    for i, name in enumerate(names):
        mae  = float(np.mean(np.abs(pred[:, i] - cmd[:, i])))
        rmse = float(np.sqrt(np.mean((pred[:, i] - cmd[:, i]) ** 2)))
        print(f"  {name:10s}: MAE={mae:.6f}  RMSE={rmse:.6f}  "
              f"label_range=[{cmd[:,i].min():.4f}, {cmd[:,i].max():.4f}]")

    # --- save weights ---
    np.savez(args.output, **weights)
    print(f"\n=== Saved weights → {args.output} ===")
    for key, arr in sorted(weights.items()):
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()
