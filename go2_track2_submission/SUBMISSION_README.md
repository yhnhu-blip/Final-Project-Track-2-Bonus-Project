# Track 2 Bonus — Submission

## Result

| Metric | Value |
|--------|-------|
| Finish time | **79.18 s** |
| Lap completion | 1.0 (full lap) |
| Fall | No |
| Boundary violation | No |
| RMS lateral error | 0.134 m |
| Min boundary margin | 1.614 m |
| Seed | 42 |

## Approach (short)

Two-layer controller:
- **Low-level locomotion**: RL policy (Brax PPO) retrained to track high-speed
  commands (up to vx = 3.0 m/s). Checkpoint at `artifacts/run_fastv2/best_checkpoint/`.
- **High-level planner**: a small MLP (5 -> 32 -> 32 -> 2, ~1300 params) that
  outputs (vx_residual, yaw_residual) on top of a geometric feed-forward.
  The MLP was pre-trained (supervised) to imitate a fast rule planner, then
  fine-tuned with CEM against the official evaluator. The CEM fitness balances
  finish time, lateral error, and boundary margin, so the result is fast AND
  well-centered AND safe.

Planner code: `track_bonus/planner.py` (class `StarterTrackPlanner`,
`planner_type = "mlp_residual"`).

## How to reproduce

From the repo root:

```bash
python run_track_bonus.py \
    --checkpoint-dir artifacts/run_fastv2/best_checkpoint \
    --planner-config submission/planner_config.json \
    --output-dir track_eval/repro \
    --seed 42 --no-render
```

Expected: `finish_time` approximately 79 s, `fall = false`, `boundary_violation = false`.

The planner config points to the learned weights at `submission/best_weights.npz`
(relative path, resolved from the repo root by `track_bonus/planner.py`).

## Submission contents

| Path | What |
|------|------|
| `artifacts/run_fastv2/best_checkpoint/` | Low-level locomotion checkpoint |
| `track_bonus/planner.py` | High-level planner code (MLP residual) |
| `submission/planner_config.json` | Planner config (relative weights_path) |
| `submission/best_weights.npz` | Learned MLP weights |
| `submission/results.json` | Evaluation result (79.18 s) |
| `run_track_bonus.py`, `course_common.py`, `go2_pg_env/`, `configs/course_config.json` | Files needed to run the evaluator |
