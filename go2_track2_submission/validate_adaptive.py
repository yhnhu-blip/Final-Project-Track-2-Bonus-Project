"""Quick 6-seed validator for adaptive PD planner."""
import subprocess, json, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "lowlevel_v7"
CONFIG = ROOT / "configs" / "planner_mlp_v3.json"
SEEDS = [20260527, 42, 1000, 54321, 777777, 2025]
OUTBASE = ROOT / "track_eval" / "validate_mlp_v3_clipped"

results = {}
for seed in SEEDS:
    print(f"\n===== seed={seed} =====", flush=True)
    out_dir = OUTBASE / f"seed_{seed}"
    results_file = out_dir / "results.json"
    # Skip if already evaluated (avoids re-running on partial failure restarts)
    if results_file.exists():
        print("  [cached]", flush=True)
        payload = json.loads(results_file.read_text())
        results[seed] = payload.get("metrics", payload)
        print(json.dumps(results[seed], indent=4))
        continue
    cmd = [
        sys.executable, "run_track_bonus.py",
        "--checkpoint-dir", str(CHECKPOINT_DIR),
        "--planner-config", str(CONFIG),
        "--output-dir", str(out_dir),
        "--duration-seconds", "300",
        "--seed", str(seed),
        "--no-render",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT))
    if results_file.exists():
        payload = json.loads(results_file.read_text())
        r = payload.get("metrics", payload)
        results[seed] = r
        print(json.dumps(r, indent=4))
    else:
        print("FAILED - no results.json")
        print("STDOUT:", proc.stdout[-500:])
        print("STDERR:", proc.stderr[-500:])

print("\n===== SUMMARY =====")
all_complete = True
for seed, r in results.items():
    fell = r.get("fall", True)
    ft = r.get("finish_time")
    ms = r.get("mean_progress_speed", r.get("mean_speed_mps", 0))
    mm = r.get("min_boundary_margin_m", 0)
    status = "✓" if (not fell and ft) else "✗ FALL"
    print(f"seed={seed:>8}: {status}  finish={str(ft):>8}s  mean_speed={float(ms):.4f}  min_margin={float(mm):.4f}m")
    if fell or not ft:
        all_complete = False

n_ok = sum(1 for r in results.values() if not r.get("fall") and r.get("finish_time"))
print(f"\n{n_ok}/{len(SEEDS)} stable")
if all_complete:
    avg_finish = sum(r["finish_time"] for r in results.values()) / len(SEEDS)
    print(f"Average finish time: {avg_finish:.2f}s")
