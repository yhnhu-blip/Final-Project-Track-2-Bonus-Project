"""6-seed validator for Run 1 low-level policy + adaptive PD planner."""
import subprocess, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "artifacts" / "run_fastv1" / "best_checkpoint"
# Use the same adaptive PD planner (straight=2.0, corner=1.9) as our 127.53s baseline.
# We're testing whether the new low-level policy can track these commands more
# reliably / at higher speeds.
CONFIG = ROOT / "configs" / "planner_adaptive_pd.json"
SEEDS = [20260527, 42, 1000, 54321, 777777, 2025]
OUTBASE = ROOT / "track_eval" / "validate_run_fastv1"

results = {}
for seed in SEEDS:
    print(f"\n===== seed={seed} =====", flush=True)
    out_dir = OUTBASE / f"seed_{seed}"
    results_file = out_dir / "results.json"
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
        print("STDOUT:", proc.stdout[-1000:])
        print("STDERR:", proc.stderr[-1000:])

print("\n===== SUMMARY =====")
all_complete = True
for seed, r in results.items():
    fell = r.get("fall", True)
    ft = r.get("finish_time")
    ms = r.get("mean_progress_speed", r.get("mean_speed_mps", 0))
    mm = r.get("min_boundary_margin_m", 0)
    status = "✓" if (not fell and ft) else "✗ FALL/DNF"
    print(f"seed={seed:>8}: {status}  finish={str(ft):>8}s  mean_speed={float(ms):.4f}  min_margin={float(mm):.4f}m")
    if fell or not ft:
        all_complete = False

n_ok = sum(1 for r in results.values() if not r.get("fall") and r.get("finish_time"))
print(f"\n{n_ok}/{len(SEEDS)} stable")
if n_ok > 0:
    finish_times = [r["finish_time"] for r in results.values() if not r.get("fall") and r.get("finish_time")]
    print(f"Avg finish time (finishers): {sum(finish_times)/len(finish_times):.2f}s")
    if all_complete:
        print(f"Avg finish time (all 6):    {sum(finish_times)/len(SEEDS):.2f}s")
        print(f"Baseline (MLP v3):           127.53s")
        delta = sum(finish_times)/len(SEEDS) - 127.53
        print(f"Delta vs baseline:           {delta:+.2f}s")
