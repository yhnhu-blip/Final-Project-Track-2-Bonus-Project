"""6-seed validator for Run 2 checkpoint vs Run 1 baseline (107.05s, min_margin 1.04m)."""
import subprocess, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "artifacts" / "run_fastv2" / "best_checkpoint"
CONFIG = ROOT / "configs" / "planner_fastv1_250.json"  # same 2.5/2.1 as Run 1 best
SEEDS = [20260527, 42, 1000, 54321, 777777, 2025]
OUTBASE = ROOT / "track_eval" / "validate_run_fastv2"

RUN1_AVG = 107.05
RUN1_MIN_MARGIN = 1.04

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
        print("STDERR:", proc.stderr[-800:])

print("\n===== RUN 2 vs RUN 1 COMPARISON =====")
n_ok = sum(1 for r in results.values() if not r.get("fall") and r.get("finish_time"))
finish_times = [r["finish_time"] for r in results.values() if not r.get("fall") and r.get("finish_time")]
margins = [r.get("min_boundary_margin_m", 0) for r in results.values() if not r.get("fall") and r.get("finish_time")]
speeds = [r.get("mean_progress_speed", r.get("mean_speed_mps", 0)) for r in results.values() if not r.get("fall") and r.get("finish_time")]

for seed, r in results.items():
    fell = r.get("fall", True)
    ft = r.get("finish_time")
    ms = r.get("mean_progress_speed", r.get("mean_speed_mps", 0))
    mm = r.get("min_boundary_margin_m", 0)
    status = "✓" if (not fell and ft) else "✗ FALL/DNF"
    print(f"  seed={seed:>8}: {status}  finish={str(ft):>8}s  speed={float(ms):.4f}  min_margin={float(mm):.4f}m")

print(f"\n{'='*50}")
print(f"  Run 2: {n_ok}/6 stable")
if finish_times:
    avg2 = sum(finish_times) / len(finish_times)
    min_margin2 = min(margins) if margins else 0
    avg_speed2 = sum(speeds) / len(speeds) if speeds else 0
    print(f"  Run 2 avg finish:   {avg2:.2f}s  (Run 1: {RUN1_AVG:.2f}s)")
    print(f"  Run 2 min_margin:   {min_margin2:.4f}m  (Run 1: {RUN1_MIN_MARGIN:.2f}m)")
    print(f"  Run 2 avg speed:    {avg_speed2:.4f} m/s")
    print()
    faster = n_ok == 6 and avg2 < RUN1_AVG
    safer_margin = min_margin2 >= RUN1_MIN_MARGIN
    if faster and safer_margin:
        print(f"  VERDICT: ✓ RUN 2 WINS — faster ({avg2:.2f}s < {RUN1_AVG:.2f}s) AND margin OK ({min_margin2:.3f}m >= {RUN1_MIN_MARGIN:.2f}m)")
        print(f"  → USE Run 2 as new best.")
    elif faster and not safer_margin:
        print(f"  VERDICT: ⚠ RUN 2 FASTER but MARGIN DROPPED ({min_margin2:.3f}m < {RUN1_MIN_MARGIN:.2f}m)")
        print(f"  → Run 2 approaches track boundary limit — evaluate risk carefully.")
    else:
        print(f"  VERDICT: ✗ RUN 2 NOT BETTER — keep Run 1 (fastv1, {RUN1_AVG:.2f}s) as current best.")
else:
    print(f"  VERDICT: ✗ RUN 2 FAILED — keep Run 1 as current best.")
