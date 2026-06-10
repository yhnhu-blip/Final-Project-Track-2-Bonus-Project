"""Robustness check for Run2 + 3.0/2.2: 12 seeds (6 cached + 6 new).
Goal: confirm 104s is reliable, not lucky. Flag any margin < 0.6m.
"""
import subprocess, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "artifacts" / "run_fastv2" / "best_checkpoint"
CONFIG     = ROOT / "configs" / "planner_r2_300_220.json"
OUTBASE    = ROOT / "track_eval" / "robust_300_220"

# 6 previously tested + 6 brand-new uncorrelated seeds
SEEDS = [
    20260527, 42, 1000, 54321, 777777, 2025,   # cached from sweep
    99999, 12345, 314159, 9999999, 88888, 31337, # new
]
PREV_CACHE = ROOT / "track_eval" / "r2_speed_sweep" / "planner_r2_300_220"

MARGIN_HARD_LIMIT = 0.6

results = {}
for seed in SEEDS:
    print(f"\n===== seed={seed} =====", flush=True)
    out_dir = OUTBASE / f"seed_{seed}"
    results_file = out_dir / "results.json"

    # reuse existing sweep results for the 6 already-done seeds
    cached = PREV_CACHE / f"seed_{seed}" / "results.json"
    if cached.exists() and not results_file.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(cached, results_file)

    if results_file.exists():
        print("  [cached]", flush=True)
        payload = json.loads(results_file.read_text())
        results[seed] = payload.get("metrics", payload)
        r = results[seed]
        fell = r.get("fall", True); ft = r.get("finish_time")
        ms = r.get("mean_progress_speed", r.get("mean_speed_mps", 0))
        mm = r.get("min_boundary_margin_m", 0)
        flag = " *** MARGIN WARNING ***" if float(mm) < MARGIN_HARD_LIMIT else ""
        status = "✓" if (not fell and ft) else "✗ FALL/DNF"
        print(f"  {status}  finish={ft}s  speed={float(ms):.4f}  min_margin={float(mm):.4f}m{flag}", flush=True)
        continue

    cmd = [sys.executable, "run_track_bonus.py",
           "--checkpoint-dir", str(CHECKPOINT),
           "--planner-config", str(CONFIG),
           "--output-dir", str(out_dir),
           "--duration-seconds", "300",
           "--seed", str(seed), "--no-render"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT))
    if results_file.exists():
        payload = json.loads(results_file.read_text())
        r = payload.get("metrics", payload)
        results[seed] = r
        fell = r.get("fall", True); ft = r.get("finish_time")
        ms = r.get("mean_progress_speed", r.get("mean_speed_mps", 0))
        mm = r.get("min_boundary_margin_m", 0)
        flag = " *** MARGIN WARNING ***" if float(mm) < MARGIN_HARD_LIMIT else ""
        status = "✓" if (not fell and ft) else "✗ FALL/DNF"
        print(f"  {status}  finish={ft}s  speed={float(ms):.4f}  min_margin={float(mm):.4f}m{flag}", flush=True)
    else:
        print(f"  FAILED - no results.json", flush=True)
        print(f"  STDERR: {proc.stderr[-600:]}", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"ROBUSTNESS REPORT — Run2 + 3.0/2.2  ({len(results)} seeds)", flush=True)
print(f"{'='*60}", flush=True)

all_ok = True
min_margins = []
finish_times = []
for seed in SEEDS:
    r = results.get(seed)
    if not r:
        print(f"  seed={seed:>9}: NO RESULT"); all_ok = False; continue
    fell = r.get("fall", True); ft = r.get("finish_time")
    ms = r.get("mean_progress_speed", r.get("mean_speed_mps", 0))
    mm = float(r.get("min_boundary_margin_m", 0))
    status = "✓" if (not fell and ft) else "✗ FALL"
    warn = " ⚠ LOW MARGIN" if mm < MARGIN_HARD_LIMIT else ""
    print(f"  seed={seed:>9}: {status}  finish={str(ft):>8}s  margin={mm:.4f}m{warn}", flush=True)
    if fell or not ft:
        all_ok = False
    else:
        min_margins.append(mm)
        finish_times.append(ft)

n_ok = len(finish_times)
print(f"\n  {n_ok}/{len(SEEDS)} completed", flush=True)
if finish_times:
    avg_ft  = sum(finish_times) / len(finish_times)
    min_mm  = min(min_margins)
    print(f"  avg finish:     {avg_ft:.2f}s", flush=True)
    print(f"  min margin:     {min_mm:.4f}m  (worst seed)", flush=True)
    print(f"  margin < 0.6m:  {sum(1 for m in min_margins if m < MARGIN_HARD_LIMIT)} seeds", flush=True)
    print(f"  margin < 0.8m:  {sum(1 for m in min_margins if m < 0.8)} seeds", flush=True)
    print(f"  margin < 1.0m:  {sum(1 for m in min_margins if m < 1.0)} seeds", flush=True)

print(flush=True)
if n_ok == len(SEEDS) and all(m >= MARGIN_HARD_LIMIT for m in min_margins):
    if min(min_margins) >= 0.8:
        verdict = "CONFIRMED RELIABLE — use 3.0/2.2"
    else:
        verdict = "TECHNICALLY PASSES (>0.6m) but margin tight — consider 2.5/2.1 for submission"
else:
    verdict = "UNSTABLE or MARGIN BREACH — revert to 2.5/2.1"
print(f"  VERDICT: {verdict}", flush=True)
