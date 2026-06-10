"""Speed sweep: test fastv1 checkpoint with increasingly aggressive planners.
Tests 2 seeds per config quickly, then full 6-seed for best candidate."""
import subprocess, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "artifacts" / "run_fastv1" / "best_checkpoint"

# Quick 2-seed probe: seeds known to be challenging
PROBE_SEEDS = [1000, 54321]
FULL_SEEDS  = [20260527, 42, 1000, 54321, 777777, 2025]

CONFIGS = [
    ("2.0/1.9 (baseline)", "configs/planner_adaptive_pd.json"),
    ("2.3/2.0",            "configs/planner_fastv1_230.json"),
    ("2.5/2.1",            "configs/planner_fastv1_250.json"),
]

def run_seed(config_path, seed, out_dir):
    results_file = Path(out_dir) / "results.json"
    if results_file.exists():
        payload = json.loads(results_file.read_text())
        return payload.get("metrics", payload)
    cmd = [
        sys.executable, "run_track_bonus.py",
        "--checkpoint-dir", str(CHECKPOINT_DIR),
        "--planner-config", str(ROOT / config_path),
        "--output-dir", str(out_dir),
        "--duration-seconds", "300",
        "--seed", str(seed),
        "--no-render",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT))
    if results_file.exists():
        payload = json.loads(results_file.read_text())
        return payload.get("metrics", payload)
    print(f"  FAILED seed={seed}")
    print("  STDERR:", proc.stderr[-500:])
    return None

probe_results = {}  # config_label -> [result, ...]

print("=== PHASE 1: 2-seed probe ===")
for label, cfg in CONFIGS:
    print(f"\n--- {label} ---")
    cfg_tag = cfg.replace("configs/", "").replace(".json", "")
    probe_results[label] = []
    for seed in PROBE_SEEDS:
        out_dir = ROOT / "track_eval" / "speed_sweep" / cfg_tag / f"seed_{seed}"
        r = run_seed(cfg, seed, out_dir)
        if r:
            fell = r.get("fall", True)
            ft = r.get("finish_time")
            ms = r.get("mean_progress_speed", r.get("mean_speed_mps", 0))
            status = "✓" if (not fell and ft) else "✗ FALL"
            print(f"  seed={seed}: {status}  finish={ft}s  speed={float(ms):.4f}")
            probe_results[label].append(r)
        else:
            probe_results[label].append(None)

print("\n=== PHASE 1 SUMMARY ===")
best_label = None
best_all_stable = False
for label, results in probe_results.items():
    n_ok = sum(1 for r in results if r and not r.get("fall") and r.get("finish_time"))
    ft_list = [r["finish_time"] for r in results if r and not r.get("fall") and r.get("finish_time")]
    avg_ft = sum(ft_list)/len(ft_list) if ft_list else None
    stable = n_ok == len(PROBE_SEEDS)
    print(f"  {label}: {n_ok}/{len(PROBE_SEEDS)} stable, avg_finish={avg_ft}")
    if stable:
        best_label = label
        best_cfg = next(c for l, c in CONFIGS if l == label)

print(f"\nBest stable config so far: {best_label}")

# Phase 2: full 6-seed on the best stable config
if best_label and best_label != CONFIGS[0][0]:  # only if it's faster than baseline
    print(f"\n=== PHASE 2: 6-seed full validation for {best_label} ===")
    cfg_tag = best_cfg.replace("configs/", "").replace(".json", "")
    full_results = {}
    for seed in FULL_SEEDS:
        out_dir = ROOT / "track_eval" / "speed_sweep" / cfg_tag / f"seed_{seed}"
        r = run_seed(best_cfg, seed, out_dir)
        if r:
            fell = r.get("fall", True)
            ft = r.get("finish_time")
            ms = r.get("mean_progress_speed", r.get("mean_speed_mps", 0))
            mm = r.get("min_boundary_margin_m", 0)
            status = "✓" if (not fell and ft) else "✗ FALL"
            print(f"  seed={seed:>8}: {status}  finish={str(ft):>8}s  speed={float(ms):.4f}  margin={float(mm):.4f}m")
            full_results[seed] = r
    n_ok = sum(1 for r in full_results.values() if not r.get("fall") and r.get("finish_time"))
    print(f"\n{n_ok}/{len(FULL_SEEDS)} stable ({best_label})")
    if n_ok == len(FULL_SEEDS):
        avg = sum(r["finish_time"] for r in full_results.values()) / len(FULL_SEEDS)
        print(f"Avg finish: {avg:.2f}s  (vs 111.62s @ 2.0/1.9, vs 127.53s baseline)")
else:
    print("\n[skipping Phase 2 — baseline is already best or fastest stable config]")
