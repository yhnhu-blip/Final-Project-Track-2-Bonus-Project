"""Run 2 planner speed sweep: 4 configs × 6 seeds.
Strategy: 2-seed probe first; skip full 6-seed if probe fails.
Baseline: Run 2 @ 2.5/2.1 = 106.06s, min_margin 1.164m.
Acceptance: 6/6 stable + min_margin >= 0.6m + faster than 106.06s.
"""
import subprocess, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "artifacts" / "run_fastv2" / "best_checkpoint"
PROBE_SEEDS = [1000, 54321]   # historically challenging seeds
ALL_SEEDS   = [20260527, 42, 1000, 54321, 777777, 2025]
OUTBASE     = ROOT / "track_eval" / "r2_speed_sweep"

BASELINE_TIME   = 106.06
BASELINE_MARGIN = 1.164
MIN_MARGIN      = 0.6

CONFIGS = [
    ("2.8/1.9", "configs/planner_r2_280_190.json"),
    ("3.0/1.9", "configs/planner_r2_300_190.json"),
    ("2.8/2.1", "configs/planner_r2_280_210.json"),
    ("3.0/2.2", "configs/planner_r2_300_220.json"),
]

def run_one(cfg_path, seed, out_dir):
    rf = Path(out_dir) / "results.json"
    if rf.exists():
        return json.loads(rf.read_text()).get("metrics", json.loads(rf.read_text()))
    cmd = [sys.executable, "run_track_bonus.py",
           "--checkpoint-dir", str(CHECKPOINT),
           "--planner-config", str(ROOT / cfg_path),
           "--output-dir", str(out_dir),
           "--duration-seconds", "300",
           "--seed", str(seed), "--no-render"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT))
    if rf.exists():
        return json.loads(rf.read_text()).get("metrics", json.loads(rf.read_text()))
    print(f"  FAILED seed={seed}\n  STDERR: {proc.stderr[-400:]}", flush=True)
    return None

def fmt(r):
    if not r: return "FAIL"
    fell = r.get("fall", True); ft = r.get("finish_time")
    if fell or not ft: return "✗FALL"
    ms = r.get("mean_progress_speed", r.get("mean_speed_mps", 0))
    mm = r.get("min_boundary_margin_m", 0)
    return f"✓ {ft:.2f}s spd={float(ms):.3f} mgn={float(mm):.3f}m"

summary = []

for label, cfg in CONFIGS:
    tag = cfg.replace("configs/", "").replace(".json", "")
    print(f"\n{'='*56}", flush=True)
    print(f"CONFIG: {label}  ({cfg})", flush=True)

    # ── Phase 1: 2-seed probe ──
    print(f"  Phase 1 probe (seeds {PROBE_SEEDS}):", flush=True)
    probe_ok = 0
    for seed in PROBE_SEEDS:
        out = OUTBASE / tag / f"seed_{seed}"
        r = run_one(cfg, seed, out)
        ok = r and not r.get("fall") and r.get("finish_time")
        if ok: probe_ok += 1
        print(f"    seed={seed}: {fmt(r)}", flush=True)

    if probe_ok < 2:
        print(f"  → probe FAILED ({probe_ok}/2) — skip full validation", flush=True)
        summary.append((label, 0, None, None, None, "PROBE_FAIL"))
        continue

    # ── Phase 2: full 6-seed ──
    print(f"  Phase 2 full 6-seed:", flush=True)
    full_results = []
    for seed in ALL_SEEDS:
        out = OUTBASE / tag / f"seed_{seed}"
        r = run_one(cfg, seed, out)
        full_results.append((seed, r))
        print(f"    seed={seed}: {fmt(r)}", flush=True)

    n_ok = sum(1 for _, r in full_results if r and not r.get("fall") and r.get("finish_time"))
    ft_list  = [r["finish_time"]             for _, r in full_results if r and not r.get("fall") and r.get("finish_time")]
    mm_list  = [r.get("min_boundary_margin_m", 0) for _, r in full_results if r and not r.get("fall") and r.get("finish_time")]
    spd_list = [r.get("mean_progress_speed", r.get("mean_speed_mps", 0))
                for _, r in full_results if r and not r.get("fall") and r.get("finish_time")]

    avg_ft  = sum(ft_list)  / len(ft_list)  if ft_list  else None
    min_mm  = min(mm_list)                  if mm_list  else None
    avg_spd = sum(spd_list) / len(spd_list) if spd_list else None

    if n_ok == 6 and avg_ft < BASELINE_TIME and min_mm >= MIN_MARGIN:
        verdict = "WINS"
    elif n_ok == 6 and avg_ft < BASELINE_TIME and min_mm < MIN_MARGIN:
        verdict = "FASTER_UNSAFE"
    elif n_ok == 6:
        verdict = "STABLE_SLOWER"
    else:
        verdict = f"UNSTABLE_{n_ok}/6"

    print(f"  → {n_ok}/6 stable  avg={avg_ft:.2f}s  min_margin={min_mm:.4f}m  verdict={verdict}", flush=True)
    summary.append((label, n_ok, avg_ft, min_mm, avg_spd, verdict))

print(f"\n{'='*56}", flush=True)
print("FINAL SWEEP SUMMARY  (baseline Run2@2.5/2.1: 106.06s, 1.164m)", flush=True)
print(f"{'Config':<12} {'Stable':>8} {'AvgTime':>10} {'MinMargin':>11} {'AvgSpeed':>10}  Verdict", flush=True)
print("-"*70, flush=True)
best_label, best_time = None, BASELINE_TIME
for label, n_ok, avg_ft, min_mm, avg_spd, verdict in summary:
    t_str  = f"{avg_ft:.2f}s"  if avg_ft  else "—"
    mm_str = f"{min_mm:.4f}m" if min_mm else "—"
    sp_str = f"{avg_spd:.4f}"  if avg_spd else "—"
    print(f"{label:<12} {str(n_ok)+'/6':>8} {t_str:>10} {mm_str:>11} {sp_str:>10}  {verdict}", flush=True)
    if verdict == "WINS" and avg_ft is not None and avg_ft < best_time:
        best_time = avg_ft
        best_label = label

print()
if best_label:
    print(f"RECOMMENDATION: use {best_label} → avg {best_time:.2f}s (beats baseline {BASELINE_TIME:.2f}s)")
else:
    print(f"RECOMMENDATION: keep Run2 @ 2.5/2.1 ({BASELINE_TIME:.2f}s) — no faster stable config found")
