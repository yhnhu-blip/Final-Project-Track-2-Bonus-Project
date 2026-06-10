#!/usr/bin/env python3
"""训练进度监控脚本。用法: python monitor_training.py artifacts/run_fastv1"""
import json, sys, time
from pathlib import Path

def fmt(v):
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)

def monitor(run_dir: Path):
    stage_dir = run_dir / "stage_2"
    progress_file = stage_dir / "progress_live.json"
    metrics_file  = stage_dir / "latest_metrics.json"

    if not progress_file.exists():
        print(f"[等待训练启动] {progress_file} 不存在...")
        return

    records = json.loads(progress_file.read_text())
    if not records:
        print("尚无数据")
        return

    total_steps = 20_000_000
    latest = records[-1]
    steps = latest["num_steps"]
    pct = steps / total_steps * 100
    m = latest.get("metrics", {})

    print(f"\n{'='*60}")
    print(f"进度: {steps:,} / {total_steps:,} steps  ({pct:.1f}%)")
    print(f"eval条数: {len(records)} 次评估 (每2M steps一次)")
    print()

    # 总体 reward
    reward = m.get("eval/episode_reward")
    print(f"  eval_reward (总):      {fmt(reward)}")
    print()

    # 追踪类 (越高越好)
    print("  [追踪 — 越高越好]")
    print(f"  tracking_lin_vel:  {fmt(m.get('reward/tracking_lin_vel'))}")
    print(f"  tracking_ang_vel:  {fmt(m.get('reward/tracking_ang_vel'))}")
    print(f"  feet_air_time:     {fmt(m.get('reward/feet_air_time'))}")
    print(f"  pose:              {fmt(m.get('reward/pose'))}")
    print()

    # 稳定性 (越接近0越好)
    print("  [稳定性 — 越接近0越好，负值×scale]")
    print(f"  termination:       {fmt(m.get('reward/termination'))}  ← 摔倒次数代理")
    print(f"  orientation:       {fmt(m.get('reward/orientation'))}")
    print(f"  lin_vel_z:         {fmt(m.get('reward/lin_vel_z'))}")
    print(f"  ang_vel_xy:        {fmt(m.get('reward/ang_vel_xy'))}")
    print()

    # 效率 (负值，越接近0越好)
    print("  [效率 — 越接近0越好]")
    print(f"  energy:            {fmt(m.get('reward/energy'))}")
    print(f"  action_rate:       {fmt(m.get('reward/action_rate'))}")
    print(f"  feet_slip:         {fmt(m.get('reward/feet_slip'))}")
    print()

    # 趋势 (最近3条)
    if len(records) >= 3:
        recent = [r["metrics"].get("eval/episode_reward") for r in records[-3:]]
        recent = [v for v in recent if v is not None]
        if len(recent) == 3:
            delta = recent[-1] - recent[0]
            trend = "↑ 上升" if delta > 0.01 else ("↓ 下降⚠" if delta < -0.05 else "→ 稳定")
            print(f"  最近3次reward趋势: {[f'{v:.3f}' for v in recent]} → {trend} (Δ={delta:+.4f})")

    # 发散检测
    if reward is not None:
        if reward < -1.0:
            print("\n  ⚠️  警告: eval_reward 过低，可能发散!")
        elif reward != reward:
            print("\n  ❌  NaN detected! 训练发散，需要停止!")

    print(f"{'='*60}")
    print(f"实时日志: tail -f artifacts/run_fastv1/train.log")
    print(f"完整指标: artifacts/run_fastv1/stage_2/progress_live.json")

if __name__ == "__main__":
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/run_fastv1")
    monitor(run_dir)
